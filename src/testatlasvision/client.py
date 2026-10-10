"""The TestAtlasVision client.

Heavy libraries (torch, transformers, peft) are imported lazily inside methods, so
`import testatlasvision` is cheap and works on machines with no GPU.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from dataclasses import replace
from typing import Optional, Union

from .auth import ensure_llm_access, mask_token, resolve_token
from .config import (
    DEFAULT_LLM,
    DEFAULT_REPO,
    DEFAULT_STAGE,
    DEFAULT_VISION,
    EOT,
    LANGUAGES,
    USER_HEADER,
    GenerationConfig,
    Stage,
    normalize_language,
)
from .exceptions import AtlasVisionError, ModelLoadError
from .images import ImageInput, load_image
from .prompting import build_conversation_text, build_text_prompt, build_translation_prompt
from .types import ChatResult, ModelInfo, Session, Turn

logger = logging.getLogger("testatlasvision")

_DEFAULT_ASK_TOKENS = 256
_DEFAULT_TEXT_TOKENS = 512


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


class TestAtlasVision:
    """Multilingual visual question answering.

    Loads SigLIP2 (vision), N-ATLaS (language, gated) and the AtlasVision projector/LoRA.
    Load ONCE per process (it takes minutes); the object is safe to share across threads,
    but generation is serialised by an internal lock because there is one GPU and the
    LoRA adapter is toggled globally.

    Parameters
    ----------
    token : Hugging Face token. If omitted, see `testatlasvision.auth` for the lookup order.
    repo, llm, vision : Hub ids (or local dirs) for the AtlasVision weights, base LLM, vision encoder.
    stage : "2b" (recommended), "2", or "1" (captioner, no LoRA).
    load_in_4bit : None = decide from GPU memory (<20 GB -> 4-bit).
    device : e.g. "cuda", "cuda:1", "cpu". None = auto.
    interactive_login : prompt for a token if none is found.
    allow_remote_images : let `image` be an http(s) URL. Turn OFF in a public API.
    check_access : verify gated-model access before the big downloads.
    """

    # Tell pytest this is not a test class, even though the name starts with "Test".
    __test__ = False

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        repo: str = DEFAULT_REPO,
        llm: str = DEFAULT_LLM,
        vision: str = DEFAULT_VISION,
        stage: Union[Stage, str] = DEFAULT_STAGE,
        load_in_4bit: Optional[bool] = None,
        device: Optional[str] = None,
        interactive_login: bool = False,
        allow_remote_images: bool = True,
        check_access: bool = True,
    ):
        try:
            self.stage = Stage(str(getattr(stage, "value", stage)))
        except ValueError:
            raise ValueError(f"Unknown stage {stage!r}. Use one of: {[s.value for s in Stage]}") from None

        self._repo, self._llm_id, self._vision_id = repo, llm, vision
        self._allow_remote_images = allow_remote_images
        self._lock = threading.Lock()
        self._token = resolve_token(token, interactive=interactive_login)
        logger.info("Hugging Face token: %s", mask_token(self._token))

        if check_access:
            ensure_llm_access(llm, self._token)

        try:
            self._load(load_in_4bit, device)
        except AtlasVisionError:
            raise
        except ImportError as e:
            raise ModelLoadError(
                f"Missing dependency ({e.name or e}). Install the SDK dependencies; "
                "4-bit loading needs `pip install testatlasvision[4bit]`."
            ) from e
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load models: {type(e).__name__}: {e}") from e

    # ------------------------------------------------------------------ loading

    def _load(self, load_in_4bit: Optional[bool], device: Optional[str]) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

        from .projector import ProjectionMLP

        tok = self._token
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cuda = self._device.type == "cuda"
        gpu_gb = torch.cuda.get_device_properties(self._device).total_memory / 1e9 if cuda else 0.0

        if load_in_4bit is None:
            load_in_4bit = cuda and gpu_gb < 20
        if load_in_4bit and not cuda:
            raise ModelLoadError("4-bit loading needs a CUDA GPU.")
        self._load_in_4bit = bool(load_in_4bit)
        self._dtype = torch.bfloat16 if (not cuda or torch.cuda.is_bf16_supported()) else torch.float16
        logger.info(
            "device=%s gpu=%.0fGB dtype=%s 4bit=%s stage=%s",
            self._device,
            gpu_gb,
            self._dtype,
            self._load_in_4bit,
            self.stage.value,
        )

        # Tokenizer
        self._tok = AutoTokenizer.from_pretrained(self._llm_id, token=tok)
        self._pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else self._tok.eos_token_id
        self._eot_id = self._tok.convert_tokens_to_ids(EOT)

        # Vision encoder (vision tower only)
        self._processor = AutoImageProcessor.from_pretrained(self._vision_id, token=tok)
        full_vision = AutoModel.from_pretrained(self._vision_id, dtype=self._dtype, token=tok)
        self._vision = full_vision.vision_model.to(self._device).eval()
        vision_dim = full_vision.config.vision_config.hidden_size
        del full_vision

        # Language model
        kw = {"dtype": self._dtype, "token": tok}
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self._dtype
            )
            kw["device_map"] = {"": self._device.index or 0}
        self._llm = AutoModelForCausalLM.from_pretrained(self._llm_id, **kw)
        if not self._load_in_4bit:
            self._llm.to(self._device)
        text_dim = self._llm.config.hidden_size

        # LoRA adapter (stages 2 / 2b)
        if self.stage is not Stage.CAPTIONER:
            from peft import PeftModel

            self._llm = PeftModel.from_pretrained(
                self._llm, self._repo, subfolder=f"stage{self.stage.value}/lora_adapter", token=tok
            )
        self._llm.eval()

        # Projector
        rel = f"stage{self.stage.value}/projector.safetensors"
        path = (
            os.path.join(self._repo, rel) if os.path.isdir(self._repo) else hf_hub_download(self._repo, rel, token=tok)
        )
        self._projector = ProjectionMLP(vision_dim, text_dim)
        self._projector.load_state_dict(load_file(path))
        self._projector.to(self._device, dtype=torch.float32).eval()

        self._prefix = torch.tensor([self._tok(USER_HEADER, add_special_tokens=True).input_ids], device=self._device)

    # --------------------------------------------------------------- generation

    def _generate(self, cfg: GenerationConfig, **inputs):
        kw = dict(
            max_new_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
            eos_token_id=self._eot_id,
            pad_token_id=self._pad_id,
        )
        if cfg.temperature > 0:
            kw.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            kw.update(do_sample=False)
        return self._llm.generate(**inputs, **kw)

    def _adapter_disabled(self):
        return self._llm.disable_adapter() if hasattr(self._llm, "disable_adapter") else contextlib.nullcontext()

    # ----------------------------------------------------------------- public API

    def ask(
        self,
        image: ImageInput,
        question: str,
        *,
        session: Optional[Session] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """One English question about an image. Pass a `Session` to enable follow-ups."""
        import torch

        question = _require_text(question, "question")
        cfg = (config or GenerationConfig()).resolved(_DEFAULT_ASK_TOKENS)
        history = session.history if session is not None else []
        pil = load_image(image, allow_remote=self._allow_remote_images)

        with self._lock, torch.no_grad():
            pv = self._processor(images=pil, return_tensors="pt").pixel_values.to(self._device, self._dtype)
            img = self._projector(self._vision(pixel_values=pv).last_hidden_state.float()).to(self._dtype)
            text = build_conversation_text(history, question)
            ids = torch.tensor([self._tok(text, add_special_tokens=False).input_ids], device=self._device)
            emb = self._llm.get_input_embeddings()
            # [BOS + user header] [196 image embeddings] [text]
            embeds = torch.cat([emb(self._prefix).to(self._dtype), img, emb(ids).to(self._dtype)], dim=1)
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=self._device)
            out = self._generate(cfg, inputs_embeds=embeds, attention_mask=mask)

        answer = self._tok.decode(out[0], skip_special_tokens=True).strip()
        if session is not None:
            session.history.append(Turn(question, answer))
        return answer

    def text(self, prompt: str, *, config: Optional[GenerationConfig] = None) -> str:
        """Plain N-ATLaS: LoRA off, no image. Used for translation and any text task."""
        import torch

        prompt = _require_text(prompt, "prompt")
        cfg = (config or GenerationConfig()).resolved(_DEFAULT_TEXT_TOKENS)
        ids = self._tok(build_text_prompt(prompt), add_special_tokens=True, return_tensors="pt").input_ids
        ids = ids.to(self._device)
        with self._lock, torch.no_grad(), self._adapter_disabled():
            out = self._generate(cfg, input_ids=ids, attention_mask=torch.ones_like(ids))
        return self._tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    def translate(
        self,
        text: str,
        target: str,
        source: str = "en",
        *,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Translate between en / ig / yo / ha (codes or names)."""
        text = _require_text(text, "text")
        t, s = normalize_language(target), normalize_language(source)
        if t == s:
            return text
        prompt = build_translation_prompt(text, LANGUAGES[t], LANGUAGES[s])
        return self.text(prompt, config=config)

    def chat(
        self,
        image: ImageInput,
        question: str,
        *,
        lang: str = "en",
        session: Optional[Session] = None,
        translate_question: bool = True,
        config: Optional[GenerationConfig] = None,
    ) -> ChatResult:
        """Ask in English, Igbo, Yoruba or Hausa and get the answer in the same language.

        Non-English flow is a cascade: translate question -> English, answer with the vision
        model, translate answer back. That is up to 3 generations, so expect ~3x latency.
        Set `translate_question=False` if the question is already English.
        """
        code = normalize_language(lang)
        cfg = config or GenerationConfig()
        q_en = question
        if code != "en" and translate_question:
            q_en = self.translate(question, "en", source=code, config=cfg)
        a_en = self.ask(image, q_en, session=session, config=cfg)
        if code == "en":
            answer = a_en
        else:
            # Translating back needs more tokens than the English answer (Igbo/Yoruba/Hausa split into
            # more tokens), so give it double the caller's budget, as the original Colab did.
            budget = (cfg.max_new_tokens or _DEFAULT_ASK_TOKENS) * 2
            back = replace(cfg, max_new_tokens=budget)
            answer = self.translate(a_en, code, source="en", config=back)
        return ChatResult(answer=answer, language=code, english_question=q_en, english_answer=a_en)

    def describe(
        self,
        image: ImageInput,
        *,
        lang: str = "en",
        detailed: bool = True,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """Caption an image, optionally in another language."""
        q = "Describe this image in detail." if detailed else "Describe this image briefly."
        return self.chat(image, q, lang=lang, translate_question=False, config=config).answer

    # --------------------------------------------------------------------- misc

    def new_session(self) -> Session:
        return Session()

    def info(self) -> ModelInfo:
        return ModelInfo(
            stage=self.stage.value,
            llm=self._llm_id,
            vision=self._vision_id,
            repo=self._repo,
            device=str(self._device),
            dtype=str(self._dtype).replace("torch.", ""),
            load_in_4bit=self._load_in_4bit,
            supported_languages=dict(LANGUAGES),
        )

    def close(self) -> None:
        """Release GPU memory."""
        import gc

        import torch

        for name in ("_llm", "_vision", "_projector"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"TestAtlasVision(stage={self.stage.value!r}, llm={self._llm_id!r}, token={mask_token(self._token)})"
