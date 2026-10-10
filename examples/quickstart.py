"""Minimal end-to-end example. Needs a GPU and a Hugging Face token with access to N-ATLaS.

export HF_TOKEN=hf_...
python examples/quickstart.py path/to/image.jpg
"""

import sys

from testatlasvision import GenerationConfig, TestAtlasVision


def main(image_path: str) -> None:
    model = TestAtlasVision()  # load once; takes minutes the first time

    print("Caption   :", model.describe(image_path))
    print("Question  :", model.ask(image_path, "How many animals are in this image?"))

    result = model.chat(image_path, "Kedu ihe dị na foto a?", lang="ig", config=GenerationConfig(max_new_tokens=150))
    print("Igbo      :", result.answer)
    print("(English) :", result.english_answer)

    session = model.new_session()  # follow-ups
    model.ask(image_path, "What is the main object?", session=session)
    print("Follow-up :", model.ask(image_path, "What colour is it?", session=session))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
