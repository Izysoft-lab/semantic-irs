import re
import sys
from hashlib import blake2b
from typing import Any, List, Tuple

import spacy

nlp = spacy.load("en_core_web_sm")


def pre_process(text: str) -> Tuple[Any, List[str]]:
    """
    Pre-process a textual content before cluster building and indexing.
    :param text:
    :return:
    """
    processed_text = text
    processed_text = re.sub(r"#\S+", "", processed_text)
    processed_text = re.sub(r"@\S+", "", processed_text)
    processed_text = re.sub(r"\S*@\S*\s?", "", processed_text)
    processed_text = re.sub(r"http\S+", "", processed_text)
    processed_text = re.sub(r"word01|word02|word03", "", processed_text)
    processed_text = re.sub(r"[^A-Za-z0-9]''", "", processed_text)
    processed_text = re.sub(f"\d+", "", processed_text)
    processed_text = re.sub(r"<[^>]*>", "", processed_text)
    processed_text = re.sub("[^A-Za-z0-9|' ']+", "", processed_text)
    doc = nlp(processed_text)
    or_per_loc = []
    for ent in doc.ents:
        if ent.label_ == "PERSON" or ent.label_ == "GPE" or ent.label_ == "ORG":
            or_per_loc.append("_".join(ent.text.split(" ")).lower())

    return doc, or_per_loc


def progress_bar(
    iterations: Any, prefix: str = "", size: int = 60, file: Any = sys.stdout
) -> None:
    """
    A function to display the progress bar related to a process.

    :param iterations:
    :param prefix:
    :param size:
    :param file:
    :return:
    """
    count = len(iterations)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(iterations):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()


def get_hash(word: str) -> str:
    """
    Build Hash of a given word.

    :param word:
    :return:
    """
    h = blake2b(digest_size=35)
    h.update(str(word).encode("utf-8"))
    return h.hexdigest()
