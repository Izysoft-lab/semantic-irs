import re
import sys
from typing import Tuple, List, Any

import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")


def preprocess(text: str) -> Tuple[Any, List[str]]:
    """
    Preprocess a textual content before cluster building and indexing.
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


def get_ontonym(word1: str, word2: str) -> bool:
    ant = list()
    wordnet_lemmatizer = WordNetLemmatizer()
    wordlema = wordnet_lemmatizer.lemmatize(word1, pos="v")
    wordlema2 = wordnet_lemmatizer.lemmatize(word2, pos="v")
    for synset in wordnet.synsets(wordlema):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                # When antonyms are available, add them into the list
                ant.append(lemma.antonyms()[0].name())
    print(ant)
    return wordlema2 in ant


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    file.write("\n")
    file.flush()
