import json
import random
from collections import Counter

import numpy as np

from processing import Processing
from tools import *

f = open(
    "train-v2.0.json",
)
data = json.load(f)
from hashlib import blake2b


def gethash(word):
    h = blake2b(digest_size=35)
    h.update(str(word).encode("utf-8"))
    return h.hexdigest()


text = ""
questions = []
text_docs = []
docs_ids = []
tokens = []
org_loc_per = []
print(len(data["data"]))
for j in progress_bar(range(0, 2), "Computing: ", 80):
    text1 = ""
    for i in range(len(data["data"]) * j // 200, len(data["data"]) * (j + 1) // 200):
        text1 += data["data"][i]["title"]
        for element in data["data"][i]["paragraphs"][0:20]:
            wordhas = element["context"][0:15] + str(
                np.array(random.sample(range(0, 500), 8)).sum()
            )
            idhas = gethash(wordhas)
            for quest in element["qas"][0:5]:
                questions.append({"ques": quest["question"], "res_id": idhas})

            text1 += element["context"]
            text_docs.append(element["context"])
            docs_ids.append(idhas)
            text1 += " "

print(len(text_docs))
process = Processing(documents=text_docs)
process.build_clusters()
process.get_ids()
print(len(process.docs_ids))
print(len(process.documents))
for e in process.db.clusters:
    if len(e["words"]) > 2:
        print(e["words"])
print(process.db.clusters[0]["words"])
process.indexation()
