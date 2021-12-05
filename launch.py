import json
import random

from flask import Flask, jsonify, make_response, render_template, request
from flask_cors import CORS

from processing import Processing
from tools import *

app = Flask(__name__)
CORS(app)


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/resquest", methods=["POST"])
def predict():
    if request.method == "POST":
        data = request.get_json()
        res_ids, response = process.model.get_response(data["request"])

        results = []

        for hit in response:
            results.append(
                {
                    "id": hit["_source"]["doc_id"],
                    "text": hit["_source"]["text"],
                    "_score": hit["_score"],
                }
            )

        return make_response(jsonify(results), 200)


if __name__ == "__main__":
    import numpy as np

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
    for j in progress_bar(range(0, 2), "Computing: ", 80):
        text1 = ""
        for i in range(
            len(data["data"]) * j // 200, len(data["data"]) * (j + 1) // 200
        ):
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

    process = Processing(documents=text_docs)
    process.build_clusters()
    process.get_ids()

    for e in process.db.clusters:
        if len(e["words"]) > 2:
            print(e["words"])

    process.indexation()

    app.run(port=5002)
