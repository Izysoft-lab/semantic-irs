import logging
import random

from elasticsearch import Elasticsearch
from nltk.corpus import stopwords

from build_clusters import Builder
from model import Vectorization
from tools import get_hash, pre_process, progress_bar

try:
    from pymagnitude import *
except:
    from pymagnitude import *


class Processing:
    def __init__(
        self,
        timeout=200,
        documents=[],
        eps=0.30,
        index_name="docume_docs_final_fin",
        magnitue_path="/home/paul//mots.magnitude",
    ):
        self.clusters = []
        self.documents = documents
        self.eps = eps
        self.vectors = Magnitude(magnitue_path)
        self.es = Elasticsearch(timeout=timeout)
        self.val_dim = 2048
        self.index_name = index_name
        self.tab_tokens = []
        self.questions = []
        self.text_docs = []
        self.docs_ids = []
        self.tokens = []
        self.org_loc_per = []
        self.db = Builder(
            eps=self.eps,
            metric="cosine",
            allwords=[],
            cluster_name_index=self.index_name,
        )
        self.model = None

    def data_division(self):
        """
        Split documents so that the max token size remains under 100000 with is the max sentence size for NER.
        :return:
        """
        texts = ""
        for doc in self.documents:
            texts += " " + doc

        val = len(texts) // 50000
        for i in range(0, val - 1):
            self.tab_tokens.append(texts[i * 50000 : (i + 1) * 50000])
        self.tab_tokens.append(texts[val * 50000 : len(texts)])

    def build_clusters(self) -> None:
        """
        Gathering tokens of the vocabulary into clusters.
        :return:
        """
        logging.info("Splitting documents ...")
        self.data_division()

        logging.info("Starting word clustering ...")
        for text_old in progress_bar(self.tab_tokens, "Computing: ", 80):
            tokens_old, org_loc_per_old = pre_process(text_old)
            self.tokens.extend(tokens_old)
            self.org_loc_per.extend(org_loc_per_old)
        tokens_add = []

        for word in progress_bar(self.tokens, "Computing: ", 80):
            if (
                word.text.lower() not in tokens_add
                and len(word.text) > 2
                and word.text.lower() not in stopwords.words("english")
                and word.text.lower() not in self.org_loc_per
            ):
                self.db.add_cluster(word.text.lower(), word.pos_ == "PROPN", word.pos_)
                tokens_add.append(word.text.lower())

        data_set = list(set(self.org_loc_per))
        for word in progress_bar(
            data_set[0 : len(set(self.org_loc_per))], "Computing: ", 80
        ):
            if len(word) > 2 and word not in stopwords.words("english"):
                self.db.add_cluster(word, True, "AVION")

        logging.info("End of word Clustering ...")

    def get_ids(self):
        """
        Build documents IDs using te Has word function.
        :return:
        """
        for doc in self.documents:
            wordhas = doc[0 : len(doc) // 10] + str(
                np.array(random.sample(range(0, 500), 8)).sum()
            )
            idhas = get_hash(wordhas)
            self.docs_ids.append(idhas)

    def indexation(self):
        """
        Building the Vector representation of documents in the cluster space.
        :return:
        """
        self.model = Vectorization(documents=[], clusters=self.db.clusters).fit(
            self.documents, self.docs_ids
        )
