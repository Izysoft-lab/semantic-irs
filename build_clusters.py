import logging

from sklearn.base import BaseEstimator, ClusterMixin

from tools import get_hash

try:
    from pymagnitude import *
except:
    logging.exception("An error occurs during Py-magnitude import ...")
    from pymagnitude import *

import random

from elasticsearch import Elasticsearch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.spatial import distance


class Builder(ClusterMixin, BaseEstimator):
    def __init__(
        self,
        eps=0.5,
        *,
        metric="euclidean",
        metric_params=None,
        allwords=[],
        cluster_name_index="no_cluster_name",
        word_embedding_path="/home/user/embeddings.magnitude"
    ):
        self.eps = eps
        self.metric = metric
        self.metric_params = metric_params
        self.clusters = []
        self.vectors = Magnitude(word_embedding_path)
        self.allwords = allwords
        self.es = Elasticsearch()
        self.cluster_name_index = cluster_name_index
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def add_cluster(self, word: str, is_pronoun: bool, word_typ: str):
        """
        Adding a token in cluster list.

        :param word: Word to insert
        :param is_pronoun: whether word is pronoun or not.
        :param word_typ: The word Named Entity class or POS.
        :return:
        """
        point = {
            "word": word,
            "in_cluster": False,
            "cluster_id": None,
        }

        if len(self.clusters) == 0 or is_pronoun or word_typ == "NUM":
            wordhas = word + str(np.array(random.sample(range(0, 500), 8)).sum())
            idhas = get_hash(wordhas)
            point["cluster_id"] = idhas
            self.allwords.append(point)
            cluster = {
                "id": idhas,
                "centroid": self.vectors.query(word),
                "words": [word],
            }
            self.clusters.append(cluster)

        else:
            clustermin_index = None
            vector_word = self.vectors.query(word)
            val_min = distance.cosine(self.clusters[0]["centroid"], vector_word)
            for index, cluster in enumerate(self.clusters):
                dis = distance.cosine(cluster["centroid"], vector_word)
                if dis < self.eps and val_min >= dis:
                    clustermin_index = index
                    val_min = dis

            if clustermin_index is None:
                wordhas = word + str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = get_hash(wordhas)
                point["cluster_id"] = idhas
                self.allwords.append(point)
                cluster = {
                    "id": idhas,
                    "centroid": vector_word,
                    "words": [word],
                }
                self.clusters.append(cluster)

            else:
                if (
                    word_typ == "ADJ"
                    or word_typ == "NOUN"
                    or word_typ == "VERB"
                    or word_typ == "ADV"
                ):
                    if (
                        self.get_antonym(
                            self.clusters[clustermin_index]["words"][0], word
                        )
                        == True
                    ):
                        wordhas = word + str(
                            np.array(random.sample(range(0, 500), 8)).sum()
                        )
                        idhas = get_hash(wordhas)
                        point["cluster_id"] = idhas
                        self.allwords.append(point)
                        cluster = {
                            "id": idhas,
                            "centroid": self.vectors.query(word),
                            "words": [word],
                        }
                        self.clusters.append(cluster)
                    else:
                        self.clusters[clustermin_index]["words"].append(word)
                else:
                    self.clusters[clustermin_index]["words"].append(word)

    def get_antonym(self, word1: str, word2: str) -> bool:
        """
        Check whether two words are antonyms or not.
        This is useful since we do not want two antonyms to be in the same cluster.

        :param word1:
        :param word2:
        :return:
        """
        ant = list()
        wordlema = self.wordnet_lemmatizer.lemmatize(word1, pos="v")
        wordlema2 = self.wordnet_lemmatizer.lemmatize(word2, pos="v")
        for synset in wordnet.synsets(wordlema):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    # When antonyms are available, add them into the list
                    ant.append(lemma.antonyms()[0].name())
        return wordlema2 in ant
