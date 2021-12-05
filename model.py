import logging
from typing import Any, Tuple, List, Optional

from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from scipy.spatial import distance

from tools import pre_process, progress_bar

try:
    from pymagnitude import *
except:
    from pymagnitude import *

es = Elasticsearch()


def term_frequence(document: dict, cluster: dict) -> Tuple[float, float]:
    """
    Compute the term frequence of a cluster regarding a document.
    :param document:
    :param cluster:
    :return:
    """
    tf = 0
    nb_pre = 0
    for word in cluster["words"]:
        val = document["tokens"].count(word)
        tf += val
        if val > 0:
            nb_pre += 1
    if len(document["tokens"]) == 0:
        return 0, nb_pre / len(cluster["words"])
    else:
        return np.log((tf / len(document["tokens"])) + 1), nb_pre / len(
            cluster["words"]
        )


def get_score(doc_id, tab) -> Tuple[float, int]:
    """
    Compute a document score using its ID.
    
    :param doc_id:
    :param tab:
    :return:
    """
    for index, el in enumerate(tab):
        if el["_source"]["doc_id"] == doc_id:
            return el["_score"], index
    return 0, 50


class Vectorization:
    def __init__(
            self,
            clusters=[],
            documents=[],
            eps=0.30,
            index_name="docume_docs_final_fin",
            bm25_name="index_bm_test",
    ):
        self.clusters = clusters
        self.documents = documents
        self.bm25_name = bm25_name
        self.docs = []
        self.eps = eps
        self.vectors = Magnitude("/home/paul//mots.magnitude")
        self.es = Elasticsearch(timeout=200)
        self.val_dim = 2048
        self.index_name = index_name
        if len(self.clusters) < 2048:
            self.val_dim = len(self.clusters)

    def get_word(self, text: str) -> str:
        """
        Get the first 15 character of word to build an hash.

        :param text:
        :return:
        """
        if len(text) < 15:
            return text[0: len(text)]
        else:
            return text[0:15]

    def nb_doc_in_cluster(self, cluster: Any) -> float:
        """
        Compute IDF of a cluster words regarding the corpus : log(N/nbr).

        N is the total number of clusters and nbr the number of clusters with a word of a document.

        :param cluster:
        :return:
        """
        nbr = 0
        for document in self.documents:
            is_in = False
            for word in set(document["tokens"]):
                if word in cluster["words"]:
                    is_in = True
                    break
            if is_in:
                nbr += 1
        if nbr == 0:
            return 0
        else:
            return np.log(len(self.clusters) / nbr)

    def get_tokens(self, text: str) -> List[str]:
        """
        Get all processable tokens of a text string.

        :param text:
        :return:
        """
        doc, or_per_loc = pre_process(text)

        tokens = []
        for token in doc:
            tokens.append(token.text.lower())

        tokens.extend([w.lower() for w in or_per_loc if not w.lower() in tokens])
        tokens_fin = [
            w.lower()
            for w in tokens
            if not w.lower() in stopwords.words("english")
               and len(w) > 2
               and w != " "
               and w != "  "
        ]
        return tokens_fin

    def build_vectors(self):
        """
        Compute vector representation of documents in cluster space.
        :return:
        """
        for document in progress_bar(self.documents, "get vectors: ", 80):
            vector = np.zeros(shape=len(self.clusters))
            for index, cluster in enumerate(self.clusters):
                tf, facto = term_frequence(document, cluster)
                vector[index] = tf * cluster["idf"]
            document["vector"] = self.neated_vectors(vector)
            document["norm"] = np.linalg.norm(vector)

    def build_documents(self, texts: List[str], ids_docs: List[str]) -> None:
        """
        Build document structure of document representation for indexing purpose.

        :param texts:
        :param ids_docs:
        :return:
        """
        for index, text in enumerate(progress_bar(texts, "build documents: ", 80)):
            document = {
                "text": text,
                "vector": None,
                "id": ids_docs[index],
                "tokens": self.get_tokens(text),
                "norm": None,
            }
            self.documents.append(document)

    def compute_idf(self):
        """
        Compute IDF of clusters.

        :return:
        """
        for cluster in progress_bar(self.clusters, "compute idf: ", 80):
            cluster["idf"] = self.nb_doc_in_cluster(cluster)

    def fit(self, docs_texts: List[str], id_docs: List[str]) -> Any:
        """
        Perform the Indexing Process regarding to the architecture described in the paper.

        :param docs_texts:
        :param id_docs:
        :return:
        """
        logging.info("Launch Indexing Process ...")
        self.build_documents(docs_texts, id_docs)
        self.compute_idf()
        self.build_vectors()
        self.build_docs()

        logging.info("Semantic Document Indexing ...")
        try:
            self.create_index()
            for doc in progress_bar(self.docs, "build docs index: ", 80):
                _ = es.index(index=self.index_name, id=doc["doc_id"], body=doc)

        except:
            logging.exception("An exception occurred during index creation ...")
            return self

        try:
            logging.info("Lexical Document Indexing ...")
            self.create_bm25_index()
            for doc in progress_bar(self.docs, "build docs index: ", 80):
                _ = es.index(index=self.bm25_name, id=doc["doc_id"], body=doc)
            return self
        except:
            logging.exception("An exception accurred during de index creation")
            return self

    def build_docs(self):
        self.docs = [
            {
                "text": doc["text"],
                "vectors": doc["vector"],
                "doc_id": doc["id"],
                "norm": doc["norm"],
            }
            for doc in self.documents
        ]

    def get_cluster(self, word: str) -> Tuple[bool, Optional[int], float]:
        """
        Get Index of a word cluster.

        :param word:
        :return:
        """
        cluster_min_index = None
        in_cluster = False
        vector = self.vectors.query(word)
        val_min = distance.cosine(self.clusters[0]["centroid"], vector)
        for index, cluster in enumerate(self.clusters):
            if word in cluster["words"]:
                cluster_min_index = index
                in_cluster = True
                return in_cluster, cluster_min_index, self.eps
            else:
                dis = distance.cosine(cluster["centroid"], vector)
                if dis < self.eps and val_min >= dis:
                    cluster_min_index = index
                    val_min = dis

        return in_cluster, cluster_min_index, self.eps - val_min

    def build_query(self, query: str) -> dict:
        """
        Get the vector representation of a query.

        :param query:
        :return:
        """
        tokens = self.get_tokens(query)
        vector = np.zeros(shape=len(self.clusters))
        f = lambda x: (5 / self.eps) * x
        vectors = []
        for word in tokens:
            in_cluster, index_cluster, val = self.get_cluster(word)
            if in_cluster:
                vector[index_cluster] += 6
                vectors.append({"indice": index_cluster, "val": 6})
            else:
                if index_cluster is not None:
                    vector[index_cluster] += 6
                    vectors.append({"indice": index_cluster, "val": 6})

        return {"vectors": vectors, "norm": np.linalg.norm(vector)}

    def neated_vectors(self, vec):
        part = len(vec) // self.val_dim
        rest = len(vec) % self.val_dim
        vector = []
        for i in range(0, part):
            vector.append(
                {"vector": vec[i * self.val_dim: i * self.val_dim + self.val_dim]}
            )
        if rest != 0:
            val = np.zeros(self.val_dim)
            for i in range(part * self.val_dim, part * self.val_dim + rest):
                val[i - part * self.val_dim] = vec[i]
            vector.append({"vector": val})
        return vector

    def create_index(self):
        """
        Create an Elasticsearch index to store data.

        :return:
        """
        settings = {
            "mappings": {
                "properties": {
                    "vectors": {
                        "type": "nested",
                        "properties": {
                            "vector": {"type": "dense_vector", "dims": self.val_dim},
                        },
                    },
                    "doc_id": {"type": "text"},
                    "text": {"type": "text"},
                    "norm": {"type": "double"},
                }
            }
        }

        self.es.indices.create(index=self.index_name, ignore=400, body=settings)

    def get_res_query(self, query_text: str):

        query_vect = self.build_query(query_text)

        query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": """
                        double dot_produit = 0.0;
                        double norm_souce = 0.0;
                        double norn_query = 0.0;
                        def li = params['_source'].vectors; 
                        def nor_doc = params['_source'].norm;
            
                        for(int i=0;  i<params.query_vector.length ; i++){    
                            int part = params.query_vector[i].indice/params.val_dim;
                            int rest = params.query_vector[i].indice % params.val_dim;
                            dot_produit += li[part].vector[rest]*params.query_vector[i].val;
                        
                        }
                        if(dot_produit==0){
                        return 0
                        } 
                        return dot_produit/(nor_doc*params.norm);
        
                    """,
                        "params": {
                            "query_vector": query_vect["vectors"],
                            "norm": query_vect["norm"],
                            "val_dim": self.val_dim,
                        },
                    },
                    "min_score": 0.05,
                }
            }
        }
        res = self.es.search(index="docume_docs_final_fin", body=query)
        return res

    def create_bm25_index(
            self,
    ):
        # cree l'index pour bm25
        index_name = "index_bm_test"
        settings = {
            "settings": {
                "number_of_shards": 1,
                "index": {
                    "similarity": {"default": {"type": "BM25", "b": 0.5, "k1": 0}}
                },
            },
            "mappings": {
                "properties": {
                    "vectors": {
                        "type": "nested",
                        "properties": {
                            "vector": {"type": "dense_vector", "dims": self.val_dim},
                        },
                    },
                    "doc_id": {"type": "text"},
                    "text": {"type": "text"},
                }
            },
        }

        self.es.indices.create(index=index_name, ignore=400, body=settings)

    def get_is_in(self, val, tab):
        if val in tab:
            return True
        else:
            return False

    def model_combine(self, res_clustering: dict, res_bm25: dict) -> Tuple[List[str], List[dict]]:
        """
        Combine Clustering and bm25 results.

        :param res_clustering:
        :param res_bm25:
        :return:
        """
        import operator

        responses = []
        ids_response = []
        hists_my = res_clustering["hits"]["hits"]
        hist_my_score_num = np.array([hit["_score"] for hit in hists_my])
        a = hist_my_score_num.min()
        b = hist_my_score_num.max()
        hists = res_bm25["hits"]["hits"]
        hist_score_num = np.array([hit["_score"] for hit in hists])

        for hist in hists:
            hist["_score"] = a + ((hist["_score"] - hist_score_num.min()) * (b - a)) / (
                    hist_score_num.max() - hist_score_num.min()
            )

        for index, hit in enumerate(hists_my):
            ids_response.append(hit["_source"]["doc_id"])
            if not self.get_is_in(hit["_source"], ids_response):
                bm_score, indice = get_score(hit["_source"]["doc_id"], hists)
                if hit["_score"] + bm_score == 0:
                    responses.append({"_source": hit["_source"], "_score": 0})

                else:
                    responses.append(
                        {
                            "_source": hit["_source"],
                            "_score": (50 - index) * hit["_score"]
                                      + (50 - indice) * bm_score,
                        }
                    )

        for index, hit in enumerate(hists):
            ids_response.append(hit["_source"]["doc_id"])
            if not self.get_is_in(hit["_source"]["doc_id"], ids_response) :
                my_score, indice = get_score(
                    hit["_source"]["doc_id"], res_clustering["hits"]["hits"]
                )
                if hit["_score"] + my_score == 0:
                    responses.append({"_source": hit["_source"], "_score": 0})

                else:
                    responses.append(
                        {
                            "_source": hit["_source"],
                            "_score": (50 - index) * hit["_score"]
                                      + (50 - indice) * my_score,
                        }
                    )

        responses.sort(key=operator.itemgetter("_score"), reverse=True)
        ids_responses = []
        for hit in responses[0:50]:
            ids_responses.append(hit["_source"]["doc_id"])
        return ids_responses, responses[0:50]

    def query_clustering_model(self, query: str):
        res = self.get_res_query(query)
        ids_response = []
        for hit in res["hits"]["hits"]:
            ids_response.append(hit["_source"]["doc_id"])

        return ids_response, res

    def queyr_bm25_model(self, query: str) -> Tuple[List[str], dict]:
        """
        Query the clustering model and retrieve relevant documents.

        :param query:
        :return:
        """
        query = {
            "query": {"match": {"text": query}},
            "size": 50,
        }
        res = self.es.search(index="index_bm_test", body=query)
        ids_response = []
        for hit in res["hits"]["hits"]:
            ids_response.append(hit["_source"]["doc_id"])

        return ids_response, res

    def get_response(self, query: str) -> Tuple[List[str], dict]:
        (
            res_clustering_id,
            res_clustering,
        ) = self.query_clustering_model(query)
        res_bm25_id, res_bm25 = self.queyr_bm25_model(query)
        res_comb, com_resp = self.model_combine(res_clustering, res_bm25)
        return res_comb, com_resp
