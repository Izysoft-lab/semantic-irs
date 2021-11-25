import numpy as np
import warnings
from scipy import sparse

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import _check_sample_weight, _deprecate_positional_args
from sklearn.neighbors import NearestNeighbors

from sklearn.cluster._dbscan_inner import dbscan_inner
import numpy as np
from nltk.corpus import wordnet as wn
from pymagnitude import *
from scipy.spatial import distance
from hashlib import blake2b
import random
from datetime import datetime
from elasticsearch import Elasticsearch
from nltk.corpus import wordnet   #Import wordnet from the NLTK
from nltk.stem import WordNetLemmatizer
    

@_deprecate_positional_args
def dbscan(X, eps=0.5, *, min_samples=5, metric='minkowski',
           metric_params=None, algorithm='auto', leaf_size=30, p=2,
           sample_weight=None, n_jobs=None):
    

    est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm,
                 leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class DBSCAN(ClusterMixin, BaseEstimator):
    
    @_deprecate_positional_args
    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None,allwords=[],cluster_name_index="no_cluster_name"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.clusters = []
        self.vectors = Magnitude("C:/Users/paul/Downloads/mots.magnitude")
        self.allwords =allwords
        self.es = Elasticsearch()
        self.cluster_name_index=cluster_name_index
        self.wordnet_lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None, sample_weight=None):
       
        X = self._validate_data(X, accept_sparse='csr')

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        neighbors_model = NearestNeighbors(
            radius=self.eps, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric,
            metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X,
                                                         return_distance=False)
        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)
        
        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
      
        self.fit(X, sample_weight=sample_weight)
        return self.labels_
    
    def get_words(self,rarray,pos):
        """
        Convert a NdArray in List
        :param ndrarray:
        :return:
        """
        l = []
        for e in rarray:
            self.allwords[e]["cluster_id"]=pos
            self.allwords[e]["in_cluster"]=True
            l.append(self.allwords[e]["word"])
        return l
    
    def build_clusters(self):
        
        for pos in set(self.labels_):
            if pos!=-1:
                pos_list = (np.where(self.labels_==pos)[0]).tolist()
                wordhas= self.allwords[pos_list[0]]["word"]+str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = self.gethash(wordhas)
                words = self.get_words(pos_list,idhas)
                cluster = {
                    'id': idhas,
                    'centroid': np.mean(self.vectors.query(words),axis=0),
                    'words': words,
                    }
                self.clusters.append(cluster)
                self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
            else:
                for index, e in enumerate(np.where(self.labels_==pos)[0]):
                    words = [self.allwords[e]["word"]]
                    wordhas= words[0]+str(np.array(random.sample(range(0, 500), 8)).sum())
                    idhas = self.gethash(wordhas)
                    self.allwords[e]["cluster_id"] = idhas
                    self.allwords[e]["in_cluster"]=False
                    cluster = {
                    'id': idhas,
                    'centroid': self.vectors.query(words[0]),
                    'words': words,
                     }
                    self.clusters.append(cluster)
                    self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
                    
    def add_word(self,word):
        point ={
        "word":word,
        "in_cluster":False,
        "cluster_id":None,
        }
        n_neighbors = self.epsilonVoisinage(point)
        
       
        if len(n_neighbors)==0:
            wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
            idhas = self.gethash(wordhas)
            point["cluster_id"] = idhas
            self.allwords.append(point)
            cluster = {
             'id': idhas,
             'centroid': self.vectors.query(word),
              'words': [word],
                }
            self.clusters.append(cluster)
            self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
            
        else:  
            on_include = False
            for neighbor in n_neighbors:
                if neighbor["in_cluster"] ==True:
                    on_include = True
                    break
            
            if on_include == False:
                words = []
                wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = self.gethash(wordhas)
                for neighbor in n_neighbors:
                    words.append(neighbor["word"])
                    self.es.delete(index=self.cluster_name_index, id=neighbor["cluster_id"])
                    self.delete_cluster(neighbor["cluster_id"])
                    self.allwords[self.allwords.index(neighbor)]["cluster_id"]=idhas 
                    self.allwords[self.allwords.index(neighbor)]["in_cluster"]=True
                    
                         
                words.append(word)
                point["cluster_id"] =idhas
                point["in_cluster"] =True
                self.allwords.append(point)
                cluster = {
                'id': idhas,
                'centroid': np.mean(self.vectors.query(words),axis=0),
                'words': words,
                }
                self.clusters.append(cluster)
                self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
                
            else:
                words =[]
                idhas = n_neighbors[0]["cluster_id"]
                for neighbor in n_neighbors:
                    cluster = self.get_cluster(neighbor["cluster_id"])
                    if cluster!=-1:   
                        words.extend(cluster["words"])
                        self.es.delete(index=self.cluster_name_index, id=cluster["id"])
                        self.delete_cluster(cluster["id"])
                        
                    
                self.update_id(words,idhas) 
                words.append(word)  
                point["cluster_id"] =idhas
                point["in_cluster"] =True
                self.allwords.append(point)
                cluster = {
                    'id': idhas,
                    'centroid': np.mean(self.vectors.query(words),axis=0),
                    'words': words,
                    }
                self.clusters.append(cluster)
                self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
                
                
    def delete_cluster(self,cluster_id):
        
        for cluster in self.clusters:
            
            if cluster['id']== cluster_id:
                del self.clusters[self.clusters.index(cluster)]
                break
                 
    def get_cluster(self,cluster_id):
        
        for cluster in self.clusters:
            if cluster["id"]==cluster_id:
                return cluster
        return -1
            
    def update_id(self,words,id):
        for i in range(0,len(self.allwords)):
            if self.allwords[i]["word"] in words:
                self.allwords[i]["cluster_id"]=id
                self.allwords[i]["in_cluster"]=True
    
    def merge(self, n_neighbors, word):
        print("bonjour")
        
        
    def epsilonVoisinage(self, P):
        voisins = [];
        for e in self.allwords:
            if e["word"]==P["word"]:
                continue
            if distance.cosine(self.vectors.query(e["word"]),self.vectors.query(P["word"]))< self.eps:
                voisins.append(e)
        
        return voisins
    
    def gethash(self,word):
        h = blake2b(digest_size=35)
        h.update(str(word).encode('utf-8'))
        return h.hexdigest()   
    
    def add_cluster(self,word,is_pronom,word_typ):
        point ={
            "word":word,
            "in_cluster":False,
            "cluster_id":None,
            }
        
         
        if len(self.clusters)==0 or is_pronom==True or word_typ=="NUM":
            wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
            idhas = self.gethash(wordhas)
            point["cluster_id"] = idhas
            self.allwords.append(point)
            cluster = {
             'id': idhas,
             'centroid': self.vectors.query(word),
              'words': [word],
                }
            self.clusters.append(cluster)
            # self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
            
        else:
            clustermin_index = None
            vector_word = self.vectors.query(word)
            val_min = distance.cosine(self.clusters[0]["centroid"],vector_word)
            for index, cluster in enumerate(self.clusters):
                dis=distance.cosine(cluster["centroid"],vector_word)
                if dis < self.eps and val_min >= dis:
                    clustermin_index = index
                    val_min = dis

            
            if clustermin_index == None:
                wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = self.gethash(wordhas)
                point["cluster_id"] = idhas
                self.allwords.append(point)
                cluster = {
                'id': idhas,
                'centroid': vector_word,
                'words': [word],
                    }
                self.clusters.append(cluster)
               # self.es.index(index=self.cluster_name_index, id=cluster["id"], body=cluster)
                
            else:
                if word_typ=="ADJ" or word_typ=="NOUN" or word_typ=="VERB" or word_typ=="ADV":
                    if self.get_ontonym(self.clusters[clustermin_index]["words"][0],word)==True:
                        wordhas= word+str(np.array(random.sample(range(0, 500), 8)).sum())
                        idhas = self.gethash(wordhas)
                        point["cluster_id"] = idhas
                        self.allwords.append(point)
                        cluster = {
                         'id': idhas,
                         'centroid': self.vectors.query(word),
                          'words': [word],
                            }
                        self.clusters.append(cluster)
                    else:
                        self.clusters[clustermin_index]["words"].append(word)
                else:
                    self.clusters[clustermin_index]["words"].append(word)
                
               # self.es.index(index=self.cluster_name_index, id=self.clusters[clustermin_index]["id"], body=self.clusters[clustermin_index])
    def get_ontonym(self,word1,word2):
        ant = list()
        wordlema = self.wordnet_lemmatizer.lemmatize(word1, pos="v")
        wordlema2 = self.wordnet_lemmatizer.lemmatize(word2, pos="v")
        for synset in wordnet.synsets(wordlema):
            for lemma in synset.lemmas():
                if lemma.antonyms(): 
                #When antonyms are available, add them into the list
                    ant.append(lemma.antonyms()[0].name())
        return wordlema2 in ant

    
    def get_mean(self,u,v,longeur):
        c = longeur*u
        f = (c+v)/(longeur+1)
        return f
    
    
    