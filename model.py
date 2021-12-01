from scipy.spatial import distance
import numpy as np
from elasticsearch import Elasticsearch
import sys
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
#import spacy
from pymagnitude import *
from hashlib import blake2b
import random
es = Elasticsearch()
from scipy.special import softmax


class Vectorization:
    def __init__(self, clusters=[],documents=[],eps=0.30, index_name="docume_docs_final_fin"):
        self.clusters = clusters
        self.documents=documents
        self.docs =[]
        self.eps = eps
        self.vectors = Magnitude("C:/Users/paul/Downloads/mots.magnitude")
        self.es = Elasticsearch(timeout=200)
        self.val_dim=2048
        self.index_name=index_name
        if len(self.clusters)<2048:
            self.val_dim =len(self.clusters)
            
    
    def gethash(self,word):
        # cette méthode permet d'avoir id d'un document en hashant une partie du texte
        h = blake2b(digest_size=35)
        h.update(str(word).encode('utf-8'))
        return h.hexdigest()
    
    def getword(self,text):
        # Retourne les premiers cacactère d'un texte 
        if len(text)<15:
            return text[0:len(text)]
        else:
            return text[0:15]
    
    def nb_doc_incluster(self,cluster):
        # cette méthode permet de calculer le log(N/nbr) où nbr est nombre de clusters contenant ce mot
        nbr=0
        for document in self.documents:
            is_in = False
            for word in set(document["tokens"]):
                if word in cluster["words"]:
                    is_in  =True
                    break
            if is_in ==True:
                nbr +=1
        if nbr==0:
            return 0
        else:
            return np.log(len(self.clusters)/nbr)
        
    def df(self,document,cluster):
        
        #cette méthode permet de calculer le tf selon la formule définie
        tf=0
        nb_pre =0
        for word in cluster["words"]:
            val=document["tokens"].count(word)
            tf +=val
            if val>0:
                nb_pre+=1
        if len(document["tokens"])==0:
               return 0,nb_pre/len(cluster["words"])
        else:
            return np.log((tf/len(document["tokens"]))+1),nb_pre/len(cluster["words"])
    
    def get_tokens(self,text):
       # effectue l'ensemble des traitement sur du texte à l'aide d'expression regulière et retourne les tokens
        nlp = spacy.load("en_core_web_sm")
        text_trait = text
        text_trait = re.sub(r'#\S+', "", text_trait)
        text_trait = re.sub(r'@\S+', "", text_trait)
        text_trait = re.sub(r'\S*@\S*\s?', "", text_trait) 
        text_trait = re.sub(r'http\S+', "", text_trait)
        text_trait = re.sub(r'word01|word02|word03', "", text_trait)
        text_trait = re.sub(r"[^A-Za-z0-9]''", "", text_trait)
        text_trait = re.sub(f'\d+', "", text_trait)
        text_trait = re.sub(r'<[^>]*>', "", text_trait)
        text_trait = re.sub("[^A-Za-z0-9|' ']+", "", text_trait)
        doc = nlp(text_trait)
        or_per_loc  =[]
        tokens= []
        for ent in doc.ents:
        #print('_'.join(ent.text.split(' ')).lower(), ent.label_)
            if ent.label_=="PERSON" or ent.label_=="GPE" or ent.label_=="ORG":
                or_per_loc.append('_'.join(ent.text.split(' ')).lower())
        
        for token in doc:
            tokens.append(token.text.lower())
    
        tokens.extend([w.lower() for w in or_per_loc if not w.lower() in tokens])
        tokens_fin = [w.lower() for w in tokens if not w.lower() in stopwords.words('english') and len(w)>2 and w!=" " and w!="  "]
        return tokens_fin
    
    def get_vectors(self):
        # permet de terminer la représentation vectorielle des mots dans la base des clusters
        for document in self.progressbar(self.documents, "get vectors: ", 80):
            vector = np.zeros(shape=len(self.clusters))
            for index, cluster in enumerate(self.clusters):
                tf,facto=self.df(document,cluster)
                vector[index]=tf*cluster["idf"]
            document["vector"]=self.neated_vectors(vector)
            document["norm"]= np.linalg.norm(vector)
        print("Done.")
        
    def buil_documents(self,texts,ids_docs):
        # construre la structure document 
        for index, text in enumerate(self.progressbar(texts, "build documents: ", 80)):
            wordhas= self.getword(text)+str(np.array(random.sample(range(0, 500), 15)).sum())
            idhas = self.gethash(wordhas)
            document ={
                "text":text,
                "vector":None,
                "id":ids_docs[index],
                "tokens":self.get_tokens(text),
                "norm":None
                }
            self.documents.append(document)
        print("Done.")
    
    def compute_idf(self):
        # calcul idf de chaque clusters selon la formule definie
        for cluster in self.progressbar(self.clusters, "compute idf: ", 80):
            cluster["idf"]=self.nb_doc_incluster(cluster)
        print("Done.")
            
    def fit(self,docs_texts,id_docs):
        # appel les méthodes précendes pour indexer automatique les documents
        self.buil_documents(docs_texts,id_docs)
        self.compute_idf()
        self.get_vectors()
        self.get_docs()
        print("Done.")
        print("create index")
        try:
            self.create_index()
            print("Done.")
            for doc in self.progressbar(self.docs, "build docs index: ", 80):
                res = es.index(index=self.index_name, id=doc["doc_id"], body=doc)
            print("Done.")
            return self
        except:
            print("An exception accurred during de index creation")
            return self
            
        
    
    def get_docs(self):
        self.docs = [{"text":doc["text"],"vectors":doc["vector"],"doc_id":doc["id"],"norm":doc["norm"]} for doc in self.documents]
      
    def get_cluster(self,word):
        # retourne l'indice du cluster dans lequel se trouve un mot
        clustermin_index = None
        in_cluster = False
        vector = self.vectors.query(word)
        val_min = distance.cosine(self.clusters[0]["centroid"],vector)
        for index, cluster in enumerate(self.clusters):
            if word in cluster["words"]:
                clustermin_index = index
                in_cluster = True
                return in_cluster,clustermin_index, self.eps
            else:  
                dis=distance.cosine(cluster["centroid"],vector)
                if dis < self.eps and val_min >= dis:
                    clustermin_index = index
                    val_min = dis
                
        return in_cluster, clustermin_index, self.eps-val_min
    

    def buil_query(self,text):
        # indexer le requête et retourne sa forme vectorielle dans la base des clusters
        tokens = self.get_tokens(text)
        vector = np.zeros(shape=len(self.clusters))
        f = lambda x: (5/self.eps)*x
        vectors=[]
        for word in tokens:
            in_cluster, index_cluster,val = self.get_cluster(word)
            if in_cluster ==True:
                vector[index_cluster]+=6
                vectors.append({"indice":index_cluster,"val":6})    
            else:
                if index_cluster!=None:
                    vector[index_cluster]+=6
                    vectors.append({"indice":index_cluster,"val":6})
                
        return {"vectors":vectors, "norm":np.linalg.norm(vector)}
    
    
    def neated_vectors(self,vec):
        part = len(vec)//self.val_dim
        rest = len(vec)%self.val_dim
        vector = []
        for i in range(0,part):
            vector.append({"vector":vec[i*self.val_dim: i*self.val_dim+self.val_dim]})
        if rest!=0:
            val= np.zeros(self.val_dim)
            for i in range(part*self.val_dim,part*self.val_dim+rest):
                val[i-part*self.val_dim]=vec[i]
            vector.append({"vector":val})
        return vector

    
    def create_index(self):
        # cree l'index dans lequel sera stoker les clusters sous elastic search
        settings = {
        "mappings": {
               "properties" : {
                      "vectors":{
                            "type":"nested",
                            "properties":{
                                "vector":{
                                "type": "dense_vector",
                                "dims": self.val_dim
                            },
                        }
                        },
                      "doc_id" : {
                          "type" : "text"
                          },
                        "text": {
                             "type" : "text"
                         },
                       "norm": {
                             "type" : "double"
                         },
                    }
            }
        }
    
        self.es.indices.create(index=self.index_name, ignore=400, body=settings)
    
    
    def get_res_query(self,query_text):
        # pour un texte query_text retourne les 50 résultats pertinent
        from elasticsearch import Elasticsearch
        es = Elasticsearch(timeout=200)
   # print(query_text)
    
        query_vect =  self.buil_query(query_text)
    #print(query_vect["vectors"])
   
        query ={
            "query": {
                "script_score": {
                "query" : {
                    "match_all": {}
                },
                "script": {
                "id": "my_model_script",
                    "params": {
                        "query_vector":query_vect["vectors"],
                        "norm":query_vect["norm"],
                        "val_dim":self.val_dim
                }
              }
            }
          },
          "size": 50
        }
        res = es.search(index="docume_docs_final_fin", body=query)
        return res

    
    
    
        
    def progressbar(self,it, prefix="", size=60, file=sys.stdout):
        # pour la barre de progression
        count = len(it)
        def show(j):
            x = int(size*j/count)
            file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
            file.flush()        
        show(0)
        for i, item in enumerate(it):
            yield item
            show(i+1)
        file.write("\n")
        file.flush()
        
    def create_bm25_index(self,):
        # cree l'index pour bm25
        index_name = "index_bm_test"
        settings = {
        "settings": {
            "number_of_shards": 1,
            "index" : {
                "similarity" : {
                "default" : {
                    "type" : "BM25",
                    "b": 0.5,
                    "k1": 0
                    }
                }
                }
            },
        "mappings": {
               "properties" : {
                      "vectors":{
                            "type":"nested",
                            "properties":{
                                "vector":{
                                "type": "dense_vector",
                                "dims": self.val_dim
                            },
                        }
                        },
                      "doc_id" : {
                          "type" : "text"
                          },
                        "text": {
                             "type" : "text"
                         },
                    }
            }
        }

        self.es.indices.create(index=index_name, ignore=400, body=settings)
        
    def get_score(sefl,val,tab):
        for index, el in enumerate(tab):
            if el["_source"]["doc_id"]==val:
                return el["_score"], index
        return 0,50


    def get_is_in(self,val,tab):
        # cette fonction ne sert que pour les test de perfomance s'était juste pour vérifier si un réponse est correcte
        
        if val in tab:
            return True
        else:
            return False
        
    
    def model_combine(self,res_my,resbm24):
        # elle prend les résultats rétournés par notre model et celui de bm25 et les combine
        import operator
        responses=[]
        ids_response=[]
        hists_my  = res_my['hits']['hits']
        hist_my_score_num = np.array([hit["_score"] for hit in hists_my])
        a=hist_my_score_num.min()
        b=hist_my_score_num.max()
        hists  = resbm24['hits']['hits']
        hist_score_num = np.array([hit["_score"] for hit in hists])
        a1=hist_score_num.min()
        b1=hist_score_num.max()
        for hist in hists:
            hist["_score"]=a+((hist["_score"]-hist_score_num.min())*(b-a))/(hist_score_num.max()-hist_score_num.min())
   
   
        ids_valide=[]
        for index, hit in enumerate(hists_my):
            ids_response.append(hit["_source"]["doc_id"])
            if self.get_is_in(hit["_source"],ids_response)==False:
                bm_score,indice = self.get_score(hit["_source"]["doc_id"],hists)
                if hit["_score"]+bm_score==0:
                    responses.append({"_source":hit["_source"],"_score":0})
           
                else:
                    responses.append({"_source":hit["_source"],"_score":(50-index)*hit["_score"]+(50-indice)*bm_score})
            
        
   
        for index,hit in enumerate(hists):
            ids_response.append(hit["_source"]["doc_id"])
            if self.get_is_in(hit["_source"]["doc_id"],ids_response)==False:
                my_score,indice = self.get_score(hit["_source"]["doc_id"],res_my['hits']['hits'])
                if hit["_score"]+my_score==0:
                    responses.append({"_source":hit["_source"],"_score":0})
               
                else:
                    responses.append({"_source":hit["_source"],"_score":(50-index)*hit["_score"]+(50-indice)*my_score})
           
        
    
        responses.sort(key=operator.itemgetter('_score'),reverse= True)
        ids_responses=[]
        for hit in responses[0:50]:
            ids_responses.append(hit["_source"]["doc_id"])
        return ids_responses, responses[0:50]

    



    

