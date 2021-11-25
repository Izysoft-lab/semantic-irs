from tools import traitement
from tools import progressbar
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from dbscan import DBSCAN
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch
from pymagnitude import *
from model import Vectorization
from hashlib import blake2b
import random



class Processing:
    def __init__(self, timeout =20, documents=[],eps=0.30,index_name="docume_docs_final_fin",magnitue_path="C:/Users/paul/Downloads/mots.magnitude"):
        self.clusters = []
        self.documents=documents
        self.eps = eps
        self.vectors = Magnitude(magnitue_path)
        self.es = Elasticsearch(timeout=timeout)
        self.val_dim=2048
        self.index_name=index_name
        self.tab_tokens = []
        self.questions=[]
        self.text_docs =[]
        self.docs_ids = []
        self.tokens = []
        self.org_loc_per=[]
        self.docs_ids = []
        self.db = DBSCAN(eps=self.eps, min_samples=2, metric='cosine',allwords=[],cluster_name_index=self.index_name)
        
        
    def preprocessing(self):
        texts = ""
        for doc in self.documents:
            texts+=" "+doc

        val = len(texts)//50000
        for i in range(0,val-1):
            self.tab_tokens.append(texts[i*50000:(i+1)*50000])
        self.tab_tokens.append(texts[val*50000:len(texts)])
    

    
    def build_clusters(self):
        for text_old in self.tab_tokens:
            tokens_old,org_loc_per_old = traitement(text_old)
            self.tokens.extend(tokens_old)
            self.org_loc_per.extend(org_loc_per_old)
        tokens_add =[]
       
        for word in progressbar(self.tokens, "Computing: ", 80):
            if word.text.lower() not in tokens_add and len(word.text)>2 and word.text.lower() not in stopwords.words('english') and word.text.lower() not in org_loc_per:
                self.db.add_cluster(word.text.lower(), word.pos_=="PROPN",word.pos_)
                tokens_add.append(word.text.lower())
                
       

        tokens_add =[]
        data_set =list(set(self.org_loc_per))
        for word in progressbar(data_set[0:len(set(self.org_loc_per))], "Computing: ", 80):
            if len(word)>2 and word not in stopwords.words('english'):
                self.db.add_cluster(word, True,"AVION")
      
        print("Done.")
    
    def gethash(self,word):
        h = blake2b(digest_size=35)
        h.update(str(word).encode('utf-8'))
        return h.hexdigest() 
        
    def get_ids(self):
        for doc in self.documents:
            wordhas= doc[0:len(doc)//10]+str(np.array(random.sample(range(0, 500), 8)).sum())
            idhas = self.gethash(wordhas)
            self.docs_id.append(idhas)
    
    
    def indexation(self):
        model = Vectorization(documents=[],clusters=db.clusters).fit(self.documents, self.docs_id)
        
