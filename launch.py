from flask import Flask, jsonify, request, render_template,  make_response
from flask_cors import CORS
import json
from tools import *
import random
import numpy as np
from processing import Processing



app = Flask(__name__)
CORS(app)

@app.route('/')
def welcome():
    return render_template('index.html')
    
@app.route('/resquest', methods=['POST'])
def predict():
    if request.method == 'POST':
        val = 50
        data = request.get_json()
        print(data["request"])
        #data = {'thal':1,'ca':0, 'slope':0, 'oldpeak':2.5, 'exang':0, 'thalach':150, 'restecg':0,'fbs':1, 'chol':230, 'trestbps':140, 'cp':2, 'sex':1, 'age':60}
        print(process.db.clusters[0]["words"])
        #response = process.model.get_res_query(questions[0]["ques"])
        print(questions[0]["ques"])
        res_ids,response = process.model.get_response(data["request"])
        #print(response)
        
        results= []
        #print(response)
        
        for hit in response:
            results.append({"id":hit["_source"]["doc_id"],"text":hit["_source"]["text"],"_score":hit["_score"]})
            #print(hit["_source"]["text"])
        #print(hit["_source"]["doc_id"])
        # print(hit["_source"]["text"])
        
       
        
        return make_response(jsonify(results), 200)
    
       

if __name__ == "__main__":
    import numpy as np
    f = open('train-v2.0.json',)
    data = json.load(f)
    from hashlib import blake2b
    def gethash(word):
        h = blake2b(digest_size=35)
        h.update(str(word).encode('utf-8'))
        return h.hexdigest()   

    text=""
    questions=[]
    text_docs =[]
    docs_ids = []
    tokens = []
    org_loc_per=[]
    print(len(data["data"]))
    for j in progressbar(range(0,40), "Computing: ", 80):
        text1=""
        for i in range(len(data["data"])*j//200,len(data["data"])*(j+1)//200) :
            text1+=data["data"][i]["title"]
            for element in data["data"][i]['paragraphs'][0:20]:
                wordhas= element["context"][0:15] +str(np.array(random.sample(range(0, 500), 8)).sum())
                idhas = gethash(wordhas)
                for quest in element["qas"][0:5]:
                    questions.append({"ques":quest["question"],"res_id":idhas})
            
            text1+=element["context"]
            text_docs.append(element["context"])
            docs_ids.append(idhas)
            text1+=" "
    
    print(len(text_docs))
    process = Processing(documents=text_docs)
    process.build_clusters()
    process.get_ids()
    process.indexation()
    
    app.run(port=5002)