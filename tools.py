import re
import spacy
from nltk.corpus import wordnet  
from nltk.stem import WordNetLemmatizer
nlp = spacy.load("en_core_web_sm")

def traitement(text):
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
    for ent in doc.ents:
    #print('_'.join(ent.text.split(' ')).lower(), ent.label_)
        if ent.label_=="PERSON" or ent.label_=="GPE" or ent.label_=="ORG":
            or_per_loc.append('_'.join(ent.text.split(' ')).lower())
    
    return doc, or_per_loc




def get_ontonym(word1,word2):
    ant = list()
    wordnet_lemmatizer = WordNetLemmatizer()
    wordlema = wordnet_lemmatizer.lemmatize(word1, pos="v")
    wordlema2 = wordnet_lemmatizer.lemmatize(word2, pos="v")
    for synset in wordnet.synsets(wordlema):
        for lemma in synset.lemmas():
            if lemma.antonyms(): 
                #When antonyms are available, add them into the list
                ant.append(lemma.antonyms()[0].name())
    print(ant)
    return wordlema2 in ant






import sys

def progressbar(it, prefix="", size=60, file=sys.stdout):
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
    
    
    
    




