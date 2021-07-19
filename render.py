from flask import Flask, request, render_template
from numpy import vectorize
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

app = Flask(__name__)
@app.route('/')
def form_open():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def my_form_post():
    text = request.form['inptext']
    res = manual_testing(text)
    return render_template('index.html',Result=res)

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    ff = 'finalized_model.sav'
    classifier = pickle.load(open(ff, 'rb'))
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    vectorization = pickle.load(open("vectorizer.pickle", 'rb'))
    new_xv_test = vectorization.transform(new_x_test)
    
    pred_MNNB = classifier.predict(new_xv_test)
    res = output_lable(pred_MNNB[0])
    return res
    
if __name__ == '__main__':
    app.run(debug = False)
    