from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import joblib
import pickle
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer



app = Flask(__name__)
api = Api(app)
CORS(app)
clf_path = './bilstm.pkl'
tokenizer = './tokenizer.pkl'

with open(clf_path, 'rb') as f:
    clf = joblib.load(f)
    
@app.route('/', methods=['GET'])
def index():
    query = request.args.get('query')
    
    # vec = CountVectorizer(stop_words='english',vocabulary=pickle.load(open("vec_vocab.pkl", "rb")))
    word_to_index = tokenizer.word_index
    pad_sequences(query)

    token = tokenizer.texts_to_sequences(query)

    prediction = clf.predict(pad_sequences(token))
    print("prediction",prediction[0])

    if prediction[0][1] > 0.55 :
        pred_text = 'Positive'
    elif prediction[0][0] < 0.45:
        pred_text = 'Negative'
    else:
        pred_text = 'Neutral'

    output = {'prediction': pred_text}
    return output

if __name__ == '__main__':
    app.run(debug=True)