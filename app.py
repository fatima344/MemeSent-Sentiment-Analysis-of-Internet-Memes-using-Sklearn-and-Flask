from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model=pickle.load(open("model.pkl","rb"))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    i=request.files['i']
    image_path="./pics/"+i.filename
    i.save(image_path)
    #text = pytesseract.image_to_string(img)
    #v=TfidfVectorizer()
    #x=v.fit_transform(df.text_corrected)
    #x=x.toarray()
    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    print(f"Original Dimensions : {img.shape}")
    resized = cv2.resize(img, (100,100))
    array2 = resized.flatten()
    p=pd.DataFrame({'row':array2})
    p=p.T
    prediction=model.predict(p)
    return render_template("index.html", prediction_text="Sentiment of the meme is {}".format(prediction))


if __name__ == '__main__':
    app.run(port=3000, debug=True)