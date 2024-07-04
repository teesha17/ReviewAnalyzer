import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

pickle_in = open("Classifier.pkl","rb")
classifier=pickle.load(pickle_in)

pickle_in = open("Vectorizer.pkl","rb")
cv=pickle.load(pickle_in)

def getresponse(new_review):
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    new_y_pred = classifier.predict(new_X_test)
    if new_y_pred==1:
        return "THE REVIEW PROVIDED IS POSITIVE"
    else:
        return "THE REVIEW PROVIDED IS NEGATIVE"

st.set_page_config(page_title="Predict the Review",
                   page_icon="",
                   layout="centered",
                   initial_sidebar_state="collapsed")
st.header("Review analyzer")
new_review = st.text_input("Enter the review")

submit = st.button("Generate")

##Final response
if submit:
    st.write(getresponse(new_review=new_review))