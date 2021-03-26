import re
import pickle
import string
import numpy as np 
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = stopwords.words('english')
predict_dict = {0: 'not a spam', 1: 'a spam'}

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def stem_text(text):
    return ' '.join([stemmer.stem(t) for t in text.split()])

def remove_stop_words(text):
    return ' '.join([t for t in text.split() if t not in stop_words])

def df_clean_text(df):
    return df['text'].apply(clean_text).to_frame()

def df_stem_text(df):
    return df['text'].apply(stem_text).to_frame()
    
def df_remove_stop_words(df):
    return df['text'].apply(remove_stop_words)

class DataframeFunctionTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self
    
with open("classifier.pkl","rb") as clf_pkl:
    clf = pickle.load(clf_pkl)

def rootpage():
    return "Hello! You can evaluate if your message is a spam here!"


def predict_spam(text):
    

    target = pd.DataFrame([[text]],
                          columns = ['text'])
    pred = clf.predict(target)

    return pred


def main():
    st.title("Spam message prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Spam prediction app with an ensemble model</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    text = st.text_input('input your message here')
    
    result=""
    if st.button("Evaluate"):
        result = predict_spam(text)

        st.success('This is {}'.format(predict_dict[result[0]]))
    
    if st.button("About"):
        st.text("Spam prediction with an ML model")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    