import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stm=PorterStemmer()
def steming(text):
    stm1=re.sub("[^a-zA-Z]"," ",text)
    stm1=stm1.lower()
    stm1=stm1.split()
    stm1=[stm.stem(word) for word in stm1 if  not word in stopwords.words("english")]
    stm1=" ".join(stm1)
    return stm1
tfidf=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
st.title("Fake News Classifier")
input_news=st.text_input("enter the news")
if st.button("Predict"):
#1.preprocess
    tranform=steming(input_news)
#2.vectorize
    vector=tfidf.transform([tranform])
#3.predict
    result=model.predict(vector)[0]
#4.display
    if result ==1:
        st.header("Fake News")
    else:
        st.header("Not Fake News")


