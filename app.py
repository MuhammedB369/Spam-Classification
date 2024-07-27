import streamlit as st
import joblib
import re
import string

import warnings
warnings.filterwarnings('ignore')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


exclude = string.punctuation

model_path = 'assets/models/spam_classification.joblib'
vectorizer_path = 'assets/models/tfidf_vectorizer.pkl'

model = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    re_url = re.compile(r'https?://\S+|www\.\S+')
    text = re_url.sub('', text)
    
    # Remove punctuation
    exclude = set(string.punctuation)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords
    stopwrds = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwrds]
    
    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def predict_mess(text):
    message = preprocess_text(text)
    vectorized_mess = tfidf.transform([message]).toarray()
    prediction = model.predict(vectorized_mess)
    # st.write(prediction)
    if prediction == 0:
        st.write("message is spam")
    else:
        st.write("message is ham")

def main():
    st.title("Spam Classification")
    st.write("This is a simple spam classification model using Random Forest")
    # st.write(model)
    # st.write(tfidf)
    raw_text = st.text_input("Enter text to classify")
    if st.button("Predict"):
        predict_mess(raw_text)


if __name__ == "__main__":
    main()