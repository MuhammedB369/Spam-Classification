import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd

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

# Load your dataset
data_path = 'assets/datasets/spam-text-messages.csv'  # your dataset path
data = pd.read_csv(data_path)

# Add Total Words and Total Chars columns
data['Total Words'] = data['Message'].apply(lambda x: len(x.split()))
data['Total Chars'] = data['Message'].apply(lambda x: len(x))

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
    if prediction == 0:
        st.write("Message is Spam")
    else:
        st.write("Message is Ham")

def overview():
    st.title("Spam SMS Classification")

    st.write("""
    ### Project Summary
    This project aims to classify SMS messages as either 'Spam' or 'Ham' (non-spam) using a machine learning model. The classification is carried out using a Random Forest classifier, which is trained on a dataset of labeled SMS messages. The goal is to accurately predict whether a given SMS is spam or not based on its content.

    ### Key Components
    - **Dataset**: The dataset contains SMS messages labeled as 'Spam' or 'Ham'. It is used to train and evaluate the model's performance.
    - **Model**: A Random Forest classifier is employed for prediction. This model is trained on features extracted from the SMS messages after preprocessing.
    - **Text Processing**: The SMS messages are preprocessed to remove URLs, punctuation, and stopwords. Text is then tokenized and stemmed to prepare it for feature extraction using TF-IDF.
    - **Evaluation**: The model’s accuracy is assessed to ensure reliable predictions. The performance metrics are used to validate the effectiveness of the classification.

    """)

def infographics():
    st.title("Infographics")
    
    # Visualization 1: KDE plot for Total Words
    st.subheader("Distribution of Total Words")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=data['Total Words'], hue=data['Category'], palette='winter', shade=True)
    st.pyplot(plt)
    
    # Visualization 2: KDE plot for Total Chars
    st.subheader("Distribution of Total Characters")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=data['Total Chars'], hue=data['Category'], palette='winter', shade=True)
    st.pyplot(plt)
    
    # Visualization 3: Word Cloud for Spam Messages
    st.subheader("Word Cloud for Spam Messages")
    text_spam = " ".join(data[data['Category'] == 'spam']['Message'])
    plt.figure(figsize=(15, 10))
    wordcloud_spam = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(text_spam)
    plt.imshow(wordcloud_spam, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)
    
    # Visualization 4: Word Cloud for Ham Messages
    st.subheader("Word Cloud for Ham Messages")
    text_ham = " ".join(data[data['Category'] == 'ham']['Message'])
    plt.figure(figsize=(15, 10))
    wordcloud_ham = WordCloud(max_words=500, height=800, width=1500, background_color="black", colormap='viridis').generate(text_ham)
    plt.imshow(wordcloud_ham, interpolation="bilinear")
    plt.axis('off')
    st.pyplot(plt)

def prediction():
    st.title("Prediction")
    
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ''
    
    raw_text = st.text_input("Enter text to classify", value=st.session_state['input_text'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Predict"):
            predict_mess(raw_text)
            st.session_state['input_text'] = raw_text  
    
    with col2:
        if st.button("Clear"):
            st.session_state['input_text'] = ''
            st.experimental_rerun() 
            
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Infographics", "Prediction"])
    
    if page == "Overview":
        overview()
    elif page == "Infographics":
        infographics()
    elif page == "Prediction":
        prediction()

if __name__ == "__main__":
    main()

