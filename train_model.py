import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
import os

# Ensure the assets/models directory exists
os.makedirs('assets/models', exist_ok=True)

# Load and preprocess dataset
df = pd.read_csv('C:\\Users\\91954\\ML\\Processed_Spam_SMS_Collection.csv')

# Ensure necessary NLTK resources are available
nltk.download('stopwords')
nltk.download('wordnet')
corpus = []
wnl = WordNetLemmatizer()

for sms_string in list(df.message):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms_string)
    message = message.lower()
    words = message.split()
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ' '.join(lemmatized_words)
    corpus.append(message)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(corpus).toarray()
y = df['label']

# Save TF-IDF vectorizer
joblib.dump(tfidf, 'assets/models/tfidf_vectorizer.pkl')

# Train and save models
models = {
    "MultinomialNB": MultinomialNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'assets/models/{model_name}_model.pkl')

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')
