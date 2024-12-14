# Import necessary libraries
import re
import sys
import warnings
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Ensure NLTK data is downloaded
import os

# Set the NLTK data directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

# Suppress warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Streamlit title
st.title("Text Classification Model Accuracy Checker")

# File upload section
uploaded_file = st.file_uploader("Upload a CSV file to test the model:", type=["csv"])

if uploaded_file:
    # Load the uploaded CSV file
    data_raw = pd.read_csv(uploaded_file)
    
    # Shuffle the data
    data_raw = data_raw.sample(frac=1)

    # Preprocessing
    categories = list(data_raw.columns.values)
    categories = categories[2:]  # Adjust based on your dataset structure

    data_raw['Heading'] = data_raw['Heading'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\d+', '', regex=True).str.replace(r'<.*?>', '', regex=True)

    # Load stopwords for Swedish
    stop_words = set(stopwords.words('swedish'))

    from nltk.tokenize import word_tokenize

    def removeStopWords(sentence):
        return " ".join([word for word in word_tokenize(sentence, language='swedish') if word not in stop_words])

    data_raw['Heading'] = data_raw['Heading'].apply(removeStopWords)

    stemmer = SnowballStemmer("swedish")

    def stemming(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    # Apply stemming to the Heading column
    data_raw['Heading'] = data_raw['Heading'].apply(stemming)

    # Splitting data
    train, test = train_test_split(data_raw, random_state=42, test_size=0.30, shuffle=True)

    train_text = train['Heading']
    test_text = test['Heading']

    # Vectorizing
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(train_text)

    x_train = vectorizer.transform(train_text)
    y_train = train.drop(labels=['Id', 'Heading'], axis=1)

    x_test = vectorizer.transform(test_text)
    y_test = test.drop(labels=['Id', 'Heading'], axis=1)

    # ML pipeline
    LogReg_pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression())),
    ])

    # Hyperparameter tuning
    C_values = [0.1, 1, 10]
    penalty_values = ['l1', 'l2']
    param_grid = dict(clf__estimator__C=C_values, clf__estimator__penalty=penalty_values)

    grid = GridSearchCV(LogReg_pipeline, param_grid, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)

    # Best model
    best_clf_pipeline = grid.best_estimator_
    best_clf_pipeline.fit(x_train, y_train)

    # Prediction
    y_pred_proba = best_clf_pipeline.predict_proba(x_test)
    threshold = 0.3  # Define your threshold here
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Accuracy calculation
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display results
    st.success(f"Accuracy on uploaded dataset: {accuracy:.2%}")

# Instructions
st.markdown("""
### Instructions:
1. Upload a CSV file formatted like the training dataset.
2. The file must contain a `Heading` column for text data and appropriate label columns.
3. The model will process your data and display the accuracy.
""")
