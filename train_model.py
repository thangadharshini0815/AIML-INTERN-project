# train_model_large.py

import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

# Extended dataset
data = {
    'text': [
        "I feel sad and lonely",
        "I'm so depressed and hopeless",
        "I am happy and excited today",
        "Feeling joyful and energetic",
        "I'm anxious about tomorrow's meeting",
        "Worried and can't focus",
        "Everything seems fine right now",
        "I'm doing okay, just regular day",
        "I hate myself and want to cry",
        "I love my life and everything in it",
        "Feeling like a failure",
        "I'm overwhelmed and stressed",
        "I'm content with how things are",
        "Nothing much happening today",
        "Feeling grateful for everything",
        "I feel worthless and empty",
        "I'm nervous about my exam",
        "I’m optimistic about the future",
        "I feel mentally exhausted",
        "I'm peaceful and relaxed",
        "Things are tough but I’m managing",
        "I can't handle this anymore",
        "Everything will be alright",
        "Just another normal day",
        "I’m scared to face people",
        "Life feels good right now",
        "I feel like giving up",
        "I'm hopeful things will improve",
        "No motivation to do anything",
        "I'm confident today"
    ],
    'label': [
        "sad", "sad", "happy", "happy", "anxious", "anxious", "neutral", "neutral",
        "sad", "happy", "sad", "anxious", "neutral", "neutral", "happy", "sad",
        "anxious", "happy", "sad", "happy", "anxious", "sad", "neutral", "neutral",
        "anxious", "happy", "sad", "happy", "sad", "happy"
    ]
}

df = pd.DataFrame(data)

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['processed'] = df['text'].apply(preprocess)

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed'])
y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save to file
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

