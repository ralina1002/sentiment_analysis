import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Step 1: Dataset Preparation
positive_reviews_path = 'path/to/positive/reviews/folder'
negative_reviews_path = 'path/to/negative/reviews/folder'

positive_reviews = [open(os.path.join(positive_reviews_path, file), 'r').read() for file in os.listdir(positive_reviews_path)]
negative_reviews = [open(os.path.join(negative_reviews_path, file), 'r').read() for file in os.listdir(negative_reviews_path)]

reviews = positive_reviews + negative_reviews
labels = np.concatenate((np.ones(len(positive_reviews)), np.zeros(len(negative_reviews))))

# Step 2: Data Preprocessing
stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in stopwords]
    return ' '.join(tokens)

preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Step 3: Feature Extraction
vectorizer = CountVectorizer()
feature_vectors = vectorizer.fit_transform(preprocessed_reviews).toarray()

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
