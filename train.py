# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # Load dataset
# data = pd.read_csv("data/intents.csv")

# # Input and output
# X = data["text"]
# y = data["intent"]

# # Convert text into numbers
# vectorizer = TfidfVectorizer()

# X_vectorized = vectorizer.fit_transform(X)

# # Train NLP model
# model = LogisticRegression()

# model.fit(X_vectorized, y)

# print("Model trained successfully!")


import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# Load dataset
data = pd.read_csv("data/intents.csv")


# Input and output
X = data["text"]
y = data["intent"]


# Convert text into numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# Train NLP model
model = LogisticRegression()
model.fit(X_vectorized, y)


# Save model and vectorizer
joblib.dump(model, "email_intent_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


print("Model trained successfully!")
print("Model saved successfully!")