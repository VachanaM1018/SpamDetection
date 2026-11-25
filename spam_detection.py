
# Spam Detection Using Machine Learning

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
data = {
    'email': [
        "Hey! Are you free to meet tomorrow?",
        "Congratulations! You've won a free iPhone!",
        "Let's grab coffee at 3 pm.",
        "Click here to claim your prize now!",
        "Sure, where should we meet?",
        "Limited offer: Get 50% off on all products.",
        "I'll be at the central cafe.",
        "Don't miss out on this exclusive deal!",
        "Sounds good, see you there.",
        "You have been selected for a special bonus."
    ],
    'label': [
        "Not Spam", "Spam", "Not Spam", "Spam", "Not Spam",
        "Spam", "Not Spam", "Spam", "Not Spam", "Spam"
    ]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df.head())

# Split features and labels
X = df['email']
y = df['label']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Convert text into numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Machine Learning model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict test data
y_pred = model.predict(X_test_vec)

# Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Test with a new email
new_email = [
    "Congratulations! You won a free prize!"
]

new_email_vec = vectorizer.transform(new_email)
prediction = model.predict(new_email_vec)

print("\nNew Email:", new_email[0])
print("Prediction:", prediction[0])
