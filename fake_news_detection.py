import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Add label column (1 for true, 0 for fake)
true_data['label'] = 1
fake_data['label'] = 0

# Combine datasets
data = pd.concat([true_data, fake_data], ignore_index=True)

# Combine title and text into a single feature
data['combined_text'] = data['title'] + " " + data['text']

# Extract features and labels
X = data['combined_text']
y = data['label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Function to predict if a news article is fake or real
def predict_news(title, text):
    combined_text = title + " " + text
    text_transformed = vectorizer.transform([combined_text])
    prediction = model.predict(text_transformed)
    return "Real" if prediction[0] == 1 else "Fake"

# Example usage
sample_title = "Donald Trump Trial"
sample_text = "News about the latest trial developments."
print(f"Sample prediction: {predict_news(sample_title, sample_text)}")

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)