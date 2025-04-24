import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    raise

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]    
    return ' '.join(tokens)

try:
    true_data = pd.read_csv('True.csv')
    fake_data = pd.read_csv('Fake.csv')
except FileNotFoundError:
    print("Error: 'True.csv' or 'Fake.csv' not found. Ensure they are in the same directory.")
    raise

true_data['label'] = 1
fake_data['label'] = 0

data = pd.concat([true_data, fake_data], ignore_index=True)

data['title'] = data['title'].fillna('')
data['text'] = data['text'].fillna('')

data['combined_text'] = data['title'] + " " + data['text']

data['processed_text'] = data['combined_text'].apply(preprocess_text)

X = data['processed_text']
y = data['label']

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

def predict_news(title, text):
    combined_text = title + " " + text
    processed_text = preprocess_text(combined_text)
    text_transformed = vectorizer.transform([processed_text])
    prediction = model.predict(text_transformed)
    probability = model.predict_proba(text_transformed)[0]
    label = "Real" if prediction[0] == 1 else "Fake"
    confidence = probability[1] if prediction[0] == 1 else probability[0]
    return label, confidence

sample_title = "Donald Trump Trial"
sample_text = "News about the latest trial developments."
label, confidence = predict_news(sample_title, sample_text)
print(f"Sample prediction: {label} (Confidence: {confidence:.2%})")

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
print("Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
