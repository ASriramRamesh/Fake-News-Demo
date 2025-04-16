# 📰 Fake News Detection using Machine Learning

This project aims to identify **fake news** articles using a **Logistic Regression** model trained on a dataset of true and fake news. It includes a Streamlit-based web app that allows users to input a news article's title and content and get a prediction.

## 📁 Project Structure

```
.
├── fake_news_detection.py       # Script for data loading, model training, evaluation
├── fake_news_app.py             # Streamlit web app for prediction
├── model.pkl                    # Trained logistic regression model
├── vectorizer.pkl               # TF-IDF vectorizer used in the model
├── True.csv                     # Dataset of real news articles
├── Fake.csv                     # Dataset of fake news articles
└── README.md                    # Project documentation
```

## ⚙️ How It Works

1. Combines title and body text of news articles.
2. Converts text to TF-IDF features.
3. Trains a Logistic Regression model to classify news as Real (1) or Fake (0).
4. Provides an interactive web interface via Streamlit for users to test the model.

## 🚀 Getting Started

### Prerequisites

Install required packages:

```bash
pip install pandas scikit-learn streamlit
```

### 1. Train the Model

Run the training script:

```bash
python fake_news_detection.py
```

This will output model performance and generate `model.pkl` and `vectorizer.pkl`.

### 2. Run the Web App

```bash
streamlit run fake_news_app.py
```

Enter the title and content of a news article to see the model's prediction.

## 🧠 Model Performance

The model achieves high accuracy using TF-IDF features and logistic regression. See the console output of `fake_news_detection.py` for detailed metrics like precision, recall, and F1-score.

## ✍️ Example Prediction

**Input:**

- Title: `"Donald Trump Trial"`
- Text: `"News about the latest trial developments."`

**Output:**

```
Sample prediction: Real
```

## 📌 Notes

- The model is trained on a public dataset with news headlines and articles.
- It is intended for educational and experimental purposes only.

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).
