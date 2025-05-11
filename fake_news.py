import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

# Load and balance datasets
print("📥 Loading datasets...")
try:
    real_news = pd.read_csv("True.csv")
    fake_news = pd.read_csv("Fake.csv")
    print("✅ Datasets loaded successfully!")
except Exception as e:
    print(f"❌ Error loading datasets: {e}")
    exit()

# Balance dataset (equal number of real & fake news)
min_samples = min(len(real_news), len(fake_news))
real_news = real_news.sample(min_samples, random_state=42)
fake_news = fake_news.sample(min_samples, random_state=42)

# Assign labels
real_news["label"] = 1  # Real News
fake_news["label"] = 0  # Fake News

# Combine and shuffle dataset
df = pd.concat([real_news, fake_news], ignore_index=True)
df = shuffle(df, random_state=42).drop_duplicates(subset=["text"])

print("\n🔍 Class Balance:")
print(df["label"].value_counts())

# Text cleaning
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    text = re.sub(r"\b\w{1,2}\b", "", text)  # Remove short words
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# TF-IDF feature extraction
print("\n🔍 Extracting features...")
tfidf = TfidfVectorizer(
    max_features=10000,  # Using fewer features for better learning
    ngram_range=(1, 2),  # Unigrams and bigrams
    stop_words="english",
    min_df=5,
    max_df=0.85
)
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression model
print("\n🤖 Training Logistic Regression model...")
model = LogisticRegression(
    class_weight="balanced",  # Handles imbalanced data
    penalty="l2",
    C=0.5,
    solver="liblinear",
    max_iter=3000
)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Visualization of Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

# Smart Prediction Function
def predict_news(text, threshold=0.6):
    # Standard processing
    cleaned = clean_text(text)
    if not cleaned.strip() or len(cleaned.split()) < 4:
        return "⚠️ Requires human review (short/ambiguous)"

    vec = tfidf.transform([cleaned])
    prob = model.predict_proba(vec)[0]
    confidence = max(prob)

    if confidence < threshold:
        return f"⚪ Uncertain Prediction (REAL leaning, {confidence:.1%})" if prob[1] > 0.5 else f"⚪ Uncertain Prediction (FAKE leaning, {confidence:.1%})"
    
    return "🟢 REAL" if prob[1] > threshold else "🔴 FAKE"

# Example Predictions
test_cases = [
    "NASA has discovered a new planet with water!",
    "The stock market crashed today, losing 5% in value.",
    "COVID-19 vaccines contain microchips to control people.",
    "A new study confirms coffee reduces the risk of cancer.",
    "The White House announces new healthcare reforms to increase insurance coverage for low-income families.",
    "The United States declared Independence on July 4, 1776."
]

print("\n📝 Example Predictions:")
for news in test_cases:
    print(f"News: {news}\nPrediction: {predict_news(news)}\n")

print("\n🎉 Script finished successfully!")  



