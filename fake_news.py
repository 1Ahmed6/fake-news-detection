# fake_news.py
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from text_utils import clean_text # Assuming text_utils.py is in the same directory or accessible

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

print("\n🔍 Class Balance (after balancing and dropping duplicates):")
print(df["label"].value_counts())

# Applying the imported clean_text function
print("\n🧹 Cleaning text data...")
df["clean_text"] = df["text"].apply(clean_text)
print("✅ Text data cleaned.")

# --- TF-IDF Feature Extraction ---
print("\n🔍 Extracting features...")
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5,
    max_df=0.85
)
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

# --- K-Fold Cross-Validation ---
print("\n🔄 Performing K-Fold Cross-Validation...")
# Define the model with fixed parameters for cross-validation
cv_model = LogisticRegression(
    class_weight="balanced",
    penalty="l2",
    C=0.5,
    solver="liblinear",
    max_iter=3000,
    random_state=42
)

# Define K-Fold strategy
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Perform cross-validation for different metrics
scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = {}

for metric in scoring_metrics:
    scores = cross_val_score(cv_model, X, y, cv=skf, scoring=metric)
    cv_results[metric] = scores
    print(f"    Cross-validation {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

print("✅ Cross-Validation finished.")


# --- Final Model Training on Full Data ---
# Train the final model on the ENTIRE dataset (X, y) with the same fixed parameters
print("\n🤖 Training final Logistic Regression model on the full dataset...")
final_model = LogisticRegression(
    class_weight="balanced",
    penalty="l2",
    C=0.5,
    solver="liblinear",
    max_iter=3000,
    random_state=42
)
final_model.fit(X, y)
print("✅ Final model trained successfully.")


# --- Save Model and Vectorizer ---
# The TfidfVectorizer 'tfidf' was already fit on the full df["clean_text"]
# The 'final_model' is now trained on all X and y
joblib.dump(final_model, "logistic_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\n💾 Final model and TfidfVectorizer saved successfully!")


# --- Smart Prediction Function ---
# This function implicitly uses the 'tfidf' and 'final_model' variables defined and trained above
def predict_news(text, threshold=0.6):
    cleaned = clean_text(text)
    if not cleaned.strip() or len(cleaned.split()) < 4:
        return "⚠️ Requires human review (short/ambiguous)"

    vec = tfidf.transform([cleaned])
    prob = final_model.predict_proba(vec)[0]
    confidence = max(prob)
    predicted_class_index = prob.argmax()
    predicted_label = final_model.classes_[predicted_class_index]

    if confidence < threshold:
        if predicted_label == 1:
            return f"⚪ Uncertain Prediction (REAL leaning, {prob[1]:.1%})"
        else:
            return f"⚪ Uncertain Prediction (FAKE leaning, {prob[0]:.1%})"
    
    return "🟢 REAL" if predicted_label == 1 else "🔴 FAKE"

# --- Example Predictions ---
# This uses the trained model from this script run for immediate testing
print("\n📝 Example Predictions:")
test_cases = [
    "NASA has discovered a new planet with water!",
    "The stock market crashed today, losing 5% in value.",
    "COVID-19 vaccines contain microchips to control people.",
    "A new study confirms coffee reduces the risk of cancer.",
    "The White House announces new healthcare reforms to increase insurance coverage for low-income families.",
    "The United States declared Independence on July 4, 1776."
]

for news_item in test_cases:
    prediction_result = predict_news(news_item)
    print(f"News: {news_item}\nPrediction: {prediction_result}\n")

print("\n🎉 Script finished successfully!")
