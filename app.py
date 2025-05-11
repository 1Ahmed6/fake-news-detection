from flask import Flask, render_template, request, session, jsonify
import joblib
import re
import string
from newspaper import Article
import nltk
from utils import check_source_credibility

# Download punkt once
nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load model and vectorizer
model = joblib.load('logistic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Extract article text from link
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting article: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = 0
    source_credibility = None
    if request.method == 'POST':
        text = request.form.get('news', '').strip()
        link = request.form.get('link', '').strip()

        if link:
            source_credibility = check_source_credibility(link)
            extracted_text = extract_article_text(link)
            if extracted_text:
                text = extracted_text
            else:
                prediction = "Unable to extract text from the provided URL."
                confidence = 0.0
                return render_template("index.html", prediction=prediction, confidence=confidence, source_credibility=source_credibility, history=session.get('history', []))

        if text:
            cleaned = clean_text(text)
            transformed = vectorizer.transform([cleaned])
            pred = model.predict(transformed)[0]
            proba = model.predict_proba(transformed).max()

            prediction = f"REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌"
            confidence = float(proba)

            # Save in session history
            history = session.get('history', [])
            history.insert(0, {
                "text": text[:200] + ("..." if len(text) > 200 else ""),
                "result": "REAL" if pred == 1 else "FAKE",
                "confidence": f"{confidence*100:.2f}%",
                "source_credibility": source_credibility or "N/A"
            })
            session['history'] = history[:5]
        else:
            prediction = "Please enter a news article or a URL."
            confidence = 0.0

    return render_template("index.html", prediction=prediction, confidence=confidence, source_credibility=source_credibility, history=session.get('history', []))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
