# app.py

from flask import Flask, render_template, request, session, jsonify
import joblib
from newspaper import Article
import nltk
from utils import check_source_credibility
from text_utils import clean_text
import csv
import os
from datetime import datetime

# Download punkt once, if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

app = Flask(__name__)
# Using a secure, randomly generated secret key
app.secret_key = os.urandom(24)

# Load model and vectorizer
try:
    model = joblib.load('logistic_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and TfidfVectorizer loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model or TfidfVectorizer file not found. Please train the model by running fake_news.py first.")
    exit()
except Exception as e:
    print(f"❌ Error loading model or TfidfVectorizer: {e}")
    exit()

# Extract article text from link
def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting article from URL '{url}': {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = 0.0
    source_credibility = None
    input_text_display = ""

    if request.method == 'POST':
        text_from_form = request.form.get('news', '').strip()
        link_from_form = request.form.get('link', '').strip()
        
        text_to_process = ""

        if link_from_form:
            input_text_display = link_from_form
            source_credibility = check_source_credibility(link_from_form)
            extracted_text = extract_article_text(link_from_form)
            if extracted_text:
                text_to_process = extracted_text
            else:
                prediction = "⚠️ Unable to extract text from the provided URL."
        elif text_from_form:
            input_text_display = text_from_form
            text_to_process = text_from_form
        else:
            prediction = "Please enter a news article text or a URL."

        if text_to_process:
            cleaned_text_for_model = clean_text(text_to_process)
            
            if not cleaned_text_for_model.strip() or len(cleaned_text_for_model.split()) < 4:
                prediction = "⚠️ Text is too short or ambiguous for reliable prediction after cleaning. Please provide more content."
            else:
                try:
                    transformed_text = vectorizer.transform([cleaned_text_for_model])
                    pred_label = model.predict(transformed_text)[0]
                    proba = model.predict_proba(transformed_text)
                    confidence_score = proba[0][pred_label] 
                    
                    prediction = f"REAL NEWS ✅" if pred_label == 1 else "FAKE NEWS ❌"
                    confidence = float(confidence_score)

                    # Save in session history
                    history = session.get('history', [])
                    history_text_snippet = (text_to_process[:200] + "..." if len(text_to_process) > 200 else text_to_process)
                    history.insert(0, {
                        "text": history_text_snippet,
                        "result": "REAL" if pred_label == 1 else "FAKE",
                        "confidence": f"{confidence*100:.2f}%",
                        "source_credibility": source_credibility or "N/A"
                    })
                    session['history'] = history[:5]
                except Exception as e:
                    print(f"Error during prediction process: {e}")
                    prediction = "⚠️ Error during prediction. Please check logs."

    return render_template("index.html", 
                           prediction=prediction, 
                           confidence=confidence,
                           source_credibility=source_credibility, 
                           history=session.get('history', []),
                           input_text_display=input_text_display)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop('history', None)
    return jsonify({"status": "success"})

# Route to handle user feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    feedback_file = 'feedback.log.csv'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    feedback_data = {
        'timestamp': timestamp,
        'original_text': data.get('text', 'N/A'),
        'model_prediction': data.get('prediction', 'N/A'),
        'user_feedback': data.get('feedback', 'N/A')
    }
    
    file_exists = os.path.isfile(feedback_file)
    
    try:
        with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(feedback_data)
        
        return jsonify({"status": "success", "message": "Feedback received"}), 200
        
    except Exception as e:
        print(f"Error writing to feedback file: {e}")
        return jsonify({"status": "error", "message": "Could not save feedback"}), 500

if __name__ == '__main__':
    app.run(debug=True)