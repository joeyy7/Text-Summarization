from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Function to perform extractive summarization
def summarize_text(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]  # Clean the sentences

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(sentences)

    # Calculate sentence scores
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Rank sentences by score
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[::-1]]

    # Select top 3 sentences for the summary
    summary = " ".join(ranked_sentences[:3])
    return summary

# Home route to display the form and process input text
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        # Get the input text from the user
        input_text = request.form["text"]
        summary = summarize_text(input_text)

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
