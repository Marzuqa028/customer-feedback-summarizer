import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from feedback_summarizer import summarize_reviews, sentiment_analyzer
import json

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        reviews = []

        if 'reviews' in request.form and request.form['reviews']:
            reviews = request.form['reviews'].split('\n')

        if 'csvFile' in request.files and request.files['csvFile']:
            file = request.files['csvFile']
            if allowed_file(file.filename):
                df = pd.read_csv(file)
                reviews = df.iloc[:, 0].tolist()

        summary = summarize_reviews(reviews)
        sentiments = sentiment_analyzer(reviews)

        return jsonify({"summary": summary, "sentiments": sentiments})

    return render_template('index.html')

from flask import Response

@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    if not data or 'summary' not in data or 'sentiments' not in data:
        return jsonify({"error": "Invalid sentiment data"}), 400

    summary_text = data['summary']
    sentiment_data = data['sentiments']

    # Construct the plain text response
    file_content = f"Summary:\n{summary_text}\n\nSentiment Analysis:\n"
    for sentiment in sentiment_data:
        file_content += f"{sentiment}\n"

    # Create Flask Response object with text file headers
    response = Response(file_content, content_type='text/plain')
    response.headers["Content-Disposition"] = "attachment; filename=feedback_summary.txt"
    
    return response


if __name__ == "__main__":
    app.run(debug=True)
