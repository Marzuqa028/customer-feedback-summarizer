from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
analyzer = SentimentIntensityAnalyzer()

new_words = {
    "bestest": 3.0,
    "superb": 2.5,
    "mindblowing": 3.0,
    "awesomest": 3.0,
    "meh": -1.5,
}
analyzer.lexicon.update(new_words)

def summarize_reviews(reviews):
    text = " ".join(reviews)
    if len(text.split()) < 30:
        return text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def sentiment_analyzer(reviews):
    results = []
    for review in reviews:
        scores = analyzer.polarity_scores(review)
        compound = scores['compound']
        mood = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
        results.append((review, mood, compound))
    return results
