import tensorflow_datasets as tfds
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from feedback_summarizer import sentiment_analyzer, summarize_reviews

# Load IMDb dataset
def load_imdb_data():
    dataset, info = tfds.load("imdb_reviews", split=["train", "test"], as_supervised=True, with_info=True)

    train_data = [(str(text.numpy()), int(label.numpy())) for text, label in dataset[0]]
    test_data = [(str(text.numpy()), int(label.numpy())) for text, label in dataset[1]]

    return train_data, test_data

# Fetch IMDb dataset
train_reviews, test_reviews = load_imdb_data()
test_texts = [text for text, _ in test_reviews]
true_labels = [label for _, label in test_reviews]

# -------------------------------------
# 📌 SENTIMENT ANALYSIS ACCURACY TEST
# -------------------------------------
print("\n🔹 Evaluating Sentiment Analysis Accuracy...")

# Get predicted sentiment labels
predicted_labels = [1 if sentiment == "Positive" else 0 for _, sentiment, _ in sentiment_analyzer(test_texts)]

# Compute accuracy metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print results
print(f"✅ Sentiment Analysis Accuracy: {accuracy:.2f}")
print(f"✅ Precision: {precision:.2f}")
print(f"✅ Recall: {recall:.2f}")
print(f"✅ F1-score: {f1:.2f}")

# -------------------------------------
# 📌 SUMMARIZATION ACCURACY TEST (ROUGE Score)
# -------------------------------------
print("\n🔹 Evaluating Summarization Accuracy...")

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# Get predicted summaries
predicted_summaries = [summarize_reviews([text]) for text in test_texts[:50]]  # Test with first 50 reviews
actual_summaries = test_texts[:50]  # Assume original reviews as reference summaries

# Calculate ROUGE scores
rouge_scores = [scorer.score(pred, actual) for pred, actual in zip(predicted_summaries, actual_summaries)]

# Compute average scores
avg_rouge1 = sum([score["rouge1"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rouge2 = sum([score["rouge2"].fmeasure for score in rouge_scores]) / len(rouge_scores)
avg_rougeL = sum([score["rougeL"].fmeasure for score in rouge_scores]) / len(rouge_scores)

# Print results
print(f"✅ ROUGE-1 Score: {avg_rouge1:.2f}")
print(f"✅ ROUGE-2 Score: {avg_rouge2:.2f}")
print(f"✅ ROUGE-L Score: {avg_rougeL:.2f}")

print("\n🎯 Accuracy Evaluation Completed!")
