from transformers import pipeline

print("=" * 60)
print("FOUNDATION MODELS")
print("Sentiment Analysis and Document Classification")
print("=" * 60)

# ---------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------

print("\nLoading Sentiment Analysis Model...\n")

sentiment_analyzer = pipeline("sentiment-analysis")

text = input("Enter text for Sentiment Analysis:\n")

result = sentiment_analyzer(text)

print("\n----------- SENTIMENT ANALYSIS RESULT -----------")
print("Input Text :", text)
print("Sentiment  :", result[0]["label"])
print("Confidence :", round(result[0]["score"], 3))

# ---------------------------------------------------
# DOCUMENT CLASSIFICATION
# ---------------------------------------------------

print("\nLoading Document Classification Model...\n")

classifier = pipeline("zero-shot-classification")

document = input("Enter a document:\n")

labels = [
    "Technology",
    "Sports",
    "Politics",
    "Entertainment"
]

classification = classifier(document, labels)

print("\n----------- DOCUMENT CLASSIFICATION RESULT -----------")
print("Document:")
print(document)

print("\nPredicted Category :", classification["labels"][0])
print("Confidence Score   :", round(classification["scores"][0], 3))