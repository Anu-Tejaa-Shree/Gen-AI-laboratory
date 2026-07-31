from transformers import pipeline

print("=" * 60)
print("TEXT SUMMARIZATION AND QUESTION ANSWERING")
print("=" * 60)

# ---------------------------------------------
# Load Summarization Model
# ---------------------------------------------
print("\nLoading Summarization Model...")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

print("Model Loaded Successfully!")

# ---------------------------------------------
# Input Text
# ---------------------------------------------

text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity.

AI systems can analyze huge amounts of data,
identify patterns, make predictions, and support
intelligent decision-making.

Generative AI is a branch of Artificial Intelligence
that can create new content such as text,
images, audio, video, and computer programs.
"""

print("\nOriginal Text:\n")
print(text)

# ---------------------------------------------
# Generate Summary
# ---------------------------------------------

summary = summarizer(
    text,
    max_length=60,
    min_length=20,
    do_sample=False
)

print("\nGenerated Summary:\n")
print(summary[0]["summary_text"])

# ---------------------------------------------
# Load Question Answering Model
# ---------------------------------------------

print("\nLoading Question Answering Model...")

question_answerer = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

print("Model Loaded Successfully!")

# ---------------------------------------------
# Context
# ---------------------------------------------

context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text,
images, audio, video, and computer programs.

Large Language Models are commonly used for
text generation,
summarization,
translation,
and question answering.
"""

question = "What type of content can Generative AI create?"

print("\nQuestion:")
print(question)

# ---------------------------------------------
# Predict Answer
# ---------------------------------------------

answer = question_answerer(
    question=question,
    context=context
)

print("\nAnswer:")
print(answer["answer"])

print("\nConfidence Score:")
print(round(answer["score"],3))

print("\nExperiment Completed Successfully!")