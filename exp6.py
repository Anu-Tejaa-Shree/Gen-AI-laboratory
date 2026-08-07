# ==========================================================
# Experiment: Retrieval-Augmented Generation (RAG) using FAISS
# ==========================================================

# Install Required Libraries (Run once in terminal)
# pip install transformers sentence-transformers faiss-cpu torch sentencepiece

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ==========================================================
# STEP 1: Create the Knowledge Base
# ==========================================================

documents = [
    """
    Generative Artificial Intelligence is a branch of AI that creates
    new content such as text, images, audio, video and computer programs.
    """,

    """
    Large Language Models are transformer-based models trained on massive
    text datasets. They are used for text generation, summarization,
    translation, question answering and conversational AI.
    """,

    """
    Retrieval-Augmented Generation combines information retrieval with
    text generation. It retrieves relevant documents from an external
    knowledge base and gives them to a language model as context.
    """,

    """
    Vector databases store high-dimensional embeddings and perform
    similarity searches. Examples of vector databases include FAISS,
    ChromaDB, Pinecone, Weaviate and Milvus.
    """,

    """
    Prompt engineering is the process of designing clear instructions
    that guide a language model to produce accurate and useful responses.
    Common techniques include zero-shot, few-shot and role-based prompting.
    """,

    """
    Fine-tuning adapts a pretrained language model to a specific domain
    or task by training it further using a smaller domain-specific dataset.
    """
]

# ==========================================================
# STEP 2: Load Sentence Transformer Model
# ==========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================================
# STEP 3: Convert Documents into Embeddings
# ==========================================================

print("Generating document embeddings...")

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

document_embeddings = document_embeddings.astype("float32")

# ==========================================================
# STEP 4: Normalize Embeddings
# ==========================================================

faiss.normalize_L2(document_embeddings)

# ==========================================================
# STEP 5: Create FAISS Vector Database
# ==========================================================

embedding_dimension = document_embeddings.shape[1]

vector_database = faiss.IndexFlatIP(embedding_dimension)

vector_database.add(document_embeddings)

print(f"\nStored {vector_database.ntotal} documents in FAISS.")

# ==========================================================
# STEP 6: Load FLAN-T5 Model
# ==========================================================

print("\nLoading language model (this may take a minute)...")

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base"
)

print("Language model loaded successfully!")

# ==========================================================
# STEP 7: Retrieve Relevant Documents
# ==========================================================

def retrieve_documents(query, top_k=2):
    """
    Retrieve the most relevant documents using FAISS.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    similarity_scores, document_indices = vector_database.search(
        query_embedding,
        top_k
    )

    retrieved_documents = []

    for index, score in zip(document_indices[0], similarity_scores[0]):
        retrieved_documents.append(
            {
                "document": documents[index].strip(),
                "score": float(score)
            }
        )

    return retrieved_documents

# ==========================================================
# STEP 8: Generate Answer
# ==========================================================

def generate_answer(query, retrieved_documents):
    """
    Generate answer using retrieved context.
    """

    context = "\n\n".join(
        item["document"] for item in retrieved_documents
    )

    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{query}

Instructions:
1. Give a clear and concise answer.
2. Do not add information that is not present in the context.
3. If the answer is unavailable, reply:
"The answer is not available in the knowledge base."

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"]

# ==========================================================
# STEP 9: Main Program
# ==========================================================

print("\n")
print("=" * 60)
print(" RETRIEVAL-AUGMENTED GENERATION (RAG) SYSTEM ")
print("=" * 60)

while True:

    user_query = input("\nEnter your question (or type 'exit'): ")

    if user_query.lower() == "exit":
        print("\nThank you for using the RAG System!")
        break

    retrieved_results = retrieve_documents(
        query=user_query,
        top_k=2
    )

    answer = generate_answer(
        query=user_query,
        retrieved_documents=retrieved_results
    )

    print("\n")
    print("=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, item in enumerate(retrieved_results, start=1):
        print(f"\nDocument {i}")
        print("-" * 60)
        print(item["document"])
        print(f"\nSimilarity Score : {item['score']:.4f}")

    print("\n")
    print("=" * 60)
    print("GENERATED ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)