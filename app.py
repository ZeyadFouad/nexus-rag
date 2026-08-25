import streamlit as str
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# Configure the Streamlit web page layout
str.set_page_config(page_title="Advanced RAG Enterprise Portal", page_icon="🔐", layout="centered")
str.title("🔐 Advanced Enterprise RAG Portal")
str.subheader("Ask questions based on company security policies")

# ==========================================
# CACHE MODELS TO PREVENT RE-LOADING ON REFRESH
# ==========================================
@str.cache_resource
def load_pipeline():
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    return bi_encoder, cross_encoder, llm_pipeline

bi_encoder, cross_encoder, llm_pipeline = load_pipeline()

# Knowledge Base Chunks
documents = [
    "Password policy requires a minimum of 12 characters, including numbers and symbols.",
    "The company's remote work policy allows employees to work from home up to 3 days a week.",
    "For multi-factor authentication (MFA), users must use a registered hardware token or authenticator app.",
    "Network firewall configurations are updated bi-weekly by the SecOps infrastructure team.",
    "Annual performance reviews happen every December, followed by compensation adjustments in January.",
    "Access to production databases requires an approved JIRA ticket and temporary IAM role elevation.",
]

# Initialize keyword search index
tokenized_docs = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
doc_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)

# Helper normalization function
def normalize(scores):
    if np.max(scores) == np.min(scores):
        return np.zeros_like(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# ==========================================
# WEB APPLICATION UI & LOGIC
# ==========================================
user_query = str.text_input("Enter your policy question here:", placeholder="e.g., How do I access production databases?")

if user_query:
    with str.spinner("Running Advanced RAG Retrieval & Generation..."):
        # 1. Lexical Search
        tokenized_query = user_query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)

        # 2. Semantic Search
        query_embedding = bi_encoder.encode(user_query, convert_to_tensor=True)
        from sentence_transformers.util import cos_sim
        semantic_scores = cos_sim(query_embedding, doc_embeddings).cpu().numpy()[0]

        # 3. Hybrid Score & Candidate Selection
        hybrid_scores = 0.5 * normalize(bm25_scores) + 0.5 * normalize(semantic_scores)
        top_k_hybrid_idx = np.argsort(hybrid_scores)[::-1][:3]
        hybrid_candidates = [documents[idx] for idx in top_k_hybrid_idx]

        # 4. Cross-Encoder Re-ranking
        pairs = [[user_query, doc] for doc in hybrid_candidates]
        rerank_scores = cross_encoder.predict(pairs)
        reranked_indices = np.argsort(rerank_scores)[::-1]
        best_context = hybrid_candidates[reranked_indices[0]]

        # Display the pipeline mechanics visually to show off the backend
        with str.expander("🔍 Inspect Pipeline Mechanics (Debug Mode)"):
            str.write("**Top Retained Context Chunk after Re-ranking:**")
            str.info(best_context)

        # 5. LLM Generation
        system_prompt = "You are a secure corporate assistant. Answer the user's question accurately using ONLY the provided context. If the context doesn't contain the answer, say 'I cannot find that in the official documentation.'"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {best_context}\n\nQuestion: {user_query}"}
        ]
        
        # Suppress warnings inside generation call
        outputs = llm_pipeline(messages, max_new_tokens=150, temperature=0.1, clean_up_tokenization_spaces=False)
        final_answer = outputs[0]['generated_text'][-1]['content']

        # Print the answer to the web page
        str.write("### 🤖 Assistant Answer:")
        str.success(final_answer)