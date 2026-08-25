import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline

# ==========================================
# 1. SETUP MODELS & MOCK DATA
# ==========================================
print("🚀 Loading embedding and re-ranker models...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

documents = [
    "Password policy requires a minimum of 12 characters, including numbers and symbols.",
    "The company's remote work policy allows employees to work from home up to 3 days a week.",
    "For multi-factor authentication (MFA), users must use a registered hardware token or authenticator app.",
    "Network firewall configurations are updated bi-weekly by the SecOps infrastructure team.",
    "Annual performance reviews happen every December, followed by compensation adjustments in January.",
    "Access to production databases requires an approved JIRA ticket and temporary IAM role elevation.",
]

# ==========================================
# 2. THE QUERY
# ==========================================
query = "How do I securely log into the production database using MFA?"
print(f"🔮 User Query: '{query}'\n")

# ==========================================
# 3. ROUTE A: LEXICAL SEARCH (BM25 Keyword)
# ==========================================
tokenized_docs = [doc.lower().split(" ") for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
tokenized_query = query.lower().split(" ")
bm25_scores = bm25.get_scores(tokenized_query)

# ==========================================
# 4. ROUTE B: SEMANTIC SEARCH (Dense Embeddings)
# ==========================================
doc_embeddings = bi_encoder.encode(documents, convert_to_tensor=True)
query_embedding = bi_encoder.encode(query, convert_to_tensor=True)

from sentence_transformers.util import cos_sim
semantic_scores = cos_sim(query_embedding, doc_embeddings).cpu().numpy()[0]

# ==========================================
# 5. HYBRID MERGE & CANDIDATE SELECTION
# ==========================================
def normalize(scores):
    if np.max(scores) == np.min(scores):
        return np.zeros_like(scores)
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

hybrid_scores = 0.5 * normalize(bm25_scores) + 0.5 * normalize(semantic_scores)
top_k_hybrid_idx = np.argsort(hybrid_scores)[::-1][:3]
hybrid_candidates = [documents[idx] for idx in top_k_hybrid_idx]

# ==========================================
# 6. STAGE 2: CROSS-ENCODER RE-RANKING
# ==========================================
pairs = [[query, doc] for doc in hybrid_candidates]
rerank_scores = cross_encoder.predict(pairs)
reranked_indices = np.argsort(rerank_scores)[::-1]

# Pick the absolute best retrieved chunk to be our LLM ground truth context
best_context = hybrid_candidates[reranked_indices[0]]
print(f"🎯 Best Context Selected:\n\"{best_context}\"\n")
print("="*60 + "\n")

# ==========================================
# 7. STAGE 3: LOCAL LLM RESPONSE GENERATION
# ==========================================
print("🤖 Loading Local Generation LLM (Qwen2.5-0.5B)...")
# Initialize a ultra-lightweight local text generation model
llm_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

# Structure the prompt with explicit guardrails
system_prompt = "You are a secure corporate assistant. Answer the user's question accurately using ONLY the provided context. If the context doesn't contain the answer, say 'I cannot find that in the official documentation.'"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Context: {best_context}\n\nQuestion: {query}"}
]

print("🧠 Generating final response based strictly on context...\n")
outputs = llm_pipeline(messages, max_new_tokens=100, temperature=0.1)

# Extract and print the assistant's response text
final_answer = outputs[0]['generated_text'][-1]['content']
print("✨ [FINAL LLM GENERATION] ✨")
print(final_answer)