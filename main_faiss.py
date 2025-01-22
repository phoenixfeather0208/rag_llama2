from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import requests
import json

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
url = "http://localhost:11434/api/generate"

documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is a historic landmark in China.",
    "LLaMA models are advanced language models developed by Meta AI.",
    "The Taj Mahal is a mausoleum located in Agra, India."
]

embeddings = embedding_model.encode(documents)
dimension = embeddings.shape[1]
# Using Faiss Vector Database
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

user_query = "Where is the Eiffel Tower located?"

query_embedding = embedding_model.encode([user_query])

k = 2  # Number of documents to retrieve
distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
retrieved_docs = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

print(f"Retrieved Docs: {retrieved_docs}")

input_prompt = f"Context: {context}\n\nUser Query: {user_query}\n\nAnswer:"

payload = {
    "model": "llama2:latest",
    "prompt": "".join(input_prompt)
}

response = requests.post(url, json=payload)
full_response = ""
if response.status_code == 200:
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            # Parse each line (JSON object) and extract the "response" field
            data = json.loads(chunk)
            full_response += data.get("response", "")
    
    print("Complete Response:", full_response)
else:
    print(response.status_code)