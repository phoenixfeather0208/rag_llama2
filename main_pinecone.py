from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import json
from pinecone import Pinecone, ServerlessSpec

with open("config.json", "r") as f:
    config = json.load(f)
pc = Pinecone(api_key=config["PINECONE_API_KEY"])

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

# Using Pinecone Vector Database
index_name = "pinecone"
pc.delete_index(index_name)
pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

vectors = []
for i, document in enumerate(documents):
     vectors.append({
        "id": f"doc{i+1}",  # Unique ID for each document
        "values": np.array(embeddings).astype("float32")[i],  # The embedding vector
        "metadata": {"text": document}  # Store the original document as metadata
    })

index = pc.Index(index_name)
index.upsert(vectors, "pinecone")
print("Updating Pinecone Database.....")

while True:
    if index.describe_index_stats()['total_vector_count'] != 0:
        break


user_query = "Where is the Eiffel Tower located?"

query_embedding = embedding_model.encode([user_query])

k = 2  # Number of documents to retrieve
results = index.query(namespace="pinecone",vector=query_embedding.tolist()[0], top_k=k, include_metadata=True)
retrieved_docs = [documents[int(match['id'][3:]) - 1] for match in results["matches"]]
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