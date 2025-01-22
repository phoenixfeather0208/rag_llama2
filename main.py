from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "The Eiffel Tower is located in Paris, France.",
    "The Great Wall of China is a historic landmark in China.",
    "LLaMA models are advanced language models developed by Meta AI.",
    "The Taj Mahal is a mausoleum located in Agra, India."
]

embeddings = embedding_model.encode(documents)
dimension = embedding_model.shape[1]
index = faiss.indexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

model_name = ""
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

user_query = "Where is the Eiffel Tower located?"

query_embedding = embedding_model.encode(user_query)

k = 2
distances, indices = index.search(np.array(query_embedding).astype("float32"), k)
retrieved_docs = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

print(f"Retrieved Docs: {retrieved_docs}")

input_prompt = f"Context: {context}\n\nUser Query: {user_query}\n\nAnswer:"

inputs = tokenizer(input_prompt, return_tensors="pt").to("cpu") #cuda
outputs = model.generate(inputs["input_ids"], max_length=200, num_beams=5, no_repeat_ngram_size=2)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nGenerated Response:\n {response}")