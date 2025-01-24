# How to download LLAMA models?

```python
ollama pull llama2
```

# How to run the project?

1. Rename "example.config.json" file to "config.json" and fill out the necessary fields.
2. Run ollama server.
3. Run this command

```python
python exercise.py #add index and data to pinecone vector database
cd chatbot
python backend.py
streamlit run frontend.py
```
