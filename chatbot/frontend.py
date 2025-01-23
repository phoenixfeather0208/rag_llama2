import requests
import json
from pinecone import Pinecone
import streamlit as st
from streamlit_chat import message

with open("../config.json", "r") as f:
    config = json.load(f)
    
pc = Pinecone(api_key=config["PINECONE_API_KEY"])
url = "http://localhost:5000/chat"


# Using Pinecone Vector Database
index_name = "copmany"
st.title("ShiRui Info Tech Co.,Ltd")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for i, msg in enumerate(st.session_state["messages"]):
    message(msg["content"], is_user=msg["is_user"], key=f"msg_{i}")

def submit():
    user_query = st.session_state.input
    st.session_state["messages"].append({"content": user_query, "is_user": True})
    
    if user_query:
        resopnse = requests.post(url, json={"user_query": user_query})
        
        if resopnse.status_code == 200:
            # Append bot's response (replace with your LLM or logic)
            bot_response = resopnse.json().get("response", "")
            st.session_state["messages"].append({"content": bot_response, "is_user": False})
        elif resopnse:
            bot_response = "Server is under struction for now. Please try again later."
            st.session_state["messages"].append({"content": bot_response, "is_user": False})
        
    st.session_state.input = ""

user_query = st.text_input("You: ", "", key="input", placeholder="Type your message here...", on_change=submit)