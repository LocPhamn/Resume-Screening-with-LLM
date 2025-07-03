import base64
import io
import os
import re
import streamlit as st
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import preprocess

if __name__ == '__main__':
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    embedder = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", )


