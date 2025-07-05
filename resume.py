import base64
import glob
import io
import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings




if __name__ == '__main__':
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    embedder = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder", )


