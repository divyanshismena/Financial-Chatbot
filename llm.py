from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
import vertexai
from langchain_google_vertexai import ChatVertexAI

from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class LLM:
    def __init__(self, llm, model=None):
        if llm == 'gemini':
            if model is None:
                model = "gemini-pro"
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.3)
        elif llm == 'vertex':
            vertexai.init(project="website-254017", location="us-central1")
            if model is None:
                model = "gemini-1.5-pro-preview-0514"
            self.llm = ChatVertexAI(model_name=model, temperature=0)
        elif llm == 'openai':
            if model is None:
                model = 'gpt-3.5-turbo-0125'
            # ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
            self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=model)

        elif llm == 'mixtral':
            model = "mixtral-8x7b-32768"
            self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROK_API_KEY"), model_name=model)

        elif llm == 'llama':
            if model is None:
                model = 'llama3-8b-8192'
            self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROK_API_KEY"), model_name=model)

    def get_llm(self):
        return self.llm



