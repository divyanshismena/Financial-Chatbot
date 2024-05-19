import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

class Embeddings:

    '''
        google, models/embedding-001
        openai, openai
        cohere, cohere
        hf, all-MiniLM-L6-v2
        hf, BAAI/bge-large-en-v1.5
        hf, Alibaba-NLP/gte-large-en-v1.5, True
        ...
        ...
    '''

    def __init__(self, emb, model, trust_remote=False, normalize = False):
        self.emb=emb
        self.model = model
        self.trust_remote = trust_remote
        self.normalize = normalize
        self.embedding = self.get_embedding()
        self.seq_len = self.get_emb_len()

    def get_emb_len(self):
        return len(self.embedding.embed_query('hi how are you'))
    
    def google_embedding(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        embeddings = GoogleGenerativeAIEmbeddings(model = self.model)
        return embeddings
    
    def openai_embedding(self):
        embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        return embeddings_model

    def cohere_embedding(self):
        embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"))
        return embeddings_model

    def hf_embedding(self):
        model_args = {'trust_remote_code': True} if self.trust_remote else {}
        encode_args = {'normalize_embeddings': True} if self.normalize else {}
        embedding = HuggingFaceEmbeddings(model_name=self.model, model_kwargs = model_args, encode_kwargs = encode_args)
        return embedding

    def get_embedding(self):
        if self.emb == 'google':
            return self.google_embedding()
        elif self.emb == 'openai':
            return self.openai_embedding()
        elif self.emb == 'cohere':
            return self.cohere_embedding()
        elif self.emb == 'hf':
            return self.hf_embedding()