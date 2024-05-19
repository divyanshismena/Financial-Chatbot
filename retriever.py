from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS, Chroma, Qdrant
from qdrant_client import QdrantClient
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

class CreateBM25Retriever:
    def __init__(self, docs):
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        with open('bm25retriever.pkl', 'wb') as outp:
            pickle.dump(self.bm25_retriever, outp, pickle.HIGHEST_PROTOCOL)

class Retriever:
    def __init__(self, db,per_dir,embeddings, strategy, k, collection_name="mydocuments"):
        self.db = db
        self.strategy = strategy
        self.per_dir = per_dir
        if self.db == 'faiss':
            self.db_ = FAISS.load_local(self.per_dir, embeddings, allow_dangerous_deserialization=True)
        elif self.db == 'chroma':
            self.db_ = Chroma(persist_directory=self.per_dir, embedding_function=embeddings)
        elif self.db == 'qdrant':
            self.db_ = Qdrant(client=QdrantClient(path=self.per_dir), collection_name=collection_name, embeddings=embeddings)
        elif self.db == 'pinecone':
            self.db_ = PineconeVectorStore(pinecone_api_key=os.getenv("PINECONE_API_KEY"),index_name=collection_name, embedding=embeddings)
        self.retriever = self.db_.as_retriever(search_kwargs={"k": k})

        if strategy == 'ensemble':
            with open('bm25retriever.pkl', 'rb') as inp:
                self.bm25_retriever = pickle.load(inp)
            self.bm25_retriever.k = k
            self.retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retriever],
                                       weights=[0.5, 0.5])
            
    def get_docs(self, query):
        return self.retriever.get_relevant_documents(query)
    
    def get_context(self, query):
        docs = self.get_docs(query)
        context = ""
        # src = []
        for txt in docs:
            context += '\n\n'+txt.page_content + "\n" + "Source: "+txt.metadata['source']
        #     src.append(txt.metadata['source'])
        # src = max(set(src), key=src.count)
        return context

