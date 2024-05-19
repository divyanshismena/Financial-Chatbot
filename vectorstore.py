from langchain.vectorstores import FAISS, Chroma, Qdrant
from langchain_pinecone import PineconeVectorStore
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

class VectorDB:
    def __init__(self, db, embedding, per_dir='db', load_db=False, docs = [], collection_name="mydocuments"):
        self.dbtype = db
        if db == 'pinecone':
            os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")
        self.embedding = embedding
        self.per_dir = per_dir
        self.docs = docs
        self.collection_name = collection_name
        if load_db:
            self.db = self.load_db()
        else:
            self.db = self.build_db()

    def faiss_db(self):
        db = FAISS.from_documents(self.docs, self.embedding)
        db.save_local(self.per_dir)
        return db
    
    def chroma_db(self):
        db = Chroma.from_documents(self.docs, self.embedding, persist_directory=self.per_dir)
        db.persist()
        return db
    
    def qdrant_db(self):
        qdrant = Qdrant.from_documents(self.docs,self.embedding,path=self.per_dir,collection_name=self.collection_name,force_recreate=True)
        return qdrant
    
    def pinecone_db(self):
        pinecone = PineconeVectorStore.from_documents(self.docs, self.embedding, index_name=self.collection_name)
        return pinecone
    
    def load_db(self):
        if self.dbtype == 'faiss':
            return FAISS.load_local(self.per_dir, self.embedding, allow_dangerous_deserialization=True) 
        elif self.dbtype == 'chroma':
            return Chroma(persist_directory = self.per_dir, embedding_function = self.embedding) 
        elif self.dbtype == 'qdrant':
            return Qdrant(client=QdrantClient(path=self.per_dir), collection_name=self.collection_name, embeddings=self.embeddings)
        elif self.dbtype == 'pinecone':
            return PineconeVectorStore(index_name=self.collection_name, embedding=self.embedding)
        else:
            pass
            

    def build_db(self):
        if self.dbtype == 'faiss':
            return self.faiss_db()
        elif self.dbtype == 'chroma':
            return self.chroma_db() 
        elif self.dbtype == 'qdrant':
            return self.qdrant_db()
        elif self.dbtype == 'pinecone':
            return self.pinecone_db()
        else:
            pass 