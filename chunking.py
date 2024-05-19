from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, strategy='rcs', chunk_size=10000, chunk_overlap=1000):
        self.strategy=strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def recursive_char_splitter(self, texts, metas):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        documents = text_splitter.create_documents(texts, metadatas=metas)
        return documents
    
    def hier_splitter(self, texts, metas):
        pass

    def get_chunks(self, texts, metas):
        docs = []
        for text, meta in zip(texts, metas):
            if self.strategy=='rcs':
                docs += self.recursive_char_splitter(text, meta)
            elif self.strategy=='hr':
                pass

        return docs