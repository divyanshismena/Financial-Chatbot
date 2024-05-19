import streamlit as st
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from data_extraction import Extraction
import nest_asyncio
from chunking import Chunker
from embeddings import Embeddings
from vectorstore import VectorDB
from retriever import Retriever, CreateBM25Retriever
from llm import LLM
from langchain_core.prompts import PromptTemplate
from chain import Chain
from streamlit_chat import message

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

nest_asyncio.apply()
ext = Extraction('fast')
chnk = Chunker(chunk_size=1000, chunk_overlap=200)
emb = Embeddings("hf", "all-MiniLM-L6-v2")
_llm = LLM('vertex').get_llm()
ch = Chain(_llm, st.session_state.buffer_memory)
conversation = ch.get_chain_with_history()

def query_refiner(conversation, query):
    prompt=f"Given the following user query and historical user queries, rephrase the users current query to form a meaningful and clear question.Previously user has asked the following: \n{conversation}\n\n User's Current Query: {query}. What will be the refined query? Only provide the query without any extra details or explanations.",
    ans = _llm.invoke(prompt).content
    return ans

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        # conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def main():
    inp_dir = "./inputs"
    db = 'pinecone'
    db_dir = 'pineconedb'
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF")

    response_container = st.container()
    textcontainer = st.container()
    ret = None
    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            if ret is None:
                ret = Retriever(db, db_dir, emb.embedding, 'ensemble', 5)
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                if len(st.session_state['responses']) != 0:
                    refined_query = query_refiner(conversation_string, query)
                else:
                    refined_query = query
                st.subheader("Refined Query:")
                st.write(refined_query)
                context = ret.get_context(refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
                # response += '\n' + "Source: " + src
            st.session_state.requests.append(query)
            st.session_state.responses.append(response) 

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        pdfs = []
        if pdf_docs:
            for pdf_file in pdf_docs:
                filename = pdf_file.name
                path = os.path.join(inp_dir,filename)
                with open(path, "wb") as f:
                    f.write(pdf_file.getvalue())
                pdfs.append(path)

            with st.spinner("Processing..."):
                texts, metas = ext.get_text(pdfs)
                docs = chnk.get_chunks(texts, metas)
                vs = VectorDB(db, emb.embedding, db_dir, docs=docs)
                bm = CreateBM25Retriever(docs)
                st.success("Done")

if __name__ == "__main__":
    main()