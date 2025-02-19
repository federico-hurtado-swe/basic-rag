
from dotenv import load_dotenv
load_dotenv()  # Load environment variables as early as possible

import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")  # Explicitly set the API key

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Updated package as per deprecation warning
from langchain_community.vectorstores import FAISS  # In-memory vector store
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

# function that takes pdf documents and returns their contents in one long string
def get_pdf_text(docs):
    text = "" # variable to return
    for doc in docs:

        pdf_reader = PdfReader(doc) # note: this creates PAGES of pdf docs that we can read -> must loop through the pages
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

# given a string of text, returns a list of chunks of text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, # number of characters in each chunk 
        chunk_overlap=200, # overlap characters from chunks to prevent data loss
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectorestore = FAISS.from_texts(texts=chunks, embedding=embeddings) # generate DB
    return vectorestore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(), 
        memory = memory
    )

    return conversation_chain


def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.write(response["answer"])


def main():

    load_dotenv() # load API keys

    if "conversation" not in st.session_state: 
        st.session_state.conversation = None

    st.set_page_config(page_title="Chat with multiple PDF", page_icon=":books:")
    st.header("Ask any question")
    user_question = st.text_input("Ask question here:")

    if user_question: 
        handle_user_question(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload files here and click on process documents", accept_multiple_files=True
            )
        if st.button("process documents"):
            with st.spinner("processing....."): # add spinning wheel while processing is ongoing
                # turn pdf to text
                raw_text = get_pdf_text(pdf_docs)
                
                # chunk text into pieces
                text_chunks = get_text_chunks(raw_text)
            

                # create embeddings for each chunk of text and store in vector DB
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store) # helps access object outside of sidebar

                st.write("proccessed data")


if __name__ == '__main__':
    main()