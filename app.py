import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Streamlit UI
st.title("📄🤖 Chat with Your PDFs")

# Sidebar settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter API Key", type="password")
model = st.sidebar.selectbox("Select ChatGPT Model", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"])

# Context Mode Toggle
context_mode = st.sidebar.radio(
    "Choose Response Mode:",
    ["Strict Context Mode (PDF only)", "Flexible Mode (PDF + General Knowledge)", "Outside Context Only"]
)

# File Upload (Multiple PDFs)
st.sidebar.subheader("Upload Files")
uploaded_pdfs = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if api_key:
    st.sidebar.success("API Key saved (hidden for security)")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def process_pdfs(uploaded_files, api_key, model):
    temp_file_paths = []
    all_documents = []
    
    try:
        for uploaded_file in uploaded_files:
            temp_file_path = f"temp_{uploaded_file.name}"
            temp_file_paths.append(temp_file_path)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_loader = PyPDFLoader(temp_file_path)
            pdf_content = pdf_loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
            final_document = splitter.split_documents(pdf_content)
            all_documents.extend(final_document)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_database = FAISS.from_documents(all_documents, embeddings)
        retriever = vector_database.as_retriever()

        openai.api_key = api_key
        llm = ChatOpenAI(model=model)

        convo_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=memory
        )

        return convo_chain  # Return chatbot chain
    
    finally:
        for temp_file_path in temp_file_paths:
            os.remove(temp_file_path)



# Load chatbot based on selected mode
if uploaded_pdfs and api_key:
    convo_chain = process_pdfs(uploaded_pdfs, api_key, model)

user_input = st.text_input("🗣 **Ask a question:**", placeholder="Type your query here...")

if user_input:
    if context_mode == "Strict Context Mode (PDF only)":
        if uploaded_pdfs:
            response = convo_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            answer = response.get("answer", "❌ No valid response received.")
        else:
            answer = "⚠️ No PDFs uploaded. Please upload a PDF to use this mode."

    elif context_mode == "Flexible Mode (PDF + General Knowledge)":
        if uploaded_pdfs:
            response = convo_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})
            answer = response.get("answer", "❌ No valid response received.")
        else:
            openai.api_key = api_key
            llm = ChatOpenAI(model=model)
            answer = llm.predict(user_input)  # Changed from invoke() to predict()

    else:  # Outside Context Only Mode
        openai.api_key = api_key
        llm = ChatOpenAI(model=model)
        answer = llm.predict(user_input)  # Changed from invoke() to predict()

    # Update chat history
    st.session_state.chat_history.insert(0, (user_input, answer))

    # Display chat history
    for question, response in st.session_state.chat_history:
        st.markdown(f"**🧑‍💬 You:** {question}")
        st.markdown(f"**🤖 AI:** {response}")
