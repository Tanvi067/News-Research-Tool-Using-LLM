# ===============================
# News Research Tool – WORKING VERSION
# ===============================

from dotenv import load_dotenv
import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate

# ===============================
# Load Environment Variables
# ===============================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OPENAI_API_KEY not found in .env file")
    st.stop()

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="📰 News Research Tool", layout="wide")
st.title("📰 News Research Tool")

urls = st.text_area(
    "Enter news article URLs (one per line)",
    height=150
)

question = st.text_input("Ask a question about the articles")

process = st.button("Process Articles")

# ===============================
# Main Logic
# ===============================

if process:

    if not urls.strip():
        st.error("Please enter at least one URL")
        st.stop()

    if not question.strip():
        st.error("Please enter a question")
        st.stop()

    # Convert text area into list
    url_list = [u.strip() for u in urls.split("\n") if u.strip()]

    # Load articles
    with st.spinner("Loading articles..."):
        loader = UnstructuredURLLoader(urls=url_list)
        documents = loader.load()

    if not documents:
        st.error("❌ Could not load articles. Check URLs.")
        st.stop()

    # Split text into chunks
    with st.spinner("Splitting text..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

    if not docs:
        st.error("❌ No text found after splitting.")
        st.stop()

    # Create embeddings + vector store
    with st.spinner("Creating embeddings..."):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # Custom Prompt
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful news research assistant.
Use the following summaries to answer the question.
If you don't know the answer, say "I don't know".

Summaries:
{summaries}

Question:
{question}

Answer:
"""
    )

    # QA Chain
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    # Run question
    with st.spinner("Answering question..."):
        result = qa_chain.invoke({"question": question})

    # Output
    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    st.write(result["sources"])
