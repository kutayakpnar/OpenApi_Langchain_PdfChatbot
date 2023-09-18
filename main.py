import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")


# Function to process the uploaded PDF
def process_pdf_with_langchain(uploaded_file, question):
    raw_text = ''

    # Read the uploaded PDF file
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Split the text using Character Text Splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    document_search = FAISS.from_texts(texts, embeddings)

    # Load the Langchain QA chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Perform question answering
    docs = document_search.similarity_search(question)
    answers = chain.run(input_documents=docs, question=question)

    return answers

# Streamlit app
st.title("DAVON PDF CHATBOT")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
question = st.text_input("Enter a question related to this file:")

if uploaded_file is not None and question:
    if st.button("Process"):
        answers = process_pdf_with_langchain(uploaded_file, question)

        if answers:
            st.header("Answers:")
            combined_answer = ''.join(answers)
            st.write(combined_answer)
        else:
            st.warning("No answers found.")
