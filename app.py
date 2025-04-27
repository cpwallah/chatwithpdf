import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai  # Correctly import google.generativeai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store (using FAISS)
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Choose embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the index locally

# Function to create a conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
     Answer the question as detailed as possible from the provided content, 
     make sure to provide all the details. If the answer is not in context, just say: 
     "Answer is not available in context."
     Context:\n{context}\n
     Question:\n{question}\n

     Answer:
     """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.3)  # Use available model
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and get the model response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Use embedding model

    try:
        # Load the FAISS index locally with dangerous deserialization flag enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        # Generate response using the conversational chain
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Display the answer
        st.write("Reply: ", response["output_text"])

    except Exception as e:  # Catching all exceptions here
        st.error(f"An error occurred: {e}")

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini API")
    st.header("Chat with PDF using Google Gemini💬")

    # Get user input for the question
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Process the question if the user enters one
    if user_question:
        user_input(user_question)

    # Sidebar to upload PDF documents
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

# Run the Streamlit app
if __name__ == "__main__":
    main()
