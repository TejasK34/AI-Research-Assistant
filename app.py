import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Title
st.title("📚 AI Research Assistant")

# Sidebar
st.sidebar.title("AI Research Assistant")

feature = st.sidebar.selectbox(
    "Choose Feature",
    ["Ask Questions", "Summarize Paper"]
)

# Upload PDF
uploaded_file = st.file_uploader("Upload Research Paper", type="pdf")

if uploaded_file is not None:

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector database
    db = FAISS.from_documents(docs, embeddings)

    # Ask Questions Feature
    if feature == "Ask Questions":

        question = st.text_input("Ask a question about the research paper")

        if question:
            results = db.similarity_search(question, k=3)

            st.subheader("📖 Answer")

            for i, result in enumerate(results):
                st.write(f"Result {i+1}:")
                st.write(result.page_content)
                st.write("---")

    # Summarize Feature
    if feature == "Summarize Paper":

        if st.button("Generate Summary"):

            text = " ".join([doc.page_content for doc in docs])
            summary = text[:1000]

            st.subheader("📑 Paper Summary")
            st.write(summary)

else:
    st.info("Please upload a research paper to start.")