# api.py - A chatbot to help with 2FTS documentation.

import os
import logging
import sys
import re
import textwrap
from pathlib import Path

# Langchain libraries.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Constants.
DEBUG = False   # Set to True to force docsearch database to be recreated.
VECTOR_DATAFILE = "vector_database"
DEFAULT_DATA_DIR = Path(Path(__file__).resolve().parent.parent.parent, "data")
PROMPT_PREAMBLE = """\nAlso, show me where I can find the answer in
                the documentation."""

# Retrieve OpenAI API key.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def clean_text(text: str) -> str:
    """Clean text by removing newlines and extra spaces."""
    # Remove unwanted patterns (e.g., 'UNCONTROLLED COPY WHEN PRINTED')
    text_no_header = re.sub('UNCONTROLLED COPY WHEN PRINTED', '', text)

    return text_no_header


def database_exists() -> bool:
    """Check if docsearch database exists."""
    return Path(VECTOR_DATAFILE).exists()


def extract_text_from_file(file: str) -> list:
    """Extract text from the file and process named entities."""
    # Load file.
    reader = UnstructuredFileLoader(file_path=str(file))
    document_contents = reader.load()

    # Extract text from file.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    # Split text into chunks.
    splits = text_splitter.split_documents(document_contents)
    return splits


def format_docs(docs):
    """Format the documents for the docsearch database.

    Args:
        docs (list): List of Document objects.

    Returns:
        str: Formatted documents."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_vectorstore(document_dir=DEFAULT_DATA_DIR):
    """Create docsearch database from text files."""

    # Create embeddings object for OpenAIs.
    embeddings = OpenAIEmbeddings()

    # Check if docsearch database already exists.
    if database_exists() and not DEBUG:
        # Load docsearch database.
        vectorstore = FAISS.load_local(VECTOR_DATAFILE, embeddings)
        return vectorstore

    # Get PDF files in the directory.
    pdf_files = list(document_dir.glob("*.pdf"))

    # Extract text from PDF files.
    processed_text = []
    for pdf_file in pdf_files:
        logging.info(f"Processing {pdf_file}")
        splits = extract_text_from_file(pdf_file)
        processed_text.extend(splits)

    # Download embeddings from OpenAI.
    vectorstore = FAISS.from_documents(
        documents=processed_text,
        embedding=embeddings
    )

    # Save docsearch database.
    vectorstore.save_local(VECTOR_DATAFILE)

    return vectorstore


def query_api(question, vectorstore) -> str:
    """Query API."""
    # Setup vectorstore retriever with nearest 5 documents.
    N_SIMILAR_TEXTS = 5
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': N_SIMILAR_TEXTS}
    )

    # Initialise large language model.
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0
    )

    # Define the prompt.
    TEMPLATE = textwrap.dedent("""
        You are a helpful assistant to retrieve information from 2FTS
        documentation on the Viking glider.
        Show where to find the answer including the
        document name and section in the documentation.
        Do not include index number. Documentation is below:
        -----
        {context}
        -----

        Question: {question}
    """)
    prompt = ChatPromptTemplate.from_template(TEMPLATE)

    rag_chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Query the LLM to make sense of the related elements of text.
    response = rag_chain.invoke(question)
    return response


if __name__ == '__main__':
    # Set up logging.
    logging.basicConfig(level=logging.INFO)

    # Set up debug when called as the main file.
    DEBUG = True

    # Get directory where the PDF files are stored.
    if len(sys.argv) > 1:
        # Command line argument.
        document_dir = Path(sys.argv[1])
    else:
        # Default (no command line arguments passed).
        document_dir = DEFAULT_DATA_DIR

    # Create docsearch database.
    docsearch = create_vectorstore(DEFAULT_DATA_DIR)

    # Query API.
    response = query_api(
        question="Where do I find how to write a quarterly summary?",
        vectorstore=docsearch
    )
    logging.info(response)
