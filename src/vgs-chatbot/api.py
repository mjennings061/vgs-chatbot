# api.py - A chatbot to help with 2FTS documentation.

import os
import logging
import sys
from pathlib import Path
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv

# Constants.
DEBUG = False   # Set to True to force docsearch database to be recreated.
VECTOR_DATAFILE = "vector_database"
DEFAULT_DATA_DIR = Path(Path(__file__).resolve().parent.parent.parent, "data")
PROMPT_PREAMBLE = "\nAlso, show me where I can find the answer in the documentation."

# Retrieve OpenAI API key.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def database_exists() -> bool:
    """Check if docsearch database exists."""
    return Path(VECTOR_DATAFILE).exists()


def extract_text_from_file(file: str) -> str:
    """Extract text from the file."""
    # Read the file and split by page.
    reader = UnstructuredFileLoader(file)
    document_contents = reader.load()

    # Extract text from the file.
    raw_text = ""
    for page in document_contents:
        raw_text += page.page_content + "\n"

    return raw_text


def create_docsearch(document_dir=DEFAULT_DATA_DIR):
    """Create docsearch database from text files."""

    # Create embeddings object for OpenAIs. 
    embeddings = OpenAIEmbeddings()

    # Check if docsearch database already exists.
    if database_exists() and not DEBUG:
        # Load docsearch database.
        docsearch = FAISS.load_local(VECTOR_DATAFILE, embeddings)
        return docsearch
    
    # Get PDF files in the directory.
    pdf_files = list(document_dir.glob("*.pdf"))

    # Extract text from PDF files.
    raw_text = ""
    for pdf_file in pdf_files:
        raw_text += extract_text_from_file(pdf_file)

    # TODO: Use spacy to extract sentences.
    # TODO: Word document text extraction.
    # TODO: Filter out non-ASCII characters, except for newlines.

    # Split text by paragraph with overlap.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI.
    docsearch = FAISS.from_texts(texts, embeddings)

    # Save docsearch database.
    docsearch.save_local(VECTOR_DATAFILE)

    return docsearch


def query_api(query_input, docsearch) -> str:
    """Query API."""

    # Create chain.
    chain = load_qa_chain(
        OpenAI(),
        chain_type="stuff"
    )

    # Search for similar split elements of text.
    n_similar_texts = 5
    docs = docsearch.similarity_search(query_input, n_similar_texts)

    # Query the LLM to make sense of the related elements of text.
    response = chain.run(input_documents=docs, question=query_input)
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
    docsearch = create_docsearch(DEFAULT_DATA_DIR)

    # Query API.
    response = query_api(query_input="Where do I find how to write a quarterly summary?", 
                        docsearch=docsearch)
    logging.info(response)
