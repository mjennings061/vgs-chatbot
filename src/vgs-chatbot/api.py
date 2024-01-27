""" api.py - A chatbot to help with 2FTS documentation.

This module contains the functions to create the docsearch database
and query the API.

Example:
    from api import create_vectorstore, query_api
    vectorstore = create_vectorstore()
    response = query_api("Where do I find how to write a quarterly summary?",
                         docsearch)

"""

import os
import logging
import textwrap
from pathlib import Path

# Langchain libraries.
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Constants.
DEBUG = False   # Set to True to force docsearch database to be recreated.
VECTOR_DATAFILE = "vector_database"
DEFAULT_DATA_DIR = Path(Path(__file__).resolve().parent.parent.parent, "data")

# Retrieve OpenAI API key.
openai_api_key = os.getenv("OPENAI_API_KEY")


def database_exists() -> bool:
    """Check if docsearch database exists.

    Returns:
        bool: True if docsearch database exists."""
    return Path(VECTOR_DATAFILE).exists()


def extract_text_from_file(file: str) -> list:
    """Extract text from the file and process named entities.

    Args:
        file (str): Path to file.

    Returns:
        list [Document()]: List of text chunks."""
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


def create_vectorstore(document_dir=DEFAULT_DATA_DIR):
    """Create docsearch database from text files.

    Args:
        document_dir (Path): Path to directory containing
        PDF files.

    Returns:
        vectorstore (FAISS): Vectorstore object."""

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
        logging.info("Processing %s", pdf_file)
        splits = extract_text_from_file(pdf_file)
        processed_text.extend(splits)

    # Download embeddings from OpenAI.
    vectorstore = FAISS.from_documents(  # pylint: disable=no-member
        documents=processed_text,
        embedding=embeddings
    )

    # Save docsearch database.
    vectorstore.save_local(VECTOR_DATAFILE)
    return vectorstore


def form_prompt_with_context(user_input, chat_history):
    """Form prompt with the entire conversation history as context."""
    # Get all comments from the user in the chat history.
    user_history = [
        message["content"]
        for message in chat_history if message["role"] == "user"
    ]

    # Use MessagesPlaceholder() to include user history as context.
    prompt = ChatPromptTemplate(
        user_input,
        MessagesPlaceholder(user_history),
        prompt_preamble="Previous questions for context:\n"
    )

    return prompt


def query_api(question, vectorstore, chat_history) -> str:
    """Query the API.

    Args:
        question (str): Question to ask.
        vectorstore (FAISS): Vectorstore object.
        chat_history (list)[dict]: List of chat messages.

    Returns:
        response (str): Response to the question."""
    # Setup vectorstore retriever with nearest 3 documents.
    n_similar_texts = 3
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': n_similar_texts}
    )

    # Initialise large language model.
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        temperature=0,
        openai_api_key=openai_api_key
    )

    if chat_history:
        # Add prompt preamble and contextualise question.
        question = form_prompt_with_context(question, chat_history)

    # Define the prompt.
    template = textwrap.dedent("""
        You are a helpful assistant to retrieve information from 2FTS
        documentation on the Viking glider.
        Answer the question and show where to find the answer
        including the document name and section in the documentation.
        Do not include index number. Documentation is below:
        -----
        {context}
        -----

        Question: {question}
    """)
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Query the LLM to make sense of the related elements of text.
    chain_response = rag_chain.invoke(question)
    return chain_response


if __name__ == '__main__':
    # Set up logging.
    logging.basicConfig(level=logging.INFO)

    # Set up debug when called as the main file.
    DEBUG = True

    # Create docsearch database.
    docsearch = create_vectorstore(DEFAULT_DATA_DIR)

    # Query API.
    conversation_history = [
        {"role": "user", "content": "What is a quarterly summary?"},
        {"role": "assistant", "content": """A quarterly summary is a summary
         of the quarterly activities."""}
    ]
    response = query_api(
        question="Where do I find how to write a quarterly summary?",
        vectorstore=docsearch,
        chat_history=conversation_history
    )
    logging.info(response)
