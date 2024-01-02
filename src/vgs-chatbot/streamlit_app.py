# streamlit_app.py - GUI for chatbot.

import streamlit as st
from api import create_docsearch, query_api, PROMPT_PREAMBLE, database_exists


def fetch_response(prompt):
    """Fetch response from the vgs-chatbot"""
    with st.status(label="Finding answer...", expanded=True, state="running") as status:
        # Create docsearch database.
        st.write("Fetching vector embeddings...")
        # Check if docsearch variable exists.
        if "docsearch" not in st.session_state:
            if not database_exists():
                st.write("Generating a new vector database...")
            # Create or fetch docsearch database.
            st.session_state.docsearch = create_docsearch()

        # Query VGS bot and display the response.
        st.write("Querying VGS Bot...")
        response = query_api(prompt, st.session_state.docsearch)
        
        # Collapse status message.
        status.update(label="Complete!", state="complete", expanded=False)

    return response


def form_prompt_with_context(user_input, chat_history):
    """Form prompt with the entire conversation history as context."""
    # Get all comments from the user in the chat history.
    user_history = [message["content"] for message in chat_history if message["role"] == "user"]

    # Join user history into a single string.
    if len(user_history) > 0:
        user_history = "\n".join(user_history)
        context = f"Previous questions for context:```\n{user_history}```\n\nNew question:\n"
    else:
        context = ""

    # Add the new user input and prompt preamble.
    prompt = f"{context}{user_input}{PROMPT_PREAMBLE}"
    return prompt


def app():
    st.title('2FTS Chatbot')
    st.write("""Welcome to the 2FTS Chatbot! This chatbot is designed to help you find answers to your questions about 2FTS documentation.
                To get started, type your question in the chat window below and press Enter.""")
    st.divider()

    # Initialize chat history.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input.
    raw_prompt = st.chat_input(placeholder="Enter your query here...")
    if raw_prompt:
        # Display user input and response.
        with st.chat_message("user"):
            st.markdown(raw_prompt)

        # Add prompt preamble.
        chat_history = st.session_state.chat_history
        prompt = form_prompt_with_context(raw_prompt, chat_history)

        # Respond to user input.
        with st.chat_message("assistant"):
            # Query VGS bot and display the response.
            response = fetch_response(prompt)
            # Write response to the chat window as the assistant.
            st.write(response)

        # Add user input to chat history.
        st.session_state.chat_history.append({"role": "user", "content": raw_prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display contact information.
    with st.sidebar:
        st.markdown("""
            ## Contact Information
            For any queries or support, please contact us at 
            [michael.jennings100@rafac.mod.gov.uk](mailto:michael.jennings100@rafac.mod.gov.uk)
        """)


if __name__ == '__main__':
    # In CLI: "python -m streamlit run src/vgs-chatbot/streamlit_app.py"
    app()
