# streamlit_app.py - GUI for chatbot.

import streamlit as st
from api import create_vectorstore, query_api, \
                form_prompt_with_context, database_exists


def fetch_response(prompt, chat_history):
    """Fetch response from the vgs-chatbot"""
    with st.status(
        label="Finding answer...",
        expanded=True,
        state="running"
    ) as status:
        # Create vectorstore database.
        st.write("Fetching vector embeddings...")
        # Check if vectorstore variable exists.
        if "vectorstore" not in st.session_state:
            if not database_exists():
                st.write(
                    """
                    Generating a new vector database
                    (this will take a while)...
                    """
                )
            # Create or fetch vectorstore database.
            st.session_state.vectorstore = create_vectorstore()

        # Query VGS bot and display the response.
        st.write("Querying VGS Bot...")
        response = query_api(
            question=prompt,
            vectorstore=st.session_state.vectorstore,
            chat_history=chat_history
        )

        # Collapse status message.
        status.update(label="Complete!", state="complete", expanded=False)

    return response


def app():
    st.title('2FTS Chatbot')
    st.write("""Welcome to the 2FTS Chatbot!
             This chatbot is designed to help you find answers to your
             questions about 2FTS documentation.
             To get started, type your question in the chat window
             below and press Enter.""")
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

        # Respond to user input.
        with st.chat_message("assistant"):
            # Query VGS bot and display the response.
            response = fetch_response(raw_prompt, chat_history)
            # Write response to the chat window as the assistant.
            st.write(response)

        # Add user input to chat history.
        st.session_state.chat_history.append({
            "role": "user",
            "content": raw_prompt
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })

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
