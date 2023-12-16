# streamlit_app.py - GUI for chatbot.

import streamlit as st
from api import create_docsearch, query_api, DEFAULT_DATA_DIR, PROMPT_PREAMBLE

def app():
    st.title('2FTS Chatbot')
    st.write("Welcome to the 2FTS Chatbot! This chatbot is designed to help you find answers to your questions about 2FTS documentation.")
    st.write("To get started, type your question in the chat window below and press Enter.")
    st.divider()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input.
    if raw_prompt := st.chat_input(placeholder="Enter your query here..."):
        # Display user input and response.
        with st.chat_message("user"):
            st.markdown(raw_prompt)

        # Add prompt preamble.
        prompt = raw_prompt + PROMPT_PREAMBLE

        # Respond to user input.
        with st.chat_message("assistant"):
            with st.status(label="Finding answer...", expanded=True, state="running") as status:
                # Create docsearch database.
                st.write("Fetching vector embeddings...")
                # Check if docsearch variable exists.
                if "docsearch" not in st.session_state:
                    # Create docsearch database.
                    st.write("Creating vector database...")
                    st.session_state.docsearch = create_docsearch(DEFAULT_DATA_DIR)

                # Query QMSBot and display the response
                st.write("Querying QMSBot...")
                response = query_api(prompt, st.session_state.docsearch)

                # Update status.
                status.update(label="Found answer!", state="complete", expanded=False)

            # Write response to the chat window as the assistant.
            st.write(response)

        # Add user input to chat history.
        st.session_state.messages.append({"role": "user", "content": raw_prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    # In CLI: "python -m streamlit run src/vgs-chatbot/streamlit_app.py"
    app()
