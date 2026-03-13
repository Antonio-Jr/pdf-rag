"""Streamlit application entry point for Document Intelligence.

Initializes session state (API client, message history, thread ID),
renders the sidebar and chat interface, and handles the main
chat input loop with streaming assistant responses.
"""

import streamlit as st
import uuid
from services.api_client import APIClient
from components.sidebar import render_sidebar
from components.chat_interface import render_chat_history
from ui.components.chat_interface import handle_assistant_response

st.set_page_config(page_title="Document Intel", page_icon="📄")

if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

render_sidebar(st.session_state.api_client, st.session_state.thread_id)

st.title("🤖 Document Intelligence")
render_chat_history(st.session_state.messages)

if prompt := st.chat_input("Ask about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = handle_assistant_response(
        st.session_state.api_client, prompt, st.session_state.thread_id
    )

    if response:
        st.session_state.messages.append({"role": "assistant", "content": response})
