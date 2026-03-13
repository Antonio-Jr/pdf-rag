"""Chat interface components for the Streamlit UI.

Provides functions to render the conversation history and to handle
the assistant's streaming response with a live status indicator.
"""

import streamlit as st

from ui.services.api_client import APIClient


def render_chat_history(messages):
    """Render all previous messages in the Streamlit chat container.

    Args:
        messages: List of message dictionaries, each containing
                  ``"role"`` (``"user"`` or ``"assistant"``) and
                  ``"content"`` keys.
    """
    for message in messages:
        with st.chat_message("message['role']"):
            st.markdown(message["content"])


def handle_assistant_response(api_client: APIClient, prompt: str, thread_id: str):
    """Stream the assistant's response and display it in real time.

    Opens a streaming connection to the chat API, renders tokens
    as they arrive, and updates the status widget on completion
    or error.

    Args:
        api_client: The API client instance used for backend calls.
        prompt: The user's current query text.
        thread_id: Conversation thread identifier for memory continuity.

    Returns:
        The full response text assembled from the stream, or ``None``
        if an error occurred.
    """
    with st.chat_message("assistant"):
        with st.status("Analyzing user request...", expanded=False) as status:
            placeholder = st.empty()
            full_response = ""

            try:
                with api_client.chat_stream(prompt, thread_id) as stream:
                    if stream.status_code != 200:
                        st.error(f"Something went wrong: {stream.status_code}")
                        status.update(label="Error!", state="error")
                        return

                    def stream_generator():
                        """Yield decoded text chunks from the streaming response."""
                        for chunk in stream.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk

                    full_response = placeholder.write_stream(stream_generator())
                    status.update(label="Agent reply", state="complete", expanded=True)
            except Exception as e:
                st.error(f"Connection failed with error: {e}")
                status.update(label="Connection Error!", state="error")

            return full_response
