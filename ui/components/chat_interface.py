import streamlit as st
from ui.services.api_client import APIClient


def render_chat_history(messages):
    for message in messages:
        with st.chat_message("message['role']"):
            st.markdown(message["content"])


def handle_assistant_response(api_client: APIClient, prompt: str, thread_id: str):
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
                        for chunk in stream.iter_content(
                            chunk_size=None, decode_unicode=True
                        ):
                            if chunk:
                                yield chunk

                    full_response = placeholder.write_stream(stream_generator())
                    status.update(label="Agent reply", state="complete", expanded=True)
            except Exception as e:
                st.error(f"Connection failed with error: {e}")
                status.update(label="Connection Error!", state="error")

            return full_response
