import streamlit as st

from ui.services.api_client import APIClient


def render_sidebar(api_client: APIClient, thread_id: str):
    with st.sidebar:
        st.title("📄 Configurations")

        if api_client.check_health():
            st.success("API Online")
        else:
            st.error("API Offline")

        st.divider()
        uploaded_files = st.file_uploader(
            "Upload documents (PDF)", type="pdf", accept_multiple_files=True
        )

        if st.button("Upload", use_container_width=True):
            if uploaded_files:
                with st.spinner("Uploading..."):
                    files = [
                        ("files", (f.name, f.read(), "application/pdf"))
                        for f in uploaded_files
                    ]
                    response = api_client.upload_files(files, thread_id)
                    if response.status_code == 200:
                        st.toast("Successfully Uploaded!")
                    else:
                        st.error("Something went wrong")
