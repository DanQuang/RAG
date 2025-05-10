import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

# Display pdf
def display_pdf(file):
    binary_data = file.getvalue()
    pdf_viewer(input= binary_data,
               height= 800,
               pages_to_render= [1, 2, 3])
    
def reset_chat_state():
    """Reset all chat status when uploading new documents."""
    del st.session_state.vector_store
    del st.session_state.rag_chain
    del st.session_state.chat_history
    del st.session_state.messages