import streamlit as st
from RAG import *
from VectorStore import getVectorStore
from utils import *

def main():
    llm, embedding = loadLLMAndEmbedding()

    st.title("AI Chatbot Assistant 🔥")
    st.sidebar.title("Upload your document! (PDF, Docx, etc.)")
    uploaded_file = st.sidebar.file_uploader('Choose your file', type=['pdf', 'docx'])

    if uploaded_file is not None:
        st.success(f"Document name: {uploaded_file.name}")
        if "uploaded_file_name" not in st.session_state:
            st.session_state.uploaded_file_name = uploaded_file.name
        elif uploaded_file.name != st.session_state.uploaded_file_name:
            reset_chat_state()
            st.session_state.uploaded_file_name = uploaded_file.name
        st.sidebar.success("The document has been uploaded successfully!")

        st.sidebar.markdown("### PDF Preview")
        with st.sidebar:
            display_pdf(uploaded_file)

        # Kiểm tra và khởi tạo session_state nếu chưa có
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = getVectorStore(uploaded_file, embedding)

        if 'chat_history' not in st.session_state:
            context_sys_prompt = getContextSystemPrompt()
            history_chain = createHistoryChain(
                llm=llm,
                vector_store=st.session_state.vector_store,
                ContextSystemPrompt=context_sys_prompt
            )
            qa_sys_prompt = getQASystemPrompt()
            st.session_state.rag_chain = createRAGChain(llm, history_chain, qa_sys_prompt)
            st.session_state.chat_history = createChatHistory()
            st.session_state.messages = []  # Lưu lịch sử chat
            st.success("Đoạn chat mới đã được tạo! ❤")

        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        # User input section
        user_input = st.chat_input("Nhập câu hỏi của bạn tại đây...")

        if user_input:
            st.chat_message('user').markdown(user_input)
            st.session_state.messages.append({'role': 'user', 'content': user_input})

            # Add user message to chat history
            st.session_state.chat_history.add_user_message(user_input)

            # Generate response
            response = st.session_state.rag_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history.messages[1:]
            })

            st.chat_message('assistant').markdown(response["answer"])
            st.session_state.messages.append({'role': 'assistant', 'content': response["answer"]})

            # Add AI response to chat history
            st.session_state.chat_history.add_ai_message(response["answer"])

if __name__ == "__main__":
    main()