from langchain.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import FAISS

@st.cache_resource
def load_pipeline_model():
    model_name = ""
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto') # can ignore device_map if it does not work well
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_generation_pipeline = pipeline(model= model,
                                        tokenizer= tokenizer,
                                        task= "text-generation")
    my_pipeline = HuggingFacePipeline(pipeline= text_generation_pipeline)
    return my_pipeline

@st.cache_resource
def read_vectors_db():
    # Embeding
    vector_db_path = ""
    model_name = ""
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db

# Tao simple chain
def create_qa_chain(llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False
    )
    return llm_chain

db = read_vectors_db()
llm = load_pipeline_model()
chain = create_qa_chain()

st.title("Ask me something 👀😈")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Pass your prompt here.")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user',
                                      'content': prompt})
    
    response = chain.invoke({"query": prompt})

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role': 'assistant',
                                      'content': response})