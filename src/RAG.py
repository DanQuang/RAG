import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

load_dotenv()

def loadLLMAndEmbedding():
    llm_model_name = os.getenv("MODEL_NAME")
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")
    llm = ChatOpenAI(model= llm_model_name)
    embeddings = OpenAIEmbeddings(model= embedding_model_name)
    
    return llm, embeddings

def getContextSystemPrompt():
    system_prompt = """Với lịch sử trò chuyện và câu hỏi mới nhất của người dùng, \
    xem xét ngữ cảnh trong lịch sử cuộc trò chuyện, hãy tạo một câu hỏi độc lập dựa trên \
    các thông tin đó. KHÔNG được trả lời câu hỏi, chỉ cần xây dựng lại câu hỏi hoặc trả lại \
    câu hỏi ban đầu."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return contextualize_q_prompt

def createHistoryChain(llm, vector_store, ContextSystemPrompt):
    retriever = vector_store.as_retriever()
    return create_history_aware_retriever(llm= llm,
                                          retriever= retriever,
                                          prompt= ContextSystemPrompt)
    
def getQASystemPrompt():
    system_prompt = """Với các thông tin được cung cấp, hãy đưa cho tôi một câu trả lời chính xác \
        và chi tiết \
        CONTEXT: {context}"""
        
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return qa_prompt

def createRAGChain(llm, HistoryChain, QASystemPrompt):
    qa_chain = create_stuff_documents_chain(llm= llm,
                                            prompt= QASystemPrompt)
    return create_retrieval_chain(retriever= HistoryChain,
                                  combine_docs_chain= qa_chain)
    
def createChatHistory():
    # Initialize History chat to save all history
    msgs = ChatMessageHistory()
    
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn")
        
    return msgs