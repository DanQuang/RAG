# RAG (Retrieval-Augmented Generation) Chatbot

A powerful AI chatbot that uses Retrieval-Augmented Generation (RAG) to provide context-aware responses based on your documents. This application allows users to upload PDF or DOCX files and ask questions about their content.

## Features

- 📄 Document Upload: Support for PDF and DOCX files
- 🔍 Context-Aware Responses: Uses RAG to provide accurate answers based on document content
- 💬 Interactive Chat Interface: Built with Streamlit for a user-friendly experience
- 📚 Chat History: Maintains conversation context for better responses
- 🔄 Real-time Document Processing: Instant vectorization and indexing of uploaded documents

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone [<your-repository-url>](https://github.com/DanQuang/RAG)
cd RAG
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:

```bash
cp .env.example .env
```

Then edit the `.env` file with your API keys and model preferences.

## Usage

1. Start the application:

```bash
streamlit run src/main.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload your document using the sidebar

4. Start asking questions about your document!

## Project Structure

```
RAG/
├── src/
│   ├── main.py           # Main Streamlit application
│   ├── RAG.py            # Core RAG implementation
│   ├── VectorStore.py    # Vector store management
│   └── utils.py          # Utility functions
├── requirements.txt      # Project dependencies
├── .env.example         # Example environment variables
└── README.md            # Project documentation
```

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: The OpenAI model to use (e.g., "gpt-3.5-turbo")
- `EMBEDDING_MODEL_NAME`: The OpenAI embedding model to use (e.g., "text-embedding-ada-002")

## Dependencies

- langchain
- langchain-community
- langchain-openai
- openai
- faiss-cpu
- pypdf
- streamlit
- streamlit-pdf-viewer
- python-dotenv
