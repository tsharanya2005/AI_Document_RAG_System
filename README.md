# 📄 AI Document Q&A — RAG System

An AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)**.

Users can upload a PDF, ask questions about its content, and receive answers generated from the most relevant sections of the document. The application also displays the source pages and similarity scores used for the answer.

## 🚀 Features

* Upload a PDF document
* Extract text from the PDF
* Split document text into smaller chunks
* Generate embeddings using Sentence Transformers
* Store embeddings in a FAISS vector index
* Retrieve the most relevant document sections
* Generate answers using an LLM through OpenRouter
* Display retrieved source pages
* Display similarity scores
* Prevent answers from being generated when information is not available in the document
* Simple Streamlit interface
* Error handling for invalid or unreadable documents

## 🏗️ Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Index
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
OpenRouter LLM
    ↓
Answer + Sources
```

## 🛠️ Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Core application                |
| Streamlit             | Web interface                   |
| PyPDF                 | PDF text extraction             |
| Sentence Transformers | Text embeddings                 |
| all-MiniLM-L6-v2      | Embedding model                 |
| FAISS                 | Vector similarity search        |
| NumPy                 | Numerical operations            |
| Requests              | OpenRouter API requests         |
| python-dotenv         | Environment variable management |
| OpenRouter            | LLM API                         |

## 📁 Project Structure

```text
ai-document-rag-system/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── sample.pdf
```

## ⚙️ How the RAG System Works

### 1. PDF Text Extraction

The uploaded PDF is read using PyPDF and text is extracted page by page.

### 2. Chunking

The extracted text is divided into smaller overlapping chunks so that relevant portions can be retrieved efficiently.

The application uses:

```text
Chunk size: 700 characters
Overlap: 100 characters
Top retrieved chunks: 3
```

### 3. Embeddings

Each text chunk is converted into a numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 4. FAISS Vector Search

The generated embeddings are stored in a FAISS index.

When the user asks a question, the question is converted into an embedding and compared with the stored document embeddings using similarity search.

### 5. Answer Generation

The most relevant document chunks are provided to the LLM through OpenRouter as context.

The model is instructed to:

* use only the provided document context
* avoid inventing facts
* clearly state when the requested information cannot be found

### 6. Sources

The application displays the retrieved document pages, similarity scores, and relevant text so that users can see the information used to generate the answer.

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-document-rag-system.git
cd ai-document-rag-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the project directory:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

**Do not commit your `.env` file to GitHub.**

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

## 💡 Example Questions

After uploading a document, users can ask questions such as:

```text
What is the main purpose of this document?
What are the key points discussed?
What is RAG?
What are the benefits mentioned in the document?
```

For information that is not present in the uploaded document, the application is designed to respond:

```text
I couldn't find that information in the uploaded document.
```

## 🔍 Approach

The application follows a Retrieval-Augmented Generation (RAG) pipeline:

1. **PDF Text Extraction** – Extract text from the uploaded PDF page by page using PyPDF.
2. **Text Chunking** – Split the extracted text into smaller overlapping chunks for efficient retrieval.
3. **Embedding Generation** – Convert each chunk into a vector representation using the `all-MiniLM-L6-v2` Sentence Transformer model.
4. **Vector Indexing** – Store the embeddings in a FAISS index for similarity-based search.
5. **Question Retrieval** – Convert the user's question into an embedding and retrieve the most relevant document chunks.
6. **Answer Generation** – Pass the retrieved chunks as context to an LLM through the OpenRouter API.
7. **Source Display** – Show the generated answer along with the relevant document pages, similarity scores, and retrieved text.

The LLM is instructed to use only the retrieved document context and to state when the requested information cannot be found in the document.


## 🔮 Future Improvements

With additional development time, the system could be extended with:

* OCR support for scanned PDFs
* Persistent vector database
* Multi-document support
* Improved chunking strategies
* Reranking of retrieved results
* Chat history
* Streaming responses
* Retrieval evaluation
* Application deployment

## 🎯 Project Goal

The goal of this project is to demonstrate a practical RAG pipeline that connects document retrieval with LLM-based question answering while providing source information for greater transparency.
