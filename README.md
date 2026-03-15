# 🤖 Full-Stack AI Agent Dashboard

A modern, full-stack AI application demonstrating advanced Large Language Model (LLM) architectures. This project goes beyond basic chatbots by implementing **Retrieval-Augmented Generation (RAG)**, **Autonomous Tool Calling**, and **Stateful Conversational Memory**.

![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688) ![React](https://img.shields.io/badge/React-Frontend-61DAFB) ![Groq](https://img.shields.io/badge/Groq-Llama_3-f55036) ![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FFC107)

## ✨ Core Architectures Built

### 1. 📄 Document Q&A (RAG Engine)
Upload documents and talk directly to your data.
- **How it works:** Uses `SentenceTransformers` to convert text into embeddings, stores them in a local **ChromaDB** vector database, and performs semantic similarity searches to provide accurate, "open-book" answers.
- **Key Tech:** LangChain, Chroma, HuggingFace Embeddings.

### 2. 🌐 Web Agent (Tool Calling)
An autonomous agent that knows when to search the internet.
- **How it works:** Equipped with a Wikipedia search tool. The AI analyzes the user's prompt, decides if it lacks the factual knowledge, generates a precise search query, executes the Python tool, and synthesizes the live internet data into a natural response.
- **Key Tech:** Groq Tool Calling, Llama 3.3 70B, Wikipedia API.

### 3. 💬 Smart Chat (Stateful Memory)
A context-aware AI assistant with perfect conversational memory.
- **How it works:** Overcomes the standard "stateless" nature of REST APIs by managing and passing dynamic message arrays (chat history), allowing the AI to remember names, context, and previous instructions.
- **Key Tech:** Message array mapping, React state management.

## 🚀 Tech Stack

**Frontend:**
- React 18 + Vite
- Tailwind CSS (Premium Dark Mode UI)
- Lucide React (Icons)

**Backend:**
- FastAPI (Python)
- Pydantic (Data validation)
- Uvicorn (ASGI server)

**AI & Data:**
- Groq API (Ultra-fast inference)
- Llama 3.3 70B Versatile
- ChromaDB (Vector Database)

## 📦 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Groq API key ([Get it free](https://console.groq.com))

### Backend Setup

1. **Install Python dependencies:**
```bash
pip install fastapi uvicorn groq python-dotenv langchain-community chromadb sentence-transformers wikipedia
