# 📚 Local Multi-Document RAG QA System

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) system built for multi-turn conversational question-answering over custom documents. The application runs **100% locally** using **Ollama (Llama 3.2)**, **HuggingFace Embeddings**, **FAISS**, and **LangChain v0.3+**, exposed through an intuitive **Gradio 5+** chat interface.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1C3C3C)
![Gradio](https://img.shields.io/badge/Gradio-5%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌟 Key Features

- **🔒 100% Offline & Private** — Operates entirely on local hardware. No external API keys or cloud dependencies required.
- **📄 Multi-Format Ingestion** — Native support for `.pdf`, `.docx`, `.txt`, `.md`, and `.csv` files.
- **🧠 History-Aware Retrieval** — Contextual question reformulation enables natural, multi-turn follow-up conversations.
- **⚡ High-Performance Vector Store** — Fast similarity search via a local FAISS index with persistent disk state.
- **🔍 Transparent Context Inspection** — Real-time visibility into the exact retrieved chunks and source pages used to ground each answer.
- **🧹 Granular Memory Control** — Independently clear conversation history or purge the vector database index on demand.

---

## 🏗️ Architecture Overview

```text
[ Document Ingestion ] → [ Chunking (Recursive) ] → [ HuggingFace Embeddings ] → [ FAISS Index ]
                                                                                        │
[ User Query ] → [ History-Aware Reformulator ] → [ Vector Retrieval (k=9) ] ──────────┤
                                                                                        ▼
[ Gradio UI ] ← [ Ollama Llama 3.2 ] ← [ Context-Grounded Prompt ] ←────────────────────┘
```

| Component | Details |
|---|---|
| **LLM Engine** | `llama3.2` via Ollama (`temperature=0.0` for deterministic, factual output) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2`, normalized vector embeddings |
| **Text Chunking** | Recursive character splitting — chunk size 1200, overlap 200 |
| **Vector Store** | FAISS (local, disk-persisted) |
| **Retrieval Chain** | History-aware retriever + stuff-documents QA chain (LangChain) |
| **UI Framework** | Gradio 5+, native chat message dictionaries |

---

## 📂 Project Structure

```text
.
├── app.py              # Gradio web interface and event handlers
├── rag_engine.py        # Core RAG orchestration pipeline (loaders, embeddings, FAISS, LangChain)
├── requirements.txt     # Project dependencies
├── faiss_index/          # Persisted vector store (auto-generated)
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher
- **Ollama** installed and running — [download here](https://ollama.com)

Pull the Llama 3.2 model before launching:

```bash
ollama pull llama3.2
```

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/local-rag-qa-system.git
cd local-rag-qa-system
```

**2. Create and activate a virtual environment**

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 💻 Usage

**1. Launch the application**

```bash
python app.py
```

**2. Open the web interface**

Navigate to [http://127.0.0.1:7860](http://127.0.0.1:7860) in your browser.

**3. Workflow**

| Step | Action |
|---|---|
| 1 | Upload target documents (`.pdf`, `.docx`, `.txt`, `.csv`) in the sidebar |
| 2 | Click **Index Documents** to chunk, embed, and store vectors on disk |
| 3 | Ask questions in the chat box — follow-ups are automatically contextualized |
| 4 | Expand **🔍 Inspect Retrieved Context Snippets** to view raw chunks and sources |

---

## 🛠️ Management Controls

| UI Button | Function |
|---|---|
| **Clear Chat Memory** | Wipes active conversation history while keeping document embeddings intact. Useful for starting a fresh sub-topic on the same document set. |
| **Purge Index** | Deletes the disk-persisted `./faiss_index` database and clears chat memory. Use when switching to a new document set. |

---

## 🧩 Tech Stack

`Python` · `LangChain` · `LangChain Community / Classic` · `FAISS` · `HuggingFace Sentence-Transformers` · `Ollama` · `Gradio`

---

## 🗺️ Roadmap

- [ ] Support for additional loaders (`.pptx`, `.html`, `.json`)
- [ ] Multi-collection / namespace support for isolated document sets
- [ ] Configurable chunking and retrieval parameters via UI
- [ ] Streaming token-by-token responses
- [ ] Docker Compose setup for one-command deployment

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

Built with [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com), [FAISS](https://github.com/facebookresearch/faiss), and [Gradio](https://github.com/gradio-app/gradio).