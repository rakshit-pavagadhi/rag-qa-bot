import os
import shutil
from typing import List, Tuple, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM

INDEX_PATH = "./faiss_index"

class RAGEngine:
    def __init__(self):
        # 1. HuggingFace Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None
        self.chat_history = []
        
        # Load index from disk if present
        if os.path.exists(INDEX_PATH):
            try:
                self.vector_store = FAISS.load_local(
                    INDEX_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("Loaded existing FAISS index from disk.")
            except Exception as e:
                print(f"Could not load local index: {e}")

        # 2. Local Ollama LLM
        self.llm = OllamaLLM(model="llama3.2", temperature=0.0)

    def load_document(self, file_path: str):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".txt", ".md"]:
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        return loader.load()

    def reset_index(self) -> str:
        """Deletes vector index from disk and memory."""
        self.vector_store = None
        self.chat_history = []
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)
        return "Vector database and chat history successfully purged!"

    def process_and_index_documents(self, file_paths: List[str]) -> str:
        all_docs = []
        for path in file_paths:
            docs = self.load_document(path)
            all_docs.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(all_docs)

        # Clear existing disk index to avoid index pollution
        if os.path.exists(INDEX_PATH):
            shutil.rmtree(INDEX_PATH)

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(INDEX_PATH)
        self.chat_history = []
        
        return f"Successfully processed {len(file_paths)} file(s) into {len(chunks)} context chunks!"

    def answer_question(self, query: str) -> Tuple[str, str, List[Dict[str, Any]]]:
        if not self.vector_store:
            return "Please upload and index documents before asking questions.", "None", []

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 9}
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = (
            "You are an expert factual QA assistant. Your job is to answer the user's question "
            "thoroughly, accurately, and strictly using ONLY the provided context snippets.\n\n"
            "CRITICAL GROUNDING RULES:\n"
            "1. DO NOT invent, assume, or extrapolate information not directly stated in the context.\n"
            "2. If the answer cannot be fully derived from the provided context, state clearly what "
            "information is present and explicitly note what is missing.\n\n"
            "STRUCTURAL & RETRIEVAL INSTRUCTIONS:\n"
            "1. SCOPE & OVERARCHING RULES: Read all context snippets carefully. Identify any overarching "
            "principles, global rules, core definitions, or mandatory frameworks "
            "that govern the topic.\n"
            "2. INTEGRATED SYNTHESIS: Directly connect top-level rules with specific sub-sections, controls, "
            "or itemized criteria.\n"
            "3. CROSS-REFERENCING: If the query asks about relationships across multiple sections or clauses, "
            "explain the explicit flow or dependency between them rather than providing isolated lists.\n"
            "4. DIRECT RESPONSE: Provide a concise, well-structured response using bold headers or bullet points. "
            "Do not include meta-commentary, prompt artifacts, or references to prompt rules.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        response = rag_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
        })

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response["answer"]))

        sources = []
        snippets = []
        for i, doc in enumerate(response["context"]):
            source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
            page_num = doc.metadata.get("page", "N/A")
            source_str = f"Page {page_num} from {source_file}"
            sources.append(source_str)
            
            snippets.append({
                "chunk_id": i + 1,
                "source": source_str,
                "text": doc.page_content.strip()
            })

        formatted_sources = "\n".join([f"- {s}" for s in set(sources)]) if sources else "None"
        return response["answer"], formatted_sources, snippets

    def clear_history(self):
        self.chat_history = []