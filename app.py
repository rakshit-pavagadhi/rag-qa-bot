import gradio as gr
import shutil
import os
from rag_engine import RAGEngine

engine = RAGEngine()

def handle_document_upload(files):
    if not files:
        return "No files uploaded."
    file_paths = [file.name for file in files]
    try:
        status = engine.process_and_index_documents(file_paths)
        return status
    except Exception as e:
        return f"Error processing documents: {str(e)}"

def user_submit(user_message, history):
    if not user_message.strip():
        return "", history, "**Sources Used:** None", "No context retrieved yet."
    
    answer, sources, snippets = engine.answer_question(user_message)
    
    # Native Gradio 5/6 message dictionary structure
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    formatted_snippets = ""
    for item in snippets:
        formatted_snippets += f"### Chunk {item['chunk_id']} ({item['source']})\n"
        formatted_snippets += f"```text\n{item['text']}\n```\n\n---\n"
        
    if not formatted_snippets:
        formatted_snippets = "No context retrieved."

    sources_markdown = f"**Sources Used:**\n{sources}" if sources != "None" else "**Sources Used:** None"

    return "", history, sources_markdown, formatted_snippets

def clear_conversation():
    engine.clear_history()
    return [], "**Sources Used:** None", "No context retrieved yet."

def reset_vector_database():
    status = engine.reset_index()
    return [], "**Sources Used:** None", "No context retrieved yet.", status

# Interface Block
with gr.Blocks(title="Enterprise Conversational RAG QA") as demo:
    gr.Markdown("# 📚 Enterprise Multi-Document RAG QA System")
    gr.Markdown("Upload `.pdf`, `.docx`, `.txt`, or `.csv` files and engage in multi-turn conversational search.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Target Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".csv"]
            )
            upload_btn = gr.Button("Index Documents", variant="primary")
            status_box = gr.Textbox(label="Indexing Status", interactive=False)
            
            gr.Markdown("---")
            with gr.Row():
                clear_chat_btn = gr.Button("Clear Chat Memory", variant="secondary")
                reset_db_btn = gr.Button("Purge Index", variant="stop")
            
            sources_box = gr.Markdown(value="**Sources Used:** None")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation History", height=450)
            msg_input = gr.Textbox(
                label="Ask a Question", 
                placeholder="Ask follow-up questions... (e.g., 'Can you elaborate on that point?')"
            )
            submit_btn = gr.Button("Submit Question", variant="primary")

            with gr.Accordion("🔍 Inspect Retrieved Context Snippets", open=False):
                snippets_box = gr.Markdown(value="No context retrieved yet.")

    upload_btn.click(fn=handle_document_upload, inputs=[file_input], outputs=[status_box])
    
    submit_btn.click(
        fn=user_submit, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot, sources_box, snippets_box]
    )
    msg_input.submit(
        fn=user_submit, 
        inputs=[msg_input, chatbot], 
        outputs=[msg_input, chatbot, sources_box, snippets_box]
    )
    
    clear_chat_btn.click(fn=clear_conversation, outputs=[chatbot, sources_box, snippets_box])
    reset_db_btn.click(fn=reset_vector_database, outputs=[chatbot, sources_box, snippets_box, status_box])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), share=False)