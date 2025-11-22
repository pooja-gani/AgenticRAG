# Agentic RAG 🔍🤖  
A simple Agentic Retrieval-Augmented Generation (RAG) system that shows real-time reasoning steps using **Gemini**, **OpenAI**, and **Agno**.  
You can add URLs as knowledge sources, ask questions, and watch the agent think and answer with citations.

---

## 🚀 Features
- Add any URL to your knowledge base  
- Default article preloaded (MCP vs A2A Protocol)  
- Real-time reasoning + final answer  
- Vector search with OpenAI embeddings  
- Knowledge stored in LanceDB  
- Clear source citations  

---

## 🔑 Requirements
You need:
- **Google Gemini API Key**  
- **OpenAI API Key**  

---

## ▶️ Run the Project

```bash
git clone https://github.com/pooja-gani/AgenticRAG.git
cd AgenticRAG
pip install -r requirements.txt
streamlit run rag_reasoning_agent.py
