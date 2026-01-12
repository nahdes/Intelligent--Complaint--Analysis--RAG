
---

## 🧠 Core Technologies

- **Vector Search**: FAISS → Chroma DB
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: Google Flan-T5 (open-source)
- **Framework**: LangChain (modern + backward-compatible imports)
- **Frontend**: Streamlit
- **Evaluation**: Qualitative RAG assessment (Markdown report)

---

## 📂 Project Structure

.
├── vector_store/
│ ├── faiss_index.bin # Original FAISS index (Task 2)
│ └── metadata.json # Complaint text + metadata
│
├── notebooks/
│ └── vector_store/ # Persisted Chroma DB
│
├── RAG.py # FAISS→Chroma + RAG + Evaluation
├── streamlit_app.py # Streamlit chatbot interface
├── evaluation_report.md # Generated qualitative evaluation
├── requirements.txt
└── README.md

---

## 🔁 FAISS → Chroma Conversion

### Why Conversion Is Needed

- FAISS stores **only vectors**
- LangChain RAG pipelines expect **Document-based vector stores**
- Chroma provides:
  - Persistence
  - Metadata support
  - Native LangChain integration

### What the Converter Does

- Loads FAISS index
- Loads complaint metadata
- Reconstructs LangChain `Document` objects
- Re-embeds using the same model
- Persists everything into Chroma DB

---

## 🤖 RAG Pipeline Design

### Retrieval

- Semantic similarity search (`k = 3–10`)
- Embedding-based, not keyword-based

### Generation

- **Primary**: Flan-T5-Large (instruction-tuned)
- **Fallback**: Extractive summarization (no hallucination)

### Safety & Robustness

- Anti-hallucination prompt template
- Graceful degradation if LLM fails
- Answer length control
- Source attribution

---

## 📊 Evaluation Component (Required)

The system includes a **mandatory qualitative evaluation module**:

- 10 representative financial questions
- Generated answers + retrieved sources
- Exported to `evaluation_report.md`
- Manual quality scoring supported

This aligns with academic and industry RAG evaluation standards.

---

## 💬 Streamlit Chat Interface

### Features

- Natural language chat
- Adjustable number of retrieved sources
- Expandable source documents
- Persistent session history
- Exportable chat logs (JSON)
- Example question shortcuts

### Why Streamlit

- Fast prototyping
- Ideal for data & ML applications
- Easy deployment

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/creditrust-rag.git
cd creditrust-rag
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
3. Install Dependencies
pip install -r requirements.txt


Required packages include:

langchain

langchain-community

chromadb

faiss-cpu

sentence-transformers

transformers

streamlit

▶️ Running the System
Step 1: Build / Load Vector Store + Run Evaluation
python RAG.py


This will:

Convert FAISS → Chroma (if needed)

Initialize RAG

Run qualitative evaluation

Generate evaluation_report.md

Step 2: Launch Streamlit App
streamlit run streamlit_app.py


Open browser at:

http://localhost:8501

🧪 Example Questions

What issues do customers report with credit cards?

Why are customers unhappy with mortgages?

Are there complaints about unauthorized transactions?

What problems exist with debt collection?

What issues occur with account closures?

🔍 Key Design Strengths

✅ Open-source, no API cost

✅ Defensive imports (LangChain compatibility)

✅ Persistent vector storage

✅ Explainable AI (source attribution)

✅ Evaluation included (not optional)

✅ Production-style UI

🚀 Future Improvements

Metadata-based filtering (by product, date)

Quantitative evaluation (precision@k, recall@k)

GPU acceleration

Multi-document answer citations

Authentication & role-based access

Cloud deployment (Docker + VPS)

