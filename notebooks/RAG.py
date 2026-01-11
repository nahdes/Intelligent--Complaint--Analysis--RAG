"""
Bridge Script: Convert FAISS Index to Chroma Vector Store
Fixes the incompatibility between Task 2 and Task 3
Updated for modern LangChain versions
INCLUDES REQUIRED EVALUATION COMPONENT
"""

import json
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Try different LangChain import patterns
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_IMPORT = "community"
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
        LANGCHAIN_IMPORT = "legacy"
    except ImportError:
        print("❌ LangChain not properly installed.")
        print("\n📦 Please install required packages:")
        print("   pip install langchain langchain-community chromadb sentence-transformers")
        exit(1)

# Try different Document import patterns
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

# Also need these for LLM
try:
    if LANGCHAIN_IMPORT == "community":
        from langchain_community.llms import HuggingFacePipeline
    else:
        from langchain.llms import HuggingFacePipeline
except ImportError:
    print("⚠️  HuggingFacePipeline import failed, will try alternative")


class FAISSToChromaConverter:
    """Convert FAISS index to Chroma vector store."""
    
    def __init__(self):
        self.faiss_index_path = "vector_store/faiss_index.bin"
        self.metadata_path = "vector_store/metadata.json"
        self.chroma_path = "notebooks/vector_store"
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
    def convert(self):
        """Perform the conversion."""
        print("\n" + "=" * 70)
        print("CONVERTING FAISS TO CHROMA")
        print("=" * 70)
        
        # 1. Load FAISS index
        print("\n📂 Loading FAISS index...")
        try:
            index = faiss.read_index(self.faiss_index_path)
            print(f"✓ Loaded FAISS index with {index.ntotal:,} vectors")
        except Exception as e:
            print(f"✗ Error loading FAISS index: {e}")
            return False
        
        # 2. Load metadata
        print("\n📂 Loading metadata...")
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_records = json.load(f)
            print(f"✓ Loaded {len(metadata_records):,} metadata records")
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
            return False
        
        # 3. Create LangChain documents
        print("\n📝 Creating LangChain documents...")
        documents = []
        for record in tqdm(metadata_records, desc="Processing"):
            doc = Document(
                page_content=record['text'],
                metadata=record['metadata']
            )
            documents.append(doc)
        
        print(f"✓ Created {len(documents):,} documents")
        
        # 4. Initialize embeddings
        print("\n🧠 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✓ Embedding model loaded")
        
        # 5. Create Chroma vector store
        print("\n🗄️  Creating Chroma vector store...")
        print("   This may take several minutes...")
        
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.chroma_path,
                collection_name="complaint_chunks"
            )
            
            # For newer versions of Chroma, persist() might not be needed
            try:
                vectorstore.persist()
            except AttributeError:
                pass  # Newer versions auto-persist
            
            print(f"✓ Chroma vector store created at: {self.chroma_path}")
            
            # Verify
            try:
                collection = vectorstore._collection
                count = collection.count()
                print(f"✓ Verified: {count:,} documents in Chroma")
            except:
                print(f"✓ Vector store created successfully")
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating Chroma store: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# RAG IMPLEMENTATION
# ============================================================================

class ImprovedRAG:
    """Improved RAG with proper prompt template as per guidelines."""
    
    def __init__(self, chroma_path="notebooks/vector_store"):
        self.chroma_path = chroma_path
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        
    def initialize(self):
        """Initialize the RAG system."""
        print("\n" + "=" * 70)
        print("INITIALIZING IMPROVED RAG SYSTEM")
        print("=" * 70)
        
        # Load embeddings (same as Task 2: all-MiniLM-L6-v2)
        print("\n🧠 Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load vector store
        print("\n📂 Loading Chroma vector store...")
        self.vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings,
            collection_name="complaint_chunks"
        )
        
        try:
            count = self.vectorstore._collection.count()
            print(f"✓ Loaded {count:,} documents")
        except:
            print(f"✓ Vector store loaded")
        
        # Load LLM
        print("\n🤖 Loading LLM (Flan-T5-Large)...")
        print("   This will download ~1GB on first run...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            print("✓ LLM loaded")
            
        except Exception as e:
            print(f"⚠️  Error loading LLM: {e}")
            print("   Continuing in retrieval-only mode...")
            self.llm = None
        
        return True
    
    def _create_extractive_summary(self, docs, question):
        """Create a summary without using LLM - extractive approach."""
        question_lower = question.lower()
        
        # Extract key information from docs
        issues = []
        products = set()
        
        for doc in docs[:5]:
            product = doc.metadata.get('product', 'Unknown')
            products.add(product)
            
            # Get relevant sentences
            text = doc.page_content
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            
            for sentence in sentences[:2]:
                if any(keyword in sentence.lower() for keyword in ['charge', 'fee', 'interest', 'issue', 'problem', 'complaint']):
                    if sentence not in issues:
                        issues.append(sentence)
        
        # Build summary
        if not issues:
            return "I don't have enough information in the retrieved complaints to answer this specific question."
        
        product_list = ', '.join(list(products)[:3])
        summary = f"Based on complaints about {product_list}, customers report: "
        summary += '; '.join(issues[:2]) + '.'
        
        if len(summary) > 400:
            summary = summary[:400].rsplit('.', 1)[0] + '.'
        
        return summary
    
    def query(self, question, k=5):
        """
        Query with proper prompt template as per guidelines.
        
        Args:
            question: User's question (string)
            k: Number of top documents to retrieve (default=5)
        
        Returns:
            dict with 'answer', 'sources', 'num_sources'
        """
        print(f"\n❓ Question: {question}")
        print(f"🔍 Retrieving top {k} relevant complaints...")
        
        # Retrieve documents using similarity search
        try:
            docs = self.vectorstore.similarity_search(question, k=k)
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}")
            return {
                'answer': 'Error retrieving documents.',
                'sources': [],
                'num_sources': 0
            }
        
        if not docs:
            return {
                'answer': 'No relevant information found.',
                'sources': [],
                'num_sources': 0
            }
        
        print(f"✓ Retrieved {len(docs)} documents")
        
        # If no LLM, use extractive summarization
        if self.llm is None:
            answer = self._create_extractive_summary(docs, question)
            return {
                'answer': answer,
                'sources': docs,
                'num_sources': len(docs)
            }
        
        # Build context from retrieved chunks
        context_parts = []
        for i, doc in enumerate(docs, 1):
            product = doc.metadata.get('product', 'Unknown Product')
            text = doc.page_content[:200]  # Limit length
            context_parts.append(f"[Complaint {i}] Product: {product}\n{text}")
        
        context = "\n\n".join(context_parts)
        
        # PROPER PROMPT TEMPLATE as per guidelines
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""
        
        print("🤖 Generating answer...")
        
        # Generate answer using LLM
        try:
            if hasattr(self.llm, 'invoke'):
                raw_answer = self.llm.invoke(prompt)
            elif hasattr(self.llm, 'pipeline'):
                result = self.llm.pipeline(prompt)
                raw_answer = result[0]['generated_text'] if isinstance(result, list) else result
            else:
                raw_answer = self.llm._call(prompt)
            
            # Extract answer text
            if isinstance(raw_answer, dict):
                answer = raw_answer.get('generated_text', str(raw_answer))
            else:
                answer = str(raw_answer)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Remove prompt echo if present
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            # If answer is too short or empty, provide fallback
            if not answer or len(answer) < 20:
                answer = self._create_extractive_summary(docs, question)
            
            # Limit answer length for readability
            if len(answer) > 500:
                sentences = answer.split('.')
                answer = '. '.join(sentences[:4]) + '.'
            
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            answer = self._create_extractive_summary(docs, question)
        
        return {
            'answer': answer,
            'sources': docs,
            'num_sources': len(docs)
        }
    
    def interactive_query(self):
        """Interactive question-answering."""
        print("\n" + "=" * 70)
        print("INTERACTIVE RAG QUERY")
        print("=" * 70)
        print("\nType 'quit' to exit\n")
        
        while True:
            question = input("\n❓ Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = self.query(question, k=5)
            
            print(f"\n💡 Answer:\n{result['answer']}")
            print(f"\n📚 Sources ({result['num_sources']} documents):")
            for i, doc in enumerate(result['sources'][:3], 1):
                product = doc.metadata.get('product', 'Unknown')
                preview = doc.page_content[:100]
                print(f"[{i}] {product}: {preview}...")
            print("-" * 60)


# ============================================================================
# EVALUATION COMPONENT (REQUIRED)
# ============================================================================

class RAGEvaluator:
    """Qualitative evaluation component as required by guidelines."""
    
    def __init__(self, rag_system):
        self.rag = rag_system
        
        # Representative questions for evaluation
        self.evaluation_questions = [
            "Why are customers complaining about credit reporting?",
            "What issues do customers have with mortgage accounts?",
            "What problems are reported with credit card fees?",
            "Why do customers complain about debt collection practices?",
            "What are the main issues with checking or savings accounts?",
            "What complaints exist about student loans?",
            "How do customers describe problems with account closures?",
            "What issues are raised about identity theft?",
            "Why do customers complain about unauthorized transactions?",
            "What problems do customers report with customer service?"
        ]
    
    def evaluate(self):
        """Run evaluation on all test questions."""
        print("\n" + "=" * 70)
        print("QUALITATIVE EVALUATION (REQUIRED BY GUIDELINES)")
        print("=" * 70)
        
        results = []
        
        for i, question in enumerate(self.evaluation_questions, 1):
            print(f"\n[{i}/{len(self.evaluation_questions)}] Evaluating...")
            
            # Run RAG pipeline
            result = self.rag.query(question, k=5)
            
            # Extract source previews
            source_previews = []
            for doc in result['sources'][:2]:
                product = doc.metadata.get('product', 'Unknown')
                preview = doc.page_content[:150] + "..."
                source_previews.append(f"{product}: {preview}")
            
            # Store results
            results.append({
                'Question': question,
                'Generated Answer': result['answer'],
                'Retrieved Sources': ' | '.join(source_previews),
                'Quality Score': None,  # To be filled manually
                'Comments/Analysis': None  # To be filled manually
            })
        
        return results
    
    def generate_markdown_report(self, results, output_file="evaluation_report.md"):
        """Generate evaluation report in Markdown format."""
        print(f"\n📝 Generating evaluation report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAG System Evaluation Report\n\n")
            f.write("## Overview\n\n")
            f.write("This report presents a qualitative evaluation of the RAG (Retrieval-Augmented Generation) system built for CrediTrust customer complaint analysis.\n\n")
            
            f.write("## System Configuration\n\n")
            f.write("- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2\n")
            f.write("- **Vector Store**: Chroma\n")
            f.write("- **LLM**: Flan-T5-Large\n")
            f.write("- **Retrieval**: Top-5 similarity search (k=5)\n\n")
            
            f.write("## Evaluation Results\n\n")
            f.write("| Question | Generated Answer | Retrieved Sources (Sample) | Quality Score (1-5) | Comments/Analysis |\n")
            f.write("|----------|------------------|---------------------------|---------------------|-------------------|\n")
            
            for r in results:
                # Truncate for table readability
                question = r['Question'][:50] + "..." if len(r['Question']) > 50 else r['Question']
                answer = r['Generated Answer'][:80] + "..." if len(r['Generated Answer']) > 80 else r['Generated Answer']
                sources = r['Retrieved Sources'][:100] + "..." if len(r['Retrieved Sources']) > 100 else r['Retrieved Sources']
                
                f.write(f"| {question} | {answer} | {sources} | _TBD_ | _TBD_ |\n")
            
            f.write("\n## Analysis\n\n")
            f.write("### What Worked Well\n\n")
            f.write("- _[To be filled after manual review]_\n")
            f.write("- _[Example: Retrieval successfully identified relevant complaints]_\n")
            f.write("- _[Example: Answers were coherent and contextually relevant]_\n\n")
            
            f.write("### What Could Be Improved\n\n")
            f.write("- _[To be filled after manual review]_\n")
            f.write("- _[Example: Some answers lacked specificity]_\n")
            f.write("- _[Example: Retrieved context sometimes contained irrelevant information]_\n\n")
            
            f.write("### Recommendations\n\n")
            f.write("- _[To be filled based on evaluation findings]_\n")
        
        print(f"✓ Report generated: {output_file}")
        print("⚠️  Please manually fill in Quality Scores and Comments/Analysis")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║              FAISS → Chroma Converter                              ║
║         + Complete RAG Implementation                              ║
║         + Qualitative Evaluation (Required)                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"\n🔧 Using LangChain import method: {LANGCHAIN_IMPORT}")
    
    # Check if Chroma already exists
    chroma_path = Path("notebooks/vector_store")
    
    if not chroma_path.exists() or not list(chroma_path.glob("*")):
        print("\n⚠️  Chroma vector store not found. Converting from FAISS...")
        
        converter = FAISSToChromaConverter()
        if not converter.convert():
            print("\n❌ Conversion failed. Check that FAISS index exists.")
            return
        
        print("\n✓ Conversion complete!")
    else:
        print("\n✓ Chroma vector store already exists. Skipping conversion.")
    
    # Initialize RAG
    print("\n" + "=" * 70)
    print("STARTING RAG SYSTEM")
    print("=" * 70)
    
    rag = ImprovedRAG()
    
    try:
        if not rag.initialize():
            print("\n❌ Failed to initialize RAG")
            return
    except Exception as e:
        print(f"\n❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run evaluation (REQUIRED)
    print("\n" + "=" * 70)
    print("RUNNING QUALITATIVE EVALUATION")
    print("=" * 70)
    
    evaluator = RAGEvaluator(rag)
    evaluation_results = evaluator.evaluate()
    evaluator.generate_markdown_report(evaluation_results)
    
    print("\n✓ Evaluation complete!")
    print("📄 Review 'evaluation_report.md' and fill in quality scores and analysis")
    
    # Interactive mode
    print("\n" + "=" * 70)
    user_choice = input("\nEnter interactive mode? (y/n): ").lower()
    
    if user_choice == 'y':
        rag.interactive_query()
    
    print("\n✓ Complete!")
    print("\n📋 DELIVERABLES CHECKLIST:")
    print("   ✓ Python module (.py file) with RAG pipeline logic")
    print("   ✓ Evaluation report (evaluation_report.md)")
    print("   ⚠️  TODO: Fill in Quality Scores and Analysis in report")


if __name__ == "__main__":
    main()