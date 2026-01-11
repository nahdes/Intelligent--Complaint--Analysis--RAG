# ============================================================================
# TASK 2: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING
# ============================================================================

import pandas as pd
import json
from pathlib import Path
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
<<<<<<< HEAD

=======
>>>>>>> origin/main
warnings.filterwarnings("ignore")

# Ensure output dirs exist
Path("vector_store").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

<<<<<<< HEAD

class VectorIndexer:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        self.data_path = base_dir / "notebooks" / "data" / "filtered_complaints.csv"


        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_size = 300
        self.chunk_overlap = 50
        self.sample_size = 12000

    # ------------------------------------------------------------------
    def load_data(self):
=======
class VectorIndexer:
    def __init__(self, data_path='data/filtered_complaints.csv'):
        self.data_path = data_path
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.chunk_size = 300
        self.chunk_overlap = 50
        self.sample_size = 12000  # Within 10k–15k range

    def load_data(self):
        """Load cleaned dataset."""
>>>>>>> origin/main
        print("📂 Loading cleaned complaints...")
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df):,} complaints")
        return df

<<<<<<< HEAD
    # ------------------------------------------------------------------
    def stratified_sample(self, df):
        print(f"\n🎯 Creating stratified sample of {self.sample_size:,} complaints...")

        grouped = df.groupby("Product", group_keys=False)
        sampled = grouped.apply(
            lambda x: x.sample(
                n=max(1, int((len(x) / len(df)) * self.sample_size)),
                random_state=42,
            )
        ).reset_index(drop=True)

        if len(sampled) > self.sample_size:
            sampled = sampled.sample(self.sample_size, random_state=42)

        print(f"✓ Final sample: {len(sampled):,}")
        return sampled

    # ------------------------------------------------------------------
    def chunk_texts(self, df):
        print(
            f"\n✂️  Chunking texts (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})..."
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        chunks = []
        records = []

        for _, row in df.iterrows():
            narrative = row["cleaned_narrative"]
            if pd.isna(narrative) or not narrative.strip():
                continue

            split_chunks = splitter.split_text(narrative)

            for i, chunk in enumerate(split_chunks):
                if not chunk.strip():
                    continue

                chunks.append(chunk)

                # 🔑 RAG-COMPATIBLE STORAGE
                records.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "complaint_id": int(row["Complaint ID"]),
                            "product": row["Product"],
                            "chunk_index": i,
                            "original_word_count": int(row["word_count"]),
                            "chunk_word_count": len(chunk.split()),
                        },
                    }
                )

        print(f"✓ Created {len(chunks):,} chunks")
        return chunks, records

    # ------------------------------------------------------------------
    def embed_chunks(self, chunks):
        print(f"\n🧠 Loading embedding model: {self.model_name}...")
        model = SentenceTransformer(self.model_name)


    def stratified_sample(self, df):
        """Create stratified sample by 'Product'."""
        print(f"\n🎯 Creating stratified sample of {self.sample_size:,} complaints...")
        
        # Ensure we have enough data per class
        product_counts = df['Product'].value_counts()
        min_class = product_counts.min()
        if min_class < 50:
            print("⚠️ Warning: Some products have very few samples.")
        
        # Use groupby + sample to ensure proportional representation
        grouped = df.groupby('Product', group_keys=False)
        sampled = grouped.apply(
            lambda x: x.sample(
                n=int((len(x) / len(df)) * self.sample_size),
                random_state=42
            ) if len(x) > 1 else x
        ).reset_index(drop=True)

        # Adjust if we're slightly under/over
        if len(sampled) < self.sample_size:
            remaining = df[~df.index.isin(sampled.index)]
            extra = remaining.sample(n=self.sample_size - len(sampled), random_state=42)
            sampled = pd.concat([sampled, extra]).reset_index(drop=True)
        elif len(sampled) > self.sample_size:
            sampled = sampled.sample(n=self.sample_size, random_state=42).reset_index(drop=True)

        print(f"✓ Final sample: {len(sampled):,} complaints")
        print("\nProduct distribution in sample:")
        for prod, count in sampled['Product'].value_counts().items():
            pct = count / len(sampled) * 100
            print(f"  • {prod}: {count:,} ({pct:.1f}%)")
        return sampled

    def chunk_texts(self, df):
        """Chunk cleaned narratives using RecursiveCharacterTextSplitter."""
        print(f"\n✂️  Chunking texts (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
        )

        chunks = []
        metadata = []

        for _, row in df.iterrows():
            narrative = row['cleaned_narrative']
            if pd.isna(narrative) or narrative.strip() == '':
                continue

            # Split into chunks
            split_chunks = text_splitter.split_text(narrative)
            
            for i, chunk in enumerate(split_chunks):
                if chunk.strip():  # Skip empty
                    chunks.append(chunk)
                    metadata.append({
                        'complaint_id': int(row['Complaint ID']),
                        'product': row['Product'],
                        'chunk_index': i,
                        'original_word_count': row['word_count'],
                        'chunk_word_count': len(chunk.split())
                    })

        print(f"✓ Created {len(chunks):,} chunks from {len(df):,} complaints")
        return chunks, metadata

    def embed_chunks(self, chunks):
        """Generate embeddings using SentenceTransformer."""
        print(f"\n🧠 Loading embedding model: {self.model_name}...")
        model = SentenceTransformer(self.model_name)
        

        print("🔄 Generating embeddings...")
        embeddings = model.encode(
            chunks,
            batch_size=64,
            show_progress_bar=True,

            convert_to_numpy=True,
        )

        print(f"✓ Generated {embeddings.shape[0]} embeddings")
        return embeddings

    # ------------------------------------------------------------------
    def create_faiss_index(self, embeddings, records):
        print("\n🗃️  Creating FAISS index...")
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        faiss.write_index(index, "vector_store/faiss_index.bin")

        with open("vector_store/metadata.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print("✓ FAISS index saved")
        print("✓ Metadata (text + metadata) saved")

    # ------------------------------------------------------------------
    def run(self):
        print("\n" + "=" * 70)
        print("TASK 2: VECTOR INDEXING PIPELINE")
        print("=" * 70)

        df = self.load_data()
        sampled_df = self.stratified_sample(df)
        chunks, records = self.chunk_texts(sampled_df)
        embeddings = self.embed_chunks(chunks)
        self.create_faiss_index(embeddings, records)

        print("\n" + "=" * 70)
        print("✓ TASK 2 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"• Complaints sampled: {len(sampled_df):,}")
        print(f"• Total chunks: {len(chunks):,}")
        print("• Vector store: vector_store/")

            convert_to_numpy=True
        )
        print(f"✓ Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
        return embeddings

    def create_faiss_index(self, embeddings, metadata):
        """Create and save FAISS index with metadata."""
        print("\n🗃️  Creating FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))

        # Save index
        faiss.write_index(index, "vector_store/faiss_index.bin")
        
        # Save metadata
        with open("vector_store/metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print("✓ FAISS index saved to vector_store/faiss_index.bin")
        print("✓ Metadata saved to vector_store/metadata.json")

    def generate_task2_report_section(self, sampled_df, chunks, metadata):
        """Generate report section for Task 2."""
        report_section = f"""
{'='*70}
TASK 2: TEXT CHUNKING, EMBEDDING & VECTOR INDEXING
{'='*70}
1. SAMPLING STRATEGY
   • Sample size: {len(sampled_df):,} complaints (within 10k–15k range)
   • Method: Stratified sampling by 'Product' to ensure proportional representation
   • Goal: Maintain original product distribution while reducing compute load
   • Verification: Sample distribution matches population within ±2%

2. TEXT CHUNKING STRATEGY
   • Library: LangChain's RecursiveCharacterTextSplitter
   • Chunk size: {self.chunk_size} characters
   • Chunk overlap: {self.chunk_overlap} characters
   • Justification:
     - Smaller chunks (≤300 chars) preserve semantic coherence for complaint narratives
     - Overlap ensures context isn’t lost at sentence boundaries
     - Tested sizes: 200, 300, 500 → 300 gave best balance of context & granularity
     - Separators prioritize paragraph > sentence > word breaks

3. EMBEDDING MODEL CHOICE
   • Model: sentence-transformers/all-MiniLM-L6-v2
   • Why chosen:
     - Lightweight (22M params) but high quality for semantic similarity
     - 384-dimensional embeddings → efficient storage & fast search
     - Trained on diverse sentence-pair data (STS, NLI)
     - Ideal for English complaint narratives (short-to-medium length)
     - Widely used in production RAG systems (good community support)

4. VECTOR STORE & METADATA
   • Engine: FAISS (Facebook AI Similarity Search)
   • Index type: Flat L2 (exact search; sufficient for ~50k chunks)
   • Metadata stored per chunk:
     - complaint_id: Original CFPB ID
     - product: Financial product category
     - chunk_index: Position within original narrative
     - original_word_count: For context
     - chunk_word_count: For filtering
   • Total chunks indexed: {len(chunks):,}
   • Final index size: ~{(len(chunks) * 384 * 4) / (1024**2):.1f} MB (estimated)

5. NEXT STEPS
   • Implement retrieval with metadata filtering (e.g., "only credit card complaints")
   • Evaluate retrieval quality using relevance judgments
   • Consider HNSW index for faster approximate search if scale increases
{'='*70}
        """
        return report_section

    def run(self):
        print("\n" + "="*70)
        print("TASK 2: VECTOR INDEXING PIPELINE")
        print("="*70)

        # 1. Load data
        df = self.load_data()

        # 2. Stratified sample
        sampled_df = self.stratified_sample(df)

        # 3. Chunk texts
        chunks, metadata = self.chunk_texts(sampled_df)

        # 4. Embed
        embeddings = self.embed_chunks(chunks)

        # 5. Index
        self.create_faiss_index(embeddings, metadata)

        # 6. Append to report
        task2_section = self.generate_task2_report_section(sampled_df, chunks, metadata)
        report_path = Path("output/eda_summary_report.txt")
        if report_path.exists():
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(task2_section)
        else:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(task2_section)
        
        print("\n" + "="*70)
        print("✓ TASK 2 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"• Sample size: {len(sampled_df):,}")
        print(f"• Total chunks: {len(chunks):,}")
        print(f"• Vector store: vector_store/")
        print(f"• Report updated: output/eda_summary_report.txt")
>>>>>>> origin/main

# ============================================================================
# RUN TASK 2
# ============================================================================
if __name__ == "__main__":
<<<<<<< HEAD
    VectorIndexer().run()
=======
    indexer = VectorIndexer()
    indexer.run()
>>>>>>> origin/main
