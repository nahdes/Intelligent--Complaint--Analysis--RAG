# ============================================================================
# TASK 2: TEXT CHUNKING, EMBEDDING, AND VECTOR STORE INDEXING
# ============================================================================

import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings

warnings.filterwarnings("ignore")

# Ensure output dirs exist
Path("vector_store").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)


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
        print("📂 Loading cleaned complaints...")
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df):,} complaints")
        return df

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


# ============================================================================
# RUN TASK 2
# ============================================================================
if __name__ == "__main__":
    VectorIndexer().run()
