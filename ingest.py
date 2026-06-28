import os
import re
import pdfplumber
import chromadb
from typing import Any, List, Sequence

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.readers.base import BaseReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# ── CONFIGURATION ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
DOCS_DIR = os.path.join(BASE_DIR, "documents")
EMBED_MODEL_NAME = "google/embeddinggemma-300m"
COLLECTION_NAME = "policy_docs_collection"

os.makedirs(DB_DIR, exist_ok=True)


# ── TEXT CLEANING ──
# Strips image references, HTML tags, signature lines, and other noise
# from extracted text (PDF or Markdown).
def clean_extracted_text(text):
    """Remove images, HTML tags, and other noise from extracted text."""
    # Remove markdown image references: ![alt text](image_path)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove standalone image description lines: "Image: Handwritten signature..."
    text = re.sub(r'^Image:.*$', '', text, flags=re.MULTILINE)
    # Remove HTML tags like <sup>, </sup>, <sub>, </sub>
    text = re.sub(r'</?(?:sup|sub|br|hr|img)[^>]*>', '', text, flags=re.IGNORECASE)
    # Remove PDF figure/image captions: "Figure 1: ...", "[Image]"
    text = re.sub(r'^\s*Figure\s+\d+[.:].+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[Image\]', '', text, flags=re.IGNORECASE)
    # Collapse excess whitespace left behind by removals
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── CUSTOM PDF READER (preserves our pdfplumber extraction quality) ──
class PDFPlumberReader(BaseReader):
    """
    Custom LlamaIndex reader that uses pdfplumber for PDF text extraction.
    Preserves our existing extraction quality while integrating with the
    LlamaIndex document pipeline.
    """

    def load_data(self, file, extra_info=None, **kwargs) -> List[Document]:
        """Extract text from a PDF using pdfplumber, skipping very short / noisy pages."""
        pages_text = []
        source_name = os.path.basename(str(file))

        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    text = clean_extracted_text(text)
                    pages_text.append(text)

        full_text = "\n\n".join(pages_text)
        metadata = extra_info or {}
        metadata["source"] = source_name
        metadata["page_count"] = len(pages_text)

        print(f"    Extracted text from {len(pages_text)} pages.")
        return [Document(text=full_text, metadata=metadata)]


# ── CUSTOM TRANSFORM (preserves our domain-specific regex cleaning) ──
class CleanTextTransform(TransformComponent):
    """
    LlamaIndex TransformComponent that applies our domain-specific text
    cleaning (removes image refs, HTML tags, figure captions) to each node.
    """

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            node.set_content(clean_extracted_text(node.text))
        return nodes


# ── MAIN INGESTION ──
def ingest_documents():
    """
    LlamaIndex-based ingestion: extract, clean, chunk, embed, and index all
    documents using an IngestionPipeline with ChromaDB persistence.
    """

    # 1. Check for documents
    if not os.path.exists(DOCS_DIR):
        print(f"Documents folder not found at {DOCS_DIR}")
        return

    supported_files = []
    for fname in sorted(os.listdir(DOCS_DIR)):
        fpath = os.path.join(DOCS_DIR, fname)
        if os.path.isfile(fpath) and (fname.lower().endswith('.pdf') or fname.lower().endswith('.md')):
            supported_files.append(fpath)

    if not supported_files:
        print(f"No PDF or Markdown files found in {DOCS_DIR}")
        return

    print(f"\nFound {len(supported_files)} document(s) to ingest.")

    # 2. Load documents using our custom PDFPlumberReader + built-in MD reader
    print(f"\nLoading documents from: {DOCS_DIR}")
    file_extractor = {".pdf": PDFPlumberReader()}
    documents = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        file_extractor=file_extractor,
        filename_as_id=True,
    ).load_data()
    print(f"  Loaded {len(documents)} document(s).")

    # 3. Set up the embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL_NAME}")
    print("This model gets all available RAM since no other models are loaded.")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    # 4. Set up ChromaDB vector store
    print(f"Connecting to ChromaDB at: {DB_DIR}")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)

    # Delete old collection for clean re-indexing
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("Deleted old collection for clean re-indexing.")
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 5. Build and run the ingestion pipeline
    # Pipeline: Clean → Split → Embed → Store
    print(f"\nRunning ingestion pipeline...")
    pipeline = IngestionPipeline(
        transformations=[
            CleanTextTransform(),
            SentenceSplitter(chunk_size=1200, chunk_overlap=200),
            embed_model,
        ],
        vector_store=vector_store,
    )

    nodes = pipeline.run(documents=documents, show_progress=True)

    # 6. Report results
    print("\n" + "=" * 60)
    print(" INGESTION COMPLETE ".center(60, "="))
    print("=" * 60)
    print(f"  Documents processed: {len(supported_files)}")
    print(f"  Total nodes/chunks:  {len(nodes)}")
    print(f"  Vector DB:           {DB_DIR}")
    print(f"  Collection:          {COLLECTION_NAME}")
    print(f"  Embedding model:     {EMBED_MODEL_NAME}")
    print("=" * 60)

    return nodes


if __name__ == "__main__":
    ingest_documents()
