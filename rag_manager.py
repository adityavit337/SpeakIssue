import os
import asyncio
import threading
import chromadb
from ddgs import DDGS
from typing import Any, Dict, List

# ── Async Helper ──
# LlamaIndex's ReActAgent is built on the Workflows API (async-first).
# We run a dedicated event loop in a background thread so the sync
# Flask endpoints can call the async agent without issues.
_loop = asyncio.new_event_loop()
_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_thread.start()

def _run_async(coro, timeout=600):
    """Run an async coroutine from sync code using the background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result(timeout=timeout)

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.tools import FunctionTool
from llama_index.llms.groq import Groq
from dotenv import load_dotenv

# Load API Keys from .env
load_dotenv()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
EMBED_MODEL_NAME = "google/embeddinggemma-300m"
COLLECTION_NAME = "policy_docs_collection"


class RAGManager:
    def __init__(self):
        # ── Layer 1: LLM & Embedding Configuration ──
        print(f"Initializing LLM (Groq llama-3.3-70b-versatile)...")
        self.llm = Groq(model="llama-3.3-70b-versatile", request_timeout=60.0)

        print(f"Initializing Embedding Model ({EMBED_MODEL_NAME})...")
        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

        # Set global defaults so all LlamaIndex components use these
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # ── Layer 2: Vector Store & Index ──
        print("Connecting to local Vector Database...")
        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        chroma_collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Build the index from the existing persisted vector store
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )

        doc_count = chroma_collection.count()
        if doc_count == 0:
            print("WARNING: No documents in vector store. Run 'python ingest.py' first!")
        else:
            print(f"  Loaded {doc_count} chunks from vector store.")

        # ── Layer 3: Hybrid Retrieval (Vector + BM25) ──
        print("Setting up hybrid retrieval (Vector + BM25)...")

        # Vector retriever
        vector_retriever = self.index.as_retriever(similarity_top_k=20)

        # BM25 retriever — built from the nodes stored in the index's docstore
        # We retrieve all nodes from ChromaDB for BM25 indexing
        all_nodes = self._get_all_nodes(chroma_collection)

        if all_nodes:
            bm25_retriever = BM25Retriever(
                nodes=all_nodes,
                similarity_top_k=20,
            )
            # Fuse vector and BM25 results using reciprocal rank fusion
            self.hybrid_retriever = QueryFusionRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                similarity_top_k=10,
                num_queries=1,  # Don't generate sub-queries, just fuse results
                use_async=False,
            )
            print(f"  BM25 index built from {len(all_nodes)} nodes.")
        else:
            # Fall back to vector-only if no nodes available for BM25
            self.hybrid_retriever = vector_retriever
            print("  WARNING: BM25 unavailable, using vector-only retrieval.")

        # ── Layer 4: Reranker ──
        print("Loading Cross-Encoder for reranking (L-12)...")
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-12-v2",
            top_n=5,
        )

        # ── Layer 5: Query Engine (Retrieval + Reranking + Synthesis) ──
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.hybrid_retriever,
            node_postprocessors=[self.reranker],
            llm=self.llm,
            verbose=True,
        )

        # ── Layer 6: Agent with Tools ──
        print("Initializing ReAct Agent with tools...")
        self._setup_agent()

        print("Agentic RAG System ready.")

    def _get_all_nodes(self, chroma_collection) -> List:
        """
        Retrieve all document nodes from ChromaDB for BM25 indexing.
        Converts raw ChromaDB documents into LlamaIndex TextNode objects.
        """
        from llama_index.core.schema import TextNode

        results = chroma_collection.get(include=["documents", "metadatas"])
        nodes = []

        if results and results.get("documents"):
            for i, doc_text in enumerate(results["documents"]):
                if doc_text:
                    metadata = results["metadatas"][i] if results.get("metadatas") else {}
                    node_id = results["ids"][i] if results.get("ids") else f"node_{i}"
                    node = TextNode(
                        text=doc_text,
                        id_=node_id,
                        metadata=metadata or {},
                    )
                    nodes.append(node)

        return nodes

    def _setup_agent(self):
        """
        We've bypassed the heavy ReActAgent loop to minimize CPU workload.
        The 'agent' logic is now handled in a single-pass prompt inside query()
        combining both tools simultaneously without rethink loops.
        """
        pass

    def _search_policy_documents(self, search_query: str) -> str:
        """
        Search the local government policy database for rules, eligibility,
        procedures, and amounts. Use this tool FIRST whenever the user asks
        about PM-KISAN, PMFBY, Soil Health Card, or other agricultural policies.
        Formulate clear, specific keyword queries.
        """
        print(f"\n[TOOL EXECUTION] search_policy_documents: '{search_query}'")

        # Retrieve using hybrid retriever (vector + BM25 fusion)
        retrieved_nodes = self.hybrid_retriever.retrieve(search_query)

        if not retrieved_nodes:
            return "No documents found."

        # Rerank the retrieved nodes
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(query_str=search_query)
        reranked_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes, query_bundle=query_bundle
        )

        # Format the top results as text snippets
        snippets = []
        for node_with_score in reranked_nodes:
            text = node_with_score.node.text
            source = node_with_score.node.metadata.get("source", "unknown")
            snippets.append(f"[Source: {source}]\n{text}")

        result_text = "\n\n--- Document Snippet ---\n\n".join(snippets)
        print(f"  → Found {len(snippets)} relevant snippets.")
        return result_text

    @staticmethod
    def _search_web(search_query: str) -> str:
        """
        Search the internet for general information.
        Use this ONLY if search_policy_documents does not contain the answer.
        """
        print(f"\n[TOOL EXECUTION] search_web: '{search_query}'")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=3))
            snippets = []
            for r in results:
                snippets.append(f"Title: {r['title']}\n{r['body']}")
            return "\n\n".join(snippets) if snippets else "No web results found."
        except Exception as e:
            return f"Web search failed: {e}"

    # ── PUBLIC API (unchanged interface for app.py) ──

    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute an Agentic RAG query using LlamaIndex's ReAct agent.
        Returns the same dict shape as the old implementation for backward compatibility.
        """
        print(f"\n[{'='*40}]")
        print(f"USER QUERY: {question}")
        print(f"[{'='*40}]")

        try:
            # 1. Fetch Context via pure Python (No LLM workload)
            policy_context = self._search_policy_documents(question)
            
            # Fetch web context conditionally if policy is empty, or just parallelize.
            # For maximum context, we'll quickly grab top 2 web results.
            web_context = self._search_web(question)
            
            # 2. Build a single-pass prompt
            prompt = f"""You are "Kisan Mitra", a helpful agricultural policy assistant.
Use the provided Context to answer the User Query. 
Prioritize information from the Official Policy Documents. If they do not contain the answer, use the Web Search Context.
If neither contains the answer, say "I don't know based on the provided context".
Quote exact numbers and amounts when found.

--- Context from Official Policy Documents ---
{policy_context}

--- Context from Web Search ---
{web_context}

User Query: {question}
Answer:"""

            # 3. Single LLM generation pass (Zero rethink loops)
            async def _do_query():
                return await self.llm.acomplete(prompt)

            response = _run_async(_do_query())
            final_answer = str(response.text)
            print(f"\n--- Final Answer ---\n{final_answer.strip()}")

            return {
                "answer": final_answer.strip(),
                "source": "unified_single_pass",
                "context_used": ["policy_docs", "web_search"],
            }
        except Exception as e:
            print(f"Agent error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"I encountered an error while searching: {e}",
                "source": "error",
                "context_used": [],
            }


if __name__ == "__main__":
    manager = RAGManager()

    test_suite = [
        {
            "id": "Test 1",
            "target": "PM-KISAN",
            "query": "My father owns 1.5 hectares of cultivable land, but my mother is a retired government employee receiving a pension of ₹12,000 per month. Is our household eligible to receive the ₹6,000 annual financial benefit that is distributed in three equal installments?"
        },
        {
            "id": "Test 2",
            "target": "PMFBY",
            "query": "I am growing wheat this rabi season. What is the maximum premium percentage I have to pay under the Pradhan Mantri Fasal Bima Yojana?"
        },
        {
            "id": "Test 3",
            "target": "Soil Health Card",
            "query": "How frequently should my soil be tested under the Soil Health Card scheme, and what parameters are tested?"
        },
        {
            "id": "Test 4",
            "target": "PM-KMY",
            "query": "I am 35 years old and own 1 hectare of land. If I enroll in the Pradhan Mantri Kisan Maan-Dhan Yojana today, how much monthly pension will I receive when I turn 60?"
        },
        {
            "id": "Test 5",
            "target": "Per Drop More Crop",
            "query": "What are village level soil testing labs"
        }       
    ]

    results = []
    for test in test_suite:
        resp = manager.query(test["query"]) 
        results.append({
            "Test ID": test["id"],
            "Question": test["query"],
            "Answer": resp.get("answer", "")
        })

    print("\n" + "="*80)
    print(" RAG PIPELINE EVALUATION REPORT ".center(80, "="))
    for res in results:
        print(f"\n[{res['Test ID']}]")
        print(f"Q: {res['Question']}")
        print(f"A: {res['Answer']}")
        print("-" * 80)
