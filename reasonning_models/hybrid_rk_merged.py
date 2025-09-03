"""
Hybrid R–K System (merged):
 • Full controller + planning/critic loop
 • Real retrieval via ChromaDB or pgvector
 • OpenAI & Azure OpenAI adapters (function/tool calling)
 • TTL cache, structured facts, optional verifier pass

Usage (Chroma):
    pip install chromadb openai
    export OPENAI_API_KEY=...
    python hybrid_rk_merged.py

Usage (pgvector):
    pip install psycopg pgvector sqlalchemy openai
    # In Postgres (once):
    #   CREATE EXTENSION IF NOT EXISTS vector;
    #   CREATE TABLE IF NOT EXISTS rk_docs (
    #       id text primary key,
    #       text text not null,
    #       source text,
    #       date text,
    #       embedding vector(3072)
    #   );
    #   CREATE INDEX IF NOT EXISTS rk_docs_cos ON rk_docs USING ivfflat (embedding vector_cosine_ops);
    export OPENAI_API_KEY=...
    export PG_DSN="postgresql://user:pass@host:5432/db"
    RK_STACK=pgvector python hybrid_rk_merged.py

Swap LLM:
    # OpenAI (Responses API)
    llm = OpenAIResponsesAdapter(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
    # or Azure OpenAI
    llm = AzureOpenAIChatAdapter()
    # or for offline demo
    llm = MockLLM()
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Iterable
from time import time
import os
import json
import hashlib
from openai import OpenAI
import random

api_key="<>"
endpoint = "https://<>/openai/v1/"
deployment_name = "o4-mini"
os.environ["OPENAI_API_KEY"]=api_key
os.environ["OPENAI_DEPLOYMENT"]=deployment_name
os.environ["OPENAI_END_POINT"]=endpoint

# =========================
# ==== Data Contracts  ====
# =========================

@dataclass
class Fact:
    """Represents a single piece of factual information.

    Attributes:
        text (str): The text of the fact.
        source (str): The source of the fact (e.g., URL, document ID).
        date (str): The date the fact was recorded, in ISO format.
        conf (float): The confidence score of the fact (0.0 to 1.0).
    """
    text: str
    source: str           # e.g., URL or doc_id#line_range
    date: str             # ISO date string if available
    conf: float           # 0..1 confidence (similarity proxy)

@dataclass
class EvidencePacket:
    """A collection of facts and metadata.

    Attributes:
        facts (List[Fact]): A list of facts.
        notes (str): Any notes related to the evidence.
        coverage_ok (bool): Whether the evidence provides sufficient coverage.
    """
    facts: List[Fact] = field(default_factory=list)
    notes: str = ""
    coverage_ok: bool = True

@dataclass
class KRequest:
    """A request for knowledge from the Knowledge Agent.

    Attributes:
        task_id (str): A unique ID for the task.
        question (str): The question to be answered.
        scope_in (List[str]): A list of topics to include in the search.
        scope_out (List[str]): A list of topics to exclude from the search.
        need (List[str]): The types of information needed.
        constraints (Dict[str, Any]): Any constraints on the search.
    """
    task_id: str
    question: str
    scope_in: List[str] = field(default_factory=list)
    scope_out: List[str] = field(default_factory=list)
    need: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=lambda: {"recency_days": 365, "max_tokens": 400})

@dataclass
class KResponse:
    """A response from the Knowledge Agent.

    Attributes:
        facts (List[Fact]): A list of facts found by the agent.
        notes (str): Any notes from the agent.
        coverage_ok (bool): Whether the agent found sufficient information.
    """
    facts: List[Fact]
    notes: str
    coverage_ok: bool

@dataclass
class ClaimCheck:
    """Represents the result of a fact-check on a claim.

    Attributes:
        claim (str): The claim that was checked.
        supported_by (List[str]): A list of sources that support the claim.
        status (str): The status of the claim ("supported", "contradicted", or "uncertain").
        suggestion (str): A suggestion for improving the claim.
    """
    claim: str
    supported_by: List[str]
    status: str              # "supported" | "contradicted" | "uncertain"
    suggestion: str = ""

# ==========================================
# ==== Tiny TTL Cache for K lookups      ===
# ==========================================

class TTLCache:
    """A simple time-to-live (TTL) cache.

    This cache stores items for a specified amount of time.
    """
    def __init__(self, ttl_seconds: int = 1800, max_items: int = 512):
        """Initializes the TTLCache.

        Args:
            ttl_seconds (int): The time-to-live for cached items, in seconds.
            max_items (int): The maximum number of items to store in the cache.
        """
        self.ttl = ttl_seconds
        self.max = max_items
        self._store: Dict[str, Tuple[float, Any]] = {}

    def _evict_if_needed(self):
        """Evicts the oldest item if the cache is full."""
        if len(self._store) <= self.max:
            return
        oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
        self._store.pop(oldest_key, None)

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache.

        Args:
            key (str): The key of the item to retrieve.

        Returns:
            Optional[Any]: The cached value, or None if the item is not in the
                           cache or has expired.
        """
        rec = self._store.get(key)
        if not rec:
            return None
        ts, val = rec
        if time() - ts > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any):
        """Adds an item to the cache.

        Args:
            key (str): The key of the item to add.
            value (Any): The value of the item to add.
        """
        self._store[key] = (time(), value)
        self._evict_if_needed()

    @staticmethod
    def key_for(obj: Any) -> str:
        """Creates a cache key for an object.

        Args:
            obj (Any): The object to create a key for.

        Returns:
            str: A SHA256 hash of the object.
        """
        return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()

# =================================
# ==== LLM Client Adapters      ===
# =================================

class LLMClient:
    """An abstract base class for LLM clients."""
    def complete(self, system: str, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generates a completion from the LLM.

        Args:
            system (str): The system message to send to the LLM.
            messages (List[Dict[str, str]]): A list of messages to send to the LLM.
            tools (Optional[List[Dict]]): A list of tools that the LLM can use.

        Returns:
            Dict[str, Any]: The response from the LLM.
        """
        raise NotImplementedError

class MockLLM(LLMClient):
    """A mock LLM client for testing."""
    def complete(self, system: str, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generates a mock completion.

        Args:
            system (str): The system message.
            messages (List[Dict[str, str]]): The list of messages.
            tools (Optional[List[Dict]]): The list of tools.

        Returns:
            Dict[str, Any]: A mock response.
        """
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "PLAN_SUBQUESTIONS" in user:
            content = {
                "subquestions": [
                    {"q": "List key differences relevant to the user question.", "needs_facts": True},
                    {"q": "Synthesize the tradeoffs succinctly.", "needs_facts": False},
                ],
                "assumptions": [],
            }
            return {"function_call": {"name": "return_plan", "arguments": json.dumps(content)}}
        if "COMPOSE_FINAL" in user:
            try:
                ctx = json.loads(user.split("CTX_JSON=")[-1])
            except Exception:
                ctx = {"packets": [], "question": ""}
            lines = []
            for pkt in ctx.get("packets", []):
                for f in pkt.get("facts", []):
                    lines.append(f"- {f['text']} (source: {f['source']})")
            final = "Answer:\n" + "\n".join(lines) if lines else "Answer: (no evidence found)"
            return {"content": final}
        if "FACT_CHECK" in user:
            try:
                draft = user.split("DRAFT_START>>")[1].split("<<DRAFT_END")[0].strip()
            except Exception:
                draft = user
            items = []
            for line in draft.split("\n"):
                if not line.strip():
                    continue
                items.append({
                    "claim": line.strip(),
                    "supported_by": ["kb://demo#1"],
                    "status": "supported",
                    "suggestion": ""
                })
            return {"function_call": {"name": "return_checks", "arguments": json.dumps(items)}}
        return {"content": "OK"}

class OpenAIResponsesAdapter(LLMClient):
    """
    An adapter for the OpenAI Responses API.

    This adapter reads the following environment variables:
      - OPENAI_API_KEY
      - OPENAI_DEPLOYMENT
      - OPENAI_END_POINT
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        tool_choice: Optional[str] = None,
        max_retries: int = 5,
        base_path: str = "/openai/v1/",
        fallback_to_mock: bool = False,
    ):
        """Initializes the OpenAIResponsesAdapter.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            endpoint (Optional[str]): The OpenAI API endpoint.
            deployment (Optional[str]): The OpenAI deployment name.
            tool_choice (Optional[str]): The tool choice to use.
            max_retries (int): The maximum number of retries for API calls.
            base_path (str): The base path for the API endpoint.
            fallback_to_mock (bool): Whether to fall back to the mock LLM on failure.
        """
        from openai import OpenAI

        env_endpoint = os.getenv("OPENAI_END_POINT") or os.getenv("OPENAI_ENDPOINT")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.deployment = os.getenv("OPENAI_DEPLOYMENT")
        raw_endpoint = os.getenv("OPENAI_END_POINT")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set and api_key not provided.")
        if not self.deployment:
            raise ValueError("OPENAI_DEPLOYMENT (deployment name) not set and deployment not provided.")
        if not raw_endpoint:
            raise ValueError("OPENAI_END_POINT/OPENAI_ENDPOINT not set and endpoint not provided.")

        if raw_endpoint.rstrip("/").endswith("/openai/v1"):
            base_url = raw_endpoint.rstrip("/") + "/"
        else:
            base_url = raw_endpoint.rstrip("/") + base_path

        self.client = OpenAI(api_key=self.api_key, base_url=os.getenv("OPENAI_END_POINT"))
        self.tool_choice = tool_choice
        self.max_retries = max_retries
        self.fallback_to_mock = fallback_to_mock
        self._mock = MockLLM() if fallback_to_mock else None

    def _sleep_for_retry(self, exc: Exception, attempt: int, base_backoff: float = 0.75):
        """Sleeps for a specified amount of time before retrying an API call.

        Args:
            exc (Exception): The exception that was raised.
            attempt (int): The current retry attempt number.
            base_backoff (float): The base backoff time in seconds.
        """
        retry_after = None
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                headers = getattr(resp, "headers", {}) or {}
                retry_after = headers.get("Retry-After") or headers.get("retry-after")
            except Exception:
                pass
        if retry_after:
            try:
                time.sleep(float(retry_after))
                return
            except Exception:
                pass
        time.sleep(base_backoff * (2 ** (attempt - 1)) + random.uniform(0, 0.25))

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generates a completion from the OpenAI Responses API.

        Args:
            system (str): The system message.
            messages (List[Dict[str, str]]): The list of messages.
            tools (Optional[List[Dict]]): The list of tools.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        input_msgs = [{"role": "system", "content": system}] + messages
        kwargs: Dict[str, Any] = {
            "model": self.deployment,
            "input": input_msgs
        }
        if tools:
            kwargs["tools"] = tools
            if self.tool_choice:
                kwargs["tool_choice"] = self.tool_choice

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(**kwargs)
                out: Dict[str, Any] = {}
                if resp.output:
                    item = resp.output[0]
                    if getattr(item, "type", None) == "message":
                        if item.content and len(item.content) > 0 and getattr(item.content[0], "type", "") == "output_text":
                            out["content"] = item.content[0].text
                        if item.tool_calls:
                            tc = item.tool_calls[0]
                            out["function_call"] = {"name": tc.function.name, "arguments": tc.function.arguments}
                return out

            except Exception as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                code = None
                try:
                    body = getattr(e, "response", None)
                    if body is not None:
                        try:
                            j = body.json()
                        except Exception:
                            j = {}
                        code = (j or {}).get("error", {}).get("code")
                except Exception:
                    pass

                if code == "insufficient_quota":
                    break
                if status == 429 or (status and 500 <= int(status) < 600):
                    if attempt < self.max_retries:
                        self._sleep_for_retry(e, attempt)
                        continue
                break

        if self.fallback_to_mock and self._mock:
            return self._mock.complete(system, messages, tools)

        raise last_exc if last_exc else RuntimeError("OpenAIResponsesAdapter: unknown failure")

class OpenAIChatAdapter(LLMClient):
    """An adapter for the OpenAI Chat Completions API."""
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initializes the OpenAIChatAdapter.

        Args:
            model (str): The name of the model to use.
        """
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generates a completion from the OpenAI Chat Completions API.

        Args:
            system (str): The system message.
            messages (List[Dict[str, str]]): The list of messages.
            tools (Optional[List[Dict]]): The list of tools.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        chat_msgs = [{"role": "system", "content": system}] + messages
        kwargs: Dict[str, Any] = {"model": self.model, "messages": chat_msgs}
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        out: Dict[str, Any] = {}
        if choice.message.content:
            out["content"] = choice.message.content
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            out["function_call"] = {"name": tc.function.name, "arguments": tc.function.arguments}
        return out
    
api_key="<>"
endpoint = "https://<>.openai.azure.com/openai/v1/"
deployment_name = "o4-mini"

os.environ["OPENAI_API_KEY"]=api_key
os.environ["OPENAI_DEPLOYMENT"]=deployment_name
os.environ["OPENAI_END_POINT"]=endpoint

class AzureOpenAIChatAdapter(LLMClient):
    """An adapter for the Azure OpenAI Chat Completions API."""
    def __init__(self, api_version: Optional[str] = None, deployment_env_var: str = "AZURE_OPENAI_DEPLOYMENT"):
        """Initializes the AzureOpenAIChatAdapter.

        Args:
            api_version (Optional[str]): The Azure API version to use.
            deployment_env_var (str): The name of the environment variable that
                                      contains the deployment name.
        """
        from openai import AzureOpenAI
        endpoint = os.environ.get("OPENAI_END_POINT")
        if not endpoint:
            raise ValueError("OPENAI_END_POINT not set")
        self.client = OpenAI(
           api_key=os.getenv("OPENAI_API_KEY"),
           base_url=os.getenv("OPENAI_END_POINT")
        )
        self.deployment = os.getenv("OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise ValueError(f"{deployment_env_var} not set")

    def complete(self, system: str, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generates a completion from the Azure OpenAI Chat Completions API.

        Args:
            system (str): The system message.
            messages (List[Dict[str, str]]): The list of messages.
            tools (Optional[List[Dict]]): The list of tools.

        Returns:
            Dict[str, Any]: The response from the API.
        """
        chat_msgs = [{"role": "system", "content": system}] + messages
        kwargs: Dict[str, Any] = {"model": self.deployment, "messages": chat_msgs}
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        out: Dict[str, Any] = {}
        if choice.message.content:
            out["content"] = choice.message.content
        if choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            out["function_call"] = {"name": tc.function.name, "arguments": tc.function.arguments}
        return out

# ==========================================
# ==== Embeddings & Retrievers           ===
# ==========================================
from openai import OpenAI
os.environ["OPENAI_EMBED_END_POINT"]=  "https://<>-azure-openai-001.openai.azure.com"
os.environ["OPENAI_EMBED_DEPLOYMENT"]="text-embedding-3-large"
os.environ["OPENAI_EMBED_API_KEY"]="<>"

class Embedder:
    """An abstract base class for embedders."""
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        raise NotImplementedError

class OpenAIEmbedder(Embedder):
    """An embedder that uses the OpenAI API."""
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        base_path: str = "/openai/v1/",
        normalize_embeddings: bool = False,
        max_retries: int = 4,
        backoff_base: float = 0.6,
    ):
        """Initializes the OpenAIEmbedder.

        Args:
            api_key (Optional[str]): The OpenAI API key.
            endpoint (Optional[str]): The OpenAI API endpoint.
            deployment (Optional[str]): The OpenAI deployment name.
            base_path (str): The base path for the API endpoint.
            normalize_embeddings (bool): Whether to normalize the embeddings.
            max_retries (int): The maximum number of retries for API calls.
            backoff_base (float): The base backoff time in seconds.
        """
        from openai import OpenAI

        env_endpoint = os.environ["OPENAI_EMBED_END_POINT"]
        env_deploy = os.environ["OPENAI_EMBED_DEPLOYMENT"]
        self.api_key = os.environ["OPENAI_EMBED_API_KEY"]
        self.deployment = env_deploy
        raw_endpoint = env_endpoint

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set and api_key not provided for OpenAIEmbedder.")
        if not self.deployment:
            raise ValueError("Embedding deployment name not provided. Set OPENAI_EMBED_DEPLOYMENT or pass deployment=.")
        if not raw_endpoint:
            raise ValueError("Embedding endpoint not provided. Set OPENAI_EMBED_END_POINT or pass endpoint=.")

        base_url = raw_endpoint.rstrip("/") + "/openai/v1/"
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.normalize_embeddings = normalize_embeddings
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def _retry_sleep(self, attempt: int):
        """Sleeps for a specified amount of time before retrying an API call.

        Args:
            attempt (int): The current retry attempt number.
        """
        import math
        time.sleep(self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.2))

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using the OpenAI API.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(
                    input=texts,
                    model=os.environ["OPENAI_EMBED_DEPLOYMENT"]
                )
                vecs = [d.embedding for d in resp.data]
                if self.normalize_embeddings:
                    vecs = self._l2_normalize(vecs)
                return vecs
            except Exception as e:
                last_exc = e
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                code = None
                try:
                    j = getattr(e, "response", None)
                    j = j.json() if j is not None else {}
                    code = (j or {}).get("error", {}).get("code")
                except Exception:
                    pass
                if code == "insufficient_quota":
                    break
                if status == 429 or (status and 500 <= int(status) < 600):
                    if attempt < self.max_retries:
                        self._retry_sleep(attempt)
                        continue
                break
        raise last_exc if last_exc else RuntimeError("OpenAIEmbedder: embedding request failed")

    @staticmethod
    def _l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
        """Normalizes a list of vectors using L2 normalization.

        Args:
            vecs (List[List[float]]): A list of vectors to normalize.

        Returns:
            List[List[float]]: A list of normalized vectors.
        """
        out = []
        for v in vecs:
            s = sum(x*x for x in v) or 1.0
            n = s ** 0.5
            out.append([x / n for x in v])
        return out

class LocalMiniLMEmbedder(Embedder):
    """An embedder that uses a local Sentence Transformers model."""
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initializes the LocalMiniLMEmbedder.

        Args:
            model (str): The name of the Sentence Transformers model to use.
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts using a local model.

        Args:
            texts (List[str]): A list of texts to embed.

        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class ChromaRetriever:
    """A retriever that uses a ChromaDB vector store."""
    def __init__(self, collection: str, persist_dir: str = ".chroma", embedder: Optional[Embedder] = None):
        """Initializes the ChromaRetriever.

        Args:
            collection (str): The name of the ChromaDB collection to use.
            persist_dir (str): The directory to persist the ChromaDB data to.
            embedder (Optional[Embedder]): The embedder to use for queries.
        """
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        self.col = self._get_or_create(collection)
        self.embedder = embedder

    def _get_or_create(self, name: str):
        """Gets or creates a ChromaDB collection.

        Args:
            name (str): The name of the collection.

        Returns:
            A ChromaDB collection.
        """
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    def add(self, ids: List[str], texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Adds documents to the ChromaDB collection.

        Args:
            ids (List[str]): A list of document IDs.
            texts (List[str]): A list of document texts.
            metadatas (Optional[List[Dict[str, Any]]]): A list of document metadatas.
        """
        if not self.embedder:
            raise ValueError("embedder required for add()")
        embs = self.embedder.embed(texts)
        self.col.add(ids=ids, documents=texts, metadatas=metadatas or [{} for _ in texts], embeddings=embs)

    def count(self) -> int:
        """Returns the number of documents in the collection.

        Returns:
            int: The number of documents.
        """
        try:
            return self.col.count()
        except Exception:
            return 0

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Searches the ChromaDB collection for similar documents.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        if self.embedder is None:
            result = self.col.query(query_texts=[query], n_results=k)
        else:
            q_emb = self.embedder.embed([query])[0]
            result = self.col.query(query_embeddings=[q_emb], n_results=k)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] or []
        out = []
        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = distances[i] if i < len(distances) and distances[i] is not None else 0.3
            conf = float(max(0.0, 1.0 - dist))
            out.append({
                "text": text,
                "source": meta.get("source", f"chroma://{ids[i]}"),
                "date": meta.get("date", ""),
                "conf": conf,
            })
        return out

class PGVectorRetriever:
    """A retriever that uses a pgvector PostgreSQL database."""
    def __init__(self, dsn: str, dim: int = 3072):
        """Initializes the PGVectorRetriever.

        Args:
            dsn (str): The data source name for the PostgreSQL database.
            dim (int): The dimension of the embeddings.
        """
        import psycopg
        self.pg = psycopg.connect(dsn)
        self.dim = dim

    def add(self, rows: Iterable[Tuple[str, str, str, str, List[float]]]):
        """Adds documents to the pgvector database.

        Args:
            rows (Iterable[Tuple[str, str, str, str, List[float]]]): An iterable of
                tuples, where each tuple contains the ID, text, source, date,
                and embedding of a document.
        """
        with self.pg.cursor() as cur:
            for rid, text, source, date, emb in rows:
                cur.execute(
                    """
                    INSERT INTO rk_docs (id, text, source, date, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, source = EXCLUDED.source, date = EXCLUDED.date, embedding = EXCLUDED.embedding
                    """,
                    (rid, text, source, date, emb)
                )
        self.pg.commit()

    def search(self, query_embedding: List[float], k: int = 6) -> List[Dict[str, Any]]:
        """Searches the pgvector database for similar documents.

        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int): The number of results to return.

        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        with self.pg.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, source, COALESCE(date, ''), (embedding <=> %s) AS cos_dist
                FROM rk_docs
                ORDER BY cos_dist ASC
                LIMIT %s
                """,
                (query_embedding, k)
            )
            rows = cur.fetchall()
        out = []
        for rid, text, source, date, cos_dist in rows:
            sim = 1.0 - float(cos_dist)
            out.append({
                "text": text,
                "source": source or f"pgvec://{rid}",
                "date": date,
                "conf": max(0.0, min(1.0, sim))
            })
        return out

# ================================
# ==== Knowledge Agent (K)     ===
# ================================

class KAgent:
    """The Knowledge Agent, responsible for retrieving information."""
    def __init__(self, retriever, embedder: Optional[Embedder] = None):
        """Initializes the KAgent.

        Args:
            retriever: The retriever to use for retrieving information.
            embedder (Optional[Embedder]): The embedder to use for queries.
        """
        self.retriever = retriever
        self.embedder = embedder

    def lookup(self, kreq: KRequest) -> KResponse:
        """Looks up information based on a KRequest.

        Args:
            kreq (KRequest): The request for knowledge.

        Returns:
            KResponse: The response from the knowledge agent.
        """
        q = kreq.question.strip()
        if isinstance(self.retriever, PGVectorRetriever):
            if not self.embedder:
                raise ValueError("PGVectorRetriever needs an embedder for queries")
            [q_emb] = self.embedder.embed([q])
            hits = self.retriever.search(q_emb, k=7)
        else:
            hits = self.retriever.search(q, k=7)
        facts = [Fact(text=h["text"], source=h.get("source", ""), date=h.get("date", ""), conf=float(h.get("conf", 0.7))) for h in hits]
        return KResponse(facts=facts, notes=f"retrieval:{self.retriever.__class__.__name__}", coverage_ok=len(facts) > 0)

    def fact_check(self, draft: str, support_threshold: float = 0.65) -> List[ClaimCheck]:
        """Fact-checks a draft of text.

        Args:
            draft (str): The draft to fact-check.
            support_threshold (float): The minimum confidence score for a
                                       fact to be considered supportive.

        Returns:
            List[ClaimCheck]: A list of fact-check results.
        """
        checks: List[ClaimCheck] = []
        lines = [ln.strip() for ln in draft.split("\n") if ln.strip()]
        for ln in lines:
            if isinstance(self.retriever, PGVectorRetriever):
                if not self.embedder:
                    checks.append(ClaimCheck(claim=ln, supported_by=[], status="uncertain"))
                    continue
                [emb] = self.embedder.embed([ln])
                hits = self.retriever.search(emb, k=3)
            else:
                hits = self.retriever.search(ln, k=3)
            supported = [h for h in hits if float(h.get("conf", 0.0)) >= support_threshold]
            if supported:
                checks.append(ClaimCheck(claim=ln, supported_by=[h.get("source", "") for h in supported], status="supported"))
            else:
                checks.append(ClaimCheck(claim=ln, supported_by=[], status="uncertain"))
        return checks

# ================================
# ==== Reasoning Agent (R)     ===
# ================================

class RAgent:
    """The Reasoning Agent, responsible for planning and composing answers."""
    def __init__(self, llm: LLMClient):
        """Initializes the RAgent.

        Args:
            llm (LLMClient): The LLM client to use.
        """
        self.llm = llm

    def plan(self, user_query: str) -> Dict[str, Any]:
        """Creates a plan for answering a user's query.

        Args:
            user_query (str): The user's query.

        Returns:
            Dict[str, Any]: A dictionary containing the plan.
        """
        system = "You are a planner. Output via function call if possible."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"PLAN_SUBQUESTIONS\nQuestion: {user_query}\nReturn minimal sub-questions; mark each needs_facts True/False."}
        ]
        resp = self.llm.complete(system, messages, tools=None)
        if "function_call" in resp:
            try:
                args = json.loads(resp["function_call"]["arguments"])
                if isinstance(args, dict) and "subquestions" in args:
                    return args
            except Exception:
                pass
        return {"subquestions": [{"q": user_query, "needs_facts": True}], "assumptions": []}

    def decide_need_knowledge(self, subq: Dict[str, Any]) -> bool:
        """Decides whether a sub-question requires knowledge retrieval.

        Args:
            subq (Dict[str, Any]): The sub-question.

        Returns:
            bool: True if the sub-question requires knowledge retrieval,
                  False otherwise.
        """
        return bool(subq.get("needs_facts", True))

    def compose(self, user_query: str, packets: List[EvidencePacket]) -> str:
        """Composes an answer to a user's query based on evidence.

        Args:
            user_query (str): The user's query.
            packets (List[EvidencePacket]): A list of evidence packets.

        Returns:
            str: The composed answer.
        """
        system = "You are a synthesizer. Use only supplied evidence and include inline citations in parentheses."
        ctx = {
            "question": user_query,
            "packets": [
                {
                    "facts": [asdict(f) for f in p.facts],
                    "notes": p.notes,
                    "coverage_ok": p.coverage_ok,
                }
                for p in packets
            ],
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"COMPOSE_FINAL\nUse supplied evidence to answer succinctly with citations.\nCTX_JSON={json.dumps(ctx, ensure_ascii=False)}"},
        ]
        resp = self.llm.complete(system, messages, tools=None)
        if resp.get("content"):
            return resp["content"].strip()
        lines = []
        for p in packets:
            for f in p.facts:
                lines.append(f"- {f.text} (source: {f.source})")
        return "Answer:\n" + ("\n".join(lines) if lines else "(no evidence)")

    def request_fact_check(self, draft: str) -> List[ClaimCheck]:
        """Requests a fact-check for a draft of text.

        Args:
            draft (str): The draft to fact-check.

        Returns:
            List[ClaimCheck]: A list of fact-check results.
        """
        system = "You are a fact-check coordinator. Return a JSON list via function call."
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"FACT_CHECK\nDRAFT_START>>\n{draft}\n<<DRAFT_END"},
        ]
        resp = self.llm.complete(system, messages, tools=None)
        if "function_call" in resp:
            try:
                return [ClaimCheck(**c) for c in json.loads(resp["function_call"]["arguments"])]
            except Exception:
                return []
        return []

# ======================================
# ==== Orchestrator / Controller     ===
# ======================================

class HybridController:
    """The main controller for the Hybrid R–K System.

    This class orchestrates the reasoning and knowledge retrieval process.
    """
    def __init__(self, r: RAgent, k: KAgent, cache: Optional[TTLCache] = None,
                 max_k_calls: int = 6, enable_critic_pass: bool = True,
                 use_k_based_checker: bool = True):
        """Initializes the HybridController.

        Args:
            r (RAgent): The Reasoning Agent.
            k (KAgent): The Knowledge Agent.
            cache (Optional[TTLCache]): The cache to use for knowledge lookups.
            max_k_calls (int): The maximum number of calls to the Knowledge Agent.
            enable_critic_pass (bool): Whether to enable the critic pass.
            use_k_based_checker (bool): Whether to use the Knowledge Agent for
                                        fact-checking.
        """
        self.r = r
        self.k = k
        self.cache = cache or TTLCache(ttl_seconds=1800, max_items=512)
        self.max_k_calls = max_k_calls
        self.enable_critic_pass = enable_critic_pass
        self.use_k_based_checker = use_k_based_checker

    def _cache_get(self, kreq: KRequest) -> Optional[KResponse]:
        """Retrieves a KResponse from the cache.

        Args:
            kreq (KRequest): The KRequest to use as the cache key.

        Returns:
            Optional[KResponse]: The cached KResponse, or None if not found.
        """
        key = TTLCache.key_for(asdict(kreq))
        return self.cache.get(key)

    def _cache_set(self, kreq: KRequest, kres: KResponse):
        """Adds a KResponse to the cache.

        Args:
            kreq (KRequest): The KRequest to use as the cache key.
            kres (KResponse): The KResponse to cache.
        """
        key = TTLCache.key_for(asdict(kreq))
        self.cache.set(key, kres)

    def solve(self, user_query: str) -> Dict[str, Any]:
        """Solves a user's query.

        Args:
            user_query (str): The user's query.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the process.
        """
        plan = self.r.plan(user_query)
        subqs: List[Dict[str, Any]] = plan.get("subquestions", [])
        packets: List[EvidencePacket] = []
        k_calls = 0

        for i, sq in enumerate(subqs):
            if k_calls >= self.max_k_calls:
                break
            if not self.r.decide_need_knowledge(sq):
                packets.append(EvidencePacket(facts=[], notes=f"R handled subq[{i}] without K", coverage_ok=True))
                continue
            kreq = KRequest(
                task_id=f"sq-{i}",
                question=sq["q"],
                scope_in=["on-topic"],
                scope_out=["opinion", "speculation"],
                need=["citations", "numbers", "definitions"],
            )
            cached = self._cache_get(kreq)
            if cached:
                kres = cached
            else:
                kres = self.k.lookup(kreq)
                self._cache_set(kreq, kres)
            packets.append(EvidencePacket(facts=kres.facts, notes=kres.notes, coverage_ok=kres.coverage_ok))
            k_calls += 1

        draft = self.r.compose(user_query, packets)

        checks: List[ClaimCheck] = []
        if self.enable_critic_pass:
            checks = self.r.request_fact_check(draft)
            if self.use_k_based_checker and not checks:
                checks = self.k.fact_check(draft)

        final = self._revise_with_checks(draft, checks)

        return {
            "plan": plan,
            "k_calls": k_calls,
            "draft": draft,
            "final": final,
            "checks": [asdict(c) for c in checks],
        }

    @staticmethod
    def _revise_with_checks(draft: str, checks: List[ClaimCheck]) -> str:
        """Revises a draft based on fact-check results.

        Args:
            draft (str): The draft to revise.
            checks (List[ClaimCheck]): The fact-check results.

        Returns:
            str: The revised draft.
        """
        if not checks:
            return draft
        lines = [draft.strip(), "", "Groundedness audit:"]
        for c in checks:
            tag = {"supported": "✔", "contradicted": "✘", "uncertain": "?"}.get(c.status, "?")
            srcs = ", ".join(c.supported_by) if c.supported_by else "—"
            sug = f" | Suggestion: {c.suggestion}" if c.suggestion else ""
            lines.append(f"{tag} {c.claim}  (sources: {srcs}){sug}")
        return "\n".join(lines)

# ==========================
# ==== Demo / CLI Main   ===
# ==========================

if __name__ == "__main__":
    # Choose stack via env: RK_STACK in {"chroma","pgvector"}
    stack = os.environ.get("RK_STACK", "chroma").lower()

    # Choose LLM
    use_mock = os.environ.get("RK_USE_MOCK_LLM", "0") == "1"
    if use_mock:
        llm: LLMClient = MockLLM()
    else:
        # Default to OpenAI Responses adapter. Swap to Azure by instantiating AzureOpenAIChatAdapter().
        llm = OpenAIResponsesAdapter(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

    # Build retriever + embedder
    if stack == "pgvector":
        dsn = os.environ.get("PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
        embedder: Embedder = OpenAIEmbedder()
        retr = PGVectorRetriever(dsn=dsn, dim=3072)
        # Seed a couple docs idempotently
        docs = [
            ("pa", "Model A typical on-device latency is ~80ms for 256 tokens on Snapdragon X.", "kb://models#A_latency", "2025-01-15"),
            ("pb", "Model B typical on-device latency is ~120ms for 256 tokens on Snapdragon X.", "kb://models#B_latency", "2025-02-10"),
            ("pm", "Model A memory footprint: ~1.8GB int8 quantized; Model B: ~2.3GB int8.", "kb://models#mem", "2025-02-12"),
        ]
        embs = embedder.embed([d[1] for d in docs])
        retr.add(((d[0], d[1], d[2], d[3], e) for d, e in zip(docs, embs)))
        k_agent = KAgent(retriever=retr, embedder=embedder)
    else:
        # Chroma
        embedder = OpenAIEmbedder()
        retr = ChromaRetriever(collection="rk_demo", persist_dir=os.environ.get("CHROMA_DIR", ".chroma"), embedder=embedder)
        if retr.count() == 0:
            texts = [
                "Model A typical on-device latency is ~80ms for 256 tokens on Snapdragon X.",
                "Model B typical on-device latency is ~120ms for 256 tokens on Snapdragon X.",
                "Model A memory footprint: ~1.8GB int8 quantized; Model B: ~2.3GB int8.",
                "Model A license permits on-device commercial use with attribution.",
                "Model B license restricts redistribution of weights; on-device inference allowed.",
            ]
            metas = [
                {"source": "kb://models#A_latency", "date": "2025-01-15"},
                {"source": "kb://models#B_latency", "date": "2025-02-10"},
                {"source": "kb://models#mem", "date": "2025-02-12"},
                {"source": "kb://models#licenseA", "date": "2024-11-02"},
                {"source": "kb://models#licenseB", "date": "2025-03-01"},
            ]
            retr.add(ids=[f"r{i}" for i in range(len(texts))], texts=texts, metadatas=metas)
        k_agent = KAgent(retriever=retr)

    r_agent = RAgent(llm=llm)
    controller = HybridController(r=r_agent, k=k_agent, enable_critic_pass=True, use_k_based_checker=True)

    question = os.environ.get("RK_QUESTION", "Compare Model A vs Model B for on-device summarization.")
    result = controller.solve(question)

    print("\n=== PLAN ===")
    print(json.dumps(result["plan"], indent=2))
    print("\n=== FINAL ===")
    print(result["final"]) 
    print("\n=== CHECKS ===")
    print(json.dumps(result["checks"], indent=2))
