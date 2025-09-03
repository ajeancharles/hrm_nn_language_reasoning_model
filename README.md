# Hierarchical Reasoning and Knowledge System

This repository contains a pilot project for a Hierarchical Reasoning and Knowledge (R-K) System. It is designed to answer complex queries by breaking them down into smaller sub-questions, retrieving relevant facts from a knowledge base, and then synthesizing an answer.

The core of the application is a Python-based system that demonstrates a hybrid architecture combining a "Reasoning Agent" and a "Knowledge Agent".

## Architecture Overview

The system is built around a `HybridController` that orchestrates the workflow between two main agents:

*   **Reasoning Agent (`RAgent`)**: This agent is responsible for the higher-level reasoning tasks. It takes a user query, breaks it down into a plan of smaller, answerable sub-questions, and composes the final answer based on the evidence gathered.
*   **Knowledge Agent (`KAgent`)**: This agent is responsible for retrieving information from a knowledge base. It can perform semantic searches to find relevant facts and can also be used for fact-checking claims.

The system supports multiple backends for knowledge retrieval and LLM providers:

*   **Knowledge Retrieval**:
    *   `ChromaDB`: A local, file-based vector store.
    *   `pgvector`: A PostgreSQL-based vector store.
*   **LLM Providers**:
    *   OpenAI (Chat and Responses APIs)
    *   Azure OpenAI

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

The system is configured using environment variables.

**1. LLM Configuration:**

You need to provide credentials for your chosen LLM provider. The system primarily uses the following variables:

*   `OPENAI_API_KEY`: Your API key.
*   `OPENAI_END_POINT`: The base URL of the API endpoint.
*   `OPENAI_DEPLOYMENT`: The name of the model/deployment you want to use (e.g., `gpt-4o-mini`).

**2. Embeddings Configuration:**

Similarly, configure the endpoint for the text embedding model:

*   `OPENAI_EMBED_API_KEY`: Your API key for the embedding service.
*   `OPENAI_EMBED_END_POINT`: The endpoint for the embedding service.
*   `OPENAI_EMBED_DEPLOYMENT`: The name of the embedding model deployment (e.g., `text-embedding-3-large`).

**3. Knowledge Base Configuration:**

You can choose between `chroma` (default) and `pgvector` as the knowledge base.

*   **ChromaDB (default)**: No specific configuration is needed. It will create a local database in the `.chroma` directory.
*   **pgvector**:
    *   Set the `RK_STACK` environment variable:
        ```bash
        export RK_STACK=pgvector
        ```
    *   Set the PostgreSQL connection string:
        ```bash
        export PG_DSN="postgresql://user:pass@host:5432/db"
        ```
    *   You also need to initialize the database table and extension in PostgreSQL (this only needs to be done once):
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS rk_docs (
            id text primary key,
            text text not null,
            source text,
            date text,
            embedding vector(3072)
        );
        CREATE INDEX IF NOT EXISTS rk_docs_cos ON rk_docs USING ivfflat (embedding vector_cosine_ops);
        ```

## Usage

The main entry point for the application is `reasonning_models/hybrid_rk_merged.py`.

You can run it directly from the command line. You can set the question via an environment variable.

```bash
# Make sure you have set your OPENAI_* environment variables first
export RK_QUESTION="Compare Model A vs Model B for on-device summarization."
python reasonning_models/hybrid_rk_merged.py
```

The script will then print the execution plan, the final answer, and the results of the grounding/fact-checking audit.

## Project Structure

*   `reasonning_models/`: Contains the core logic for the R-K system.
    *   `hybrid_rk_merged.py`: The main, runnable script that orchestrates the entire reasoning and retrieval process.
*   `HRM_NN_Language_Reasononing/`: Contains configuration files and related utilities.
    *   `config/`: Contains configuration loaders and validators.
    *   `data/`: Sample data files.
*   `download_supreme_court_and_sec_filings/`: Contains scripts and data related to legal document analysis.
*   `*.ipynb`: Jupyter notebooks demonstrating various concepts and earlier versions of the system.
*   `requirements.txt`: A list of the Python packages required to run the project.

---

## Conceptual Overview

*The following is the original conceptual documentation from the initial version of this README.*

### HRM for Logical Processing and Analysis

This is more of a demonstration of structure. More data and logic are needed.

Take a look at **HRM_L0toL5.ipynb**

**Design the Hierarchical Architecture**
NN Layer Structure:
*   **Layer 0 (Token Level)**: Character/subword processing, basic pattern recognition
*   **Layer 1 (Syntactic Level)**: Grammar rules, part-of-speech tagging, dependency parsing
*   **Layer 2 (Semantic Level)**: Entity recognition, relation extraction, semantic role labeling
*   **Layer 3 (Discourse Level)**: Coreference resolution, discourse markers, paragraph coherence
*   **Layer 4 (Pragmatic Level)**: Intent understanding, context integration, world knowledge
*   **Layer 5 (Meta-Reasoning Level)**: Strategic reasoning, planning, meta-cognitive processes

### Legal Analysis with Legal Entities

**legal_analysis_with_legal_entities.ipynb** demonstrates examples of usage of legal semantic primitives to analyze legal text.
Common logical primitives in legal reasoning—such as entities, relationships, obligations, permissions, prohibitions, and conditions—serve as the key building blocks for representing, analyzing, and automating legal texts. Each primitive holds a distinct role in both human judicial reasoning and computational legal systems.

*   **Entities**: Represent the fundamental "actors" or "things" within legal scenarios—people, corporations, governments, physical items, or abstract concepts like rights or liabilities.
*   **Relationships**: Describe how entities connect in legal contexts, represented through verbs or predicates such as "owns", "employs", "transfers", or "is-party-to".
*   **Obligations**: Normative statements requiring that an entity perform a specific action, typically formulated with words like "must", "shall", or "is required to".
*   **Permissions**: Identify actions that entities are explicitly allowed to perform, denoted by terms such as "may", "can", or "is permitted to".
*   **Prohibitions**: Mark actions that are forbidden, commonly structured as "must not", "shall not", or "is prohibited from".
*   **Conditions**: Operate as antecedents for other logical primitives—often beginning with "if", "when", or "unless".

**Integration in Legal Reasoning**
Legal texts are often constructed as nested or chained structures: “If (Condition), then (Obligation/Permission/Prohibition) applies to (Entity) in relation to (Other Entity/Relationship).”
Advanced AI models and rule-based systems formalize these primitives using logical notation, decision trees, or object-oriented design, accurately reflecting the complexity and semantics of legal language.
