## Local AI Agent — Pizza Restaurant Reviews Q&A

A fully local retrieval-augmented generation (RAG) agent. Ask questions about a pizza restaurant in plain language; the agent retrieves the most relevant customer reviews and answers using only those reviews.

Everything runs on your machine through [Ollama](https://ollama.com) — no API keys, no cloud calls.

## How it works

```
question ──► Chroma vector store ──► top 5 similar reviews ──► prompt ──► llama3.2 ──► answer
             (mxbai-embed-large)
```

1. `vector.py` embeds each review (title + text) with `mxbai-embed-large` and stores it in a persistent Chroma database (`chroma_langchain_db/`). The CSV is loaded only when the collection is empty.
2. `main.py` runs an interactive loop. For each question it retrieves the 5 closest reviews, formats them as `- [rating/5, date] text`, and sends them with the question to `llama3.2`.
3. The prompt tells the model to answer only from the reviews and to say so when they don't contain the answer.

## Files

| File | Purpose |
|---|---|
| `main.py` | Interactive question loop and LLM chain |
| `vector.py` | Builds/loads the Chroma vector store and exposes `retriever` |
| `realistic_restaurant_reviews.csv` | 123 reviews (`Title`, `Date`, `Rating`, `Review`) |
| `requirements.txt` | Python dependencies |
| `chroma_langchain_db/` | Generated vector database (safe to delete; rebuilt on next run) |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- The two Ollama models:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The first run embeds all reviews, which takes a little while. Later runs reuse the saved database.

```
Ask your question (q to quit): what is the best pizza in town?
```

Type `q` to quit. Empty input is ignored.

## Using your own data

Replace `realistic_restaurant_reviews.csv` with a file that has the columns `Title`, `Date`, `Rating`, `Review`, then delete the `chroma_langchain_db/` folder so the index is rebuilt. If you change the data without deleting the folder, the old index is kept because it is not empty.

## Configuration

| What | Where |
|---|---|
| Chat model | `OllamaLLM(model="llama3.2")` in `main.py` |
| Embedding model | `OllamaEmbeddings(model="mxbai-embed-large")` in `vector.py` |
| Reviews retrieved per question | `search_kwargs={"k": 5}` in `vector.py` |
| Prompt wording | `template` in `main.py` |

If you change the embedding model, delete `chroma_langchain_db/` first. Vectors from different models are not compatible.
