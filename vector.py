"""Builds (or loads) the Chroma vector store of restaurant reviews and
exposes a `retriever` that main.py uses to look up relevant reviews."""

from pathlib import Path
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Paths are resolved relative to this file so the script works from any cwd.
BASE = Path(__file__).parent
CSV_PATH = BASE / "realistic_restaurant_reviews.csv"
DB_LOCATION = str(BASE / "chroma_langchain_db")

# Local embedding model served by Ollama; turns text into vectors.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Opens the persistent collection, creating it on the first run.
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

# Load the CSV only when the collection is empty (the folder alone existing
# is not enough: a failed first run leaves an empty DB behind).
if vector_store._collection.count() == 0:
    df = pd.read_csv(CSV_PATH)
    documents, ids = [], []
    for i, row in df.iterrows():
        # The text that gets embedded and searched: title + review body.
        # Rating and date are kept as metadata so main.py can show them.
        documents.append(Document(
            page_content=f'{row["Title"]} {row["Review"]}',
            metadata={"rating": int(row["Rating"]), "date": row["Date"]},
        ))
        # Row index as the id, so re-adding the same row cannot create duplicates.
        ids.append(str(i))
    vector_store.add_documents(documents=documents, ids=ids)

# Returns the 5 reviews most similar to a question.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
