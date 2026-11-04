from pathlib import Path
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE = Path(__file__).parent
CSV_PATH = BASE / "realistic_restaurant_reviews.csv"
DB_LOCATION = str(BASE / "chroma_langchain_db")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

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
        documents.append(Document(
            page_content=f'{row["Title"]} {row["Review"]}',
            metadata={"rating": int(row["Rating"]), "date": row["Date"]},
        ))
        ids.append(str(i))
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
