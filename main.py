"""Interactive Q&A about a pizza restaurant, answered from customer reviews
retrieved from the vector store (see vector.py)."""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import retriever  # importing this also builds the DB on first run

# The model is told to stay grounded in the retrieved reviews.
template = """
You are an expert answering questions about a pizza restaurant.
Answer using only the reviews below. If they do not contain the answer, say so.

Relevant reviews:
{reviews}

Question: {question}
"""

prompt = PromptTemplate.from_template(template=template)
model = OllamaLLM(model="llama3.2")  # local chat model served by Ollama
chain = prompt | model  # fill the prompt, then send it to the model


def format_reviews(docs):
    """Turn retrieved documents into readable lines like
    '- [5/5, 2024-03-15] Best pizza in town ...' for the prompt."""
    return "\n".join(
        f'- [{d.metadata.get("rating")}/5, {d.metadata.get("date")}] {d.page_content}'
        for d in docs
    )


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    print("\n\n")
    if question.lower() == "q":
        break
    if not question:
        continue  # ignore empty input instead of querying the model

    # 1) find the reviews most similar to the question, 2) answer from them
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": format_reviews(reviews), "question": question})
    print(result)
