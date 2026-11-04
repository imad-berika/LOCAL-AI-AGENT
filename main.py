from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import retriever

template = """
You are an expert answering questions about a pizza restaurant.
Answer using only the reviews below. If they do not contain the answer, say so.

Relevant reviews:
{reviews}

Question: {question}
"""

prompt = PromptTemplate.from_template(template=template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model


def format_reviews(docs):
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
        continue

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": format_reviews(reviews), "question": question})
    print(result)
