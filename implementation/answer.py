from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(temperature=0, model_name=MODEL)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing Insurellm.
Use the retrieved context to answer questions.
If you don’t know, say so.
Context:
{context}
"""


def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant context documents."""
    return retriever.invoke(query)


def answer_question(query: str):
    """Generate an answer with context injection."""
    docs = fetch_context(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=query),
    ]

    response = llm.invoke(messages)
    return response.content, docs
