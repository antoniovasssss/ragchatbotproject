from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

MODEL = "gpt-4.1-nano"
DB_NAME = "vector_db"
load_dotenv(override=True) # Load environment variables from .env file, overriding existing ones if necessary

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)

retriever.invoke("Who is Avery?") # This will return the most relevant documents from the vector store based on the query "Who is Avery?"

llm.invoke("Who is Avery?") # This will generate a response from the language model based on the query "Who is Avery?" without any additional context.

SYSTEM_PROMPT_TEMPLATE = """
You are InsureLLM, a knowledgeable and friendly AI assistant representing the company InsureLLM.

Your role is to help users by providing accurate, clear, and helpful information about the company and its services.

Use the provided context when it is relevant to the user's question. If the answer cannot be found in the context or you are unsure, honestly state that you do not know rather than making up information.

Context:
{context}
"""

def answer_question(question: str, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=question)])
    return response.content

answer_question("Who is Avery Lancaster?", [])

# gr.ChatInterface(answer_question).launch()