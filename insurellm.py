import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI


load_dotenv(override=True) # Load environment variables from .env file, overriding existing ones if necessary

openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-nano"
openai = OpenAI()

knowledge = {} # Dictionary to hold the knowledge base, where keys are employee names and values are their information

filenames = glob.glob("knowledge-base/employees/*") # Get all files in the knowledge-base/employees directory

for filename in filenames: # Loop through each file and read its content
    name = Path(filename).stem.split(' ')[-1] # Extract the employee name from the filename (assuming the format is "employee_name.txt")
    with open(filename, "r", encoding="utf-8") as f: # Open the file and read its content
        knowledge[name.lower()] = f.read()
        
knowledge["lancaster"]

SYSTEM_PREFIX = """
You are Insurellm AI, an intelligent assistant for Insurellm, an Insurance Technology company.

Your role is to help employees by answering questions about:
- Insurellm products and services
- Company policies and procedures
- Employee-related information
- Insurance-related knowledge contained in the provided documents

Instructions:
1. Use the provided context as your primary source of information.
2. Answer clearly, accurately, and concisely.
3. If the answer is not available in the provided context, say:
   "I don't have enough information to answer that question."
4. Do not make up facts or assumptions.
5. When possible, base your response directly on the retrieved information.

Relevant Context:
"""

def get_relevant_context(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    return [knowledge[word] for word in words if word in knowledge]   

get_relevant_context("Who is lancaster?")

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result

print(additional_context("Who is Alex Lancaster?"))

def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

view = gr.ChatInterface(chat).launch(inbrowser=True)