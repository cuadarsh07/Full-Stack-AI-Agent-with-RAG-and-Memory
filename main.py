import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import json
import wikipedia
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from typing import List
import requests
import time
from langchain_core.embeddings import Embeddings


# 1. Load the secret API key from the .env file
load_dotenv() 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (good for local testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize the Groq client (The Chef)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# --- THE BULLETPROOF BYPASS ---
class BulletproofHFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
        
        print("Sending request to Hugging Face...")
        # We force Hugging Face to wake up the model instead of immediately returning an error
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        result = response.json()
        
        # If HF still throws an error, we catch it, print it to Render logs, and retry!
        if isinstance(result, dict) and "error" in result:
            print(f"HF Error Detected: {result['error']}. Retrying in 5 seconds...")
            time.sleep(5)
            response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
            result = response.json()
            
        return result

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# --- LOAD THE DATABASE ON STARTUP ---
print("Loading Vector Database...")
# We use our custom bypass class now!
embedding_model = BulletproofHFEmbeddings()
db = Chroma(persist_directory="./my_vector_db", embedding_function=embedding_model)


class SummaryRequest(BaseModel):
    text: str
    style: str

@app.post("/summarize")
def summarize_text(request: SummaryRequest):
    # 1. Update the instructions to DEMAND a specific JSON structure
    system_prompt = f"""You are an expert editor. Summarize the text provided by the user. 
    Use this exact style: {request.style}. 
    You MUST respond in valid JSON format. 
    Your JSON must contain exactly two keys: 
    'title' (a short, catchy title for the text) and 
    'content' (the actual summary)."""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.text}
        ],
        model="llama-3.3-70b-versatile",
        # 2. The Magic Line: This forces the model to only output valid JSON
        response_format={"type": "json_object"} 
    )
    
    # 3. The AI gives us a JSON string, so we convert it into a real Python dictionary
    raw_response = chat_completion.choices[0].message.content
    structured_summary = json.loads(raw_response)
    
    # Now we can send this beautifully structured data to our frontend!
    return structured_summary

# --- NEW RAG ENDPOINT ---
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_document(request: QuestionRequest):
    # 1. Search the Vector Database for the 3 most relevant chunks
    docs = db.similarity_search(request.question, k=3)
    
    # 2. Combine those chunks into one big string of text
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. Build the strict "Open-Book" instructions for the AI
    system_prompt = f"""You are a helpful company assistant. 
    Answer the user's question using ONLY the following context. 
    If the answer is not contained in the context, say exactly: 'I am sorry, but I do not have information about that in my documents.'
    
    Context:
    {context_text}
    """
    
    # 4. Send the prompt to the Groq Chef
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ],
        model="llama-3.3-70b-versatile",
    )
    
    # 5. Return the AI's answer, PLUS the actual chunks we used so the user can see the proof!
    return {
        "answer": chat_completion.choices[0].message.content,
        "sources_used": [doc.page_content for doc in docs]
    }

# --- NEW MONTH 3: AI AGENT WITH TOOLS ---
class AgentRequest(BaseModel):
    question: str

def search_wikipedia_tool(query: str):
    """The actual Python function that searches the web."""
    try:
        return json.dumps({"result": wikipedia.summary(query, sentences=2)})
    except Exception:
        return json.dumps({"error": "Could not find a Wikipedia page for that exact term."})

@app.post("/agent")
def run_agent(request: AgentRequest):
    # 1. The Menu
    tools_menu = [{
        "type": "function",
        "function": {
            "name": "search_wikipedia_tool",
            "description": "Search Wikipedia for facts, history, or current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact search term."}
                },
                "required": ["query"],
            },
        },
    }]

    # 2. The Initial Request
    messages = [
        {
            "role": "system", 
            "content": """You are a smart AI Agent. If you do not know a fact, use the search_wikipedia_tool. 
            CRITICAL RULES FOR WIKIPEDIA: 
            1. Wikipedia only accepts short, exact page names (e.g., 'Suryakumar Yadav'). 
            2. NEVER use long sentences or action words in your search query. 
            3. If the user uses abbreviations, convert them to the full name before searching.
            4. NEVER output raw XML or <function> tags in your final response. If the tool doesn't give you the exact stats, just politely tell the user the data isn't in the Wikipedia summary."""
        },
        {"role": "user", "content": request.question}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Llama 3.3 is specifically trained for tool calling!
        messages=messages,
        tools=tools_menu,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # 3. If the AI decides to use the tool...
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        
        if tool_call.function.name == "search_wikipedia_tool":
            arguments = json.loads(tool_call.function.arguments)
            search_query = arguments.get("query")
            
            # Run the Python function!
            wiki_data = search_wikipedia_tool(search_query)

            # Hand the data back to the AI
            messages.append(response_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "search_wikipedia_tool",
                "content": wiki_data,
            })

            # Get the final human-readable answer
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            
            # Return the answer AND proof that we searched the web!
            return {
                "answer": final_response.choices[0].message.content,
                "used_tool": True,
                "search_query": search_query
            }

    # If the AI didn't need the tool (e.g., you just said "Hello")
    return {
        "answer": response_message.content, 
        "used_tool": False, 
        "search_query": None
    }

# --- NEW MONTH 4: MEMORY & CHAT HISTORY ---
class MessageItem(BaseModel):
    role: str
    content: str

class ChatHistoryRequest(BaseModel):
    # Instead of one string, we ask for a LIST of messages!
    messages: List[MessageItem]

@app.post("/chat")
def run_chat(request: ChatHistoryRequest):
    # 1. Convert the Pydantic request into a standard Python dictionary format for Groq
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    # 2. Create the System Prompt (The AI's personality)
    system_prompt = {
        "role": "system", 
        "content": "You are a friendly, conversational AI. You have perfect memory of this conversation."
    }
    
    # 3. Combine the System Prompt with the entire transcript the frontend sent us
    full_conversation = [system_prompt] + conversation_history

    # 4. Send the ENTIRE transcript to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=full_conversation
    )

    # 5. Return only the AI's newest reply
    return {"reply": response.choices[0].message.content}