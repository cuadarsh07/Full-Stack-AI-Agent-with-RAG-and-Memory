import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import json
from tavily import TavilyClient
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
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize the Groq client AND the new Tavily Web Search client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# --- THE BULLETPROOF BYPASS ---
class BulletproofHFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 100% CORRECT URL
        api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
        headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
        
        print("Sending request to Hugging Face...")
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
        
        try:
            result = response.json()
        except Exception:
            print(f"HF HTML Error. Status: {response.status_code}. Retrying in 5 seconds...")
            time.sleep(5)
            response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
            result = response.json()
            
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
embedding_model = BulletproofHFEmbeddings()
db = Chroma(persist_directory="./my_vector_db", embedding_function=embedding_model)


class SummaryRequest(BaseModel):
    text: str
    style: str

@app.post("/summarize")
def summarize_text(request: SummaryRequest):
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
        response_format={"type": "json_object"} 
    )
    
    raw_response = chat_completion.choices[0].message.content
    structured_summary = json.loads(raw_response)
    return structured_summary

# --- RAG ENDPOINT ---
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_document(request: QuestionRequest):
    docs = db.similarity_search(request.question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""You are a helpful company assistant. 
    Answer the user's question using ONLY the following context. 
    If the answer is not contained in the context, say exactly: 'I am sorry, but I do not have information about that in my documents.'
    
    Context:
    {context_text}
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ],
        model="llama-3.3-70b-versatile", 
    )
    
    return {
        "answer": chat_completion.choices[0].message.content,
        "sources_used": [doc.page_content for doc in docs]
    }

# --- NEW: LIVE WEB AGENT WITH TAVILY ---
class AgentRequest(BaseModel):
    question: str

def search_web_tool(query: str):
    """Searches the live internet and reads the top pages using Tavily."""
    try:
        # We ask Tavily to grab the top 3 web results
        response = tavily_client.search(query=query, search_depth="basic", max_results=3)
        # We clean the data so the LLM only gets the URL and the content
        results = [{"url": res["url"], "content": res["content"]} for res in response.get("results", [])]
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": f"Web search failed: {str(e)}"})

@app.post("/agent")
def run_agent(request: AgentRequest):
    tools_menu = [{
        "type": "function",
        "function": {
            "name": "search_web_tool",
            "description": "Search the live internet for current events, facts, sports scores, and real-time data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The exact Google search query to use."}
                },
                "required": ["query"],
            },
        },
    }]

    messages = [
        {
            "role": "system", 
            "content": """You are a professional, live web-search agent.
            RULES:
            1. If you don't know a fact or if it is a current event, use the search_web_tool.
            2. ALWAYS provide a direct, concise answer FIRST.
            3. NEVER narrate your internal process (e.g., never say 'I will search for...' or 'According to the tool...').
            4. NEVER output raw XML or <function> tags.
            5. If you use the tool, briefly cite the source URLs provided at the end of your answer."""
        },
        {"role": "user", "content": request.question}
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=messages,
        tools=tools_menu,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        
        if tool_call.function.name == "search_web_tool":
            arguments = json.loads(tool_call.function.arguments)
            search_query = arguments.get("query")
            
            web_data = search_web_tool(search_query)

            messages.append(response_message)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": "search_web_tool",
                "content": web_data,
            })

            final_response = client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=messages
            )
            
            return {
                "answer": final_response.choices[0].message.content,
                "used_tool": True,
                "search_query": search_query
            }

    return {
        "answer": response_message.content, 
        "used_tool": False, 
        "search_query": None
    }

# --- MEMORY & CHAT HISTORY ---
class MessageItem(BaseModel):
    role: str
    content: str

class ChatHistoryRequest(BaseModel):
    messages: List[MessageItem]

@app.post("/chat")
def run_chat(request: ChatHistoryRequest):
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    system_prompt = {
        "role": "system", 
        "content": "You are a friendly, conversational AI. You have perfect memory of this conversation."
    }
    
    full_conversation = [system_prompt] + conversation_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=full_conversation
    )

    return {"reply": response.choices[0].message.content}