import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ==========================================
# 1. THE PYTHON FUNCTION (The Hands)
# ==========================================
def get_weather(city: str):
    print(f"⚙️ [PYTHON BACKEND]: The AI asked me to check the weather for {city}...")
    # In a real app, you'd connect to a real Weather API here. 
    # For now, we just fake the database lookup to keep it simple!
    if "chennai" in city.lower():
        return '{"temp": "32°C", "condition": "Sunny"}'
    else:
        return '{"temp": "15°C", "condition": "Rainy"}'

# ==========================================
# 2. THE MENU (Telling the AI what the tool does)
# ==========================================
tools_menu = [
    {
        "type": "function",
        "function": {
            "name": "get_weather", # Must match the exact Python function name
            "description": "Get the current weather for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g. Chennai, London",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# ==========================================
# 3. THE CONVERSATION 
# ==========================================
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use your tools to answer questions."},
    {"role": "user", "content": "Do I need an umbrella in Chennai today?"}
]

print("👤 [USER]: Do I need an umbrella in Chennai today?")
print("🧠 [AI]: Thinking...")

# Send the question AND the Menu to Groq
response = client.chat.completions.create(
    model="llama-3.1-8b-instant", # This model is great at tool calling
    messages=messages,
    tools=tools_menu,
    tool_choice="auto" # This tells the AI it can choose to use a tool if it wants
)

# ==========================================
# 4. CHECK IF THE AI WANTED TO USE A TOOL
# ==========================================
response_message = response.choices[0].message

if response_message.tool_calls:
    print("🧠 [AI]: Wait, I don't know the weather! Let me call a Python tool.")
    
    # Get the details of the tool the AI wants to use
    tool_call = response_message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # If the AI asked for "get_weather", we run our Python code!
    if function_name == "get_weather":
        # Extract the city the AI found in the user's sentence
        city_to_check = arguments.get("city")
        
        # RUN OUR PYTHON FUNCTION!
        weather_data = get_weather(city_to_check)
        print(f"⚙️ [PYTHON BACKEND]: Got the data! Returning it to the AI: {weather_data}")
        
        # Now we hand the data back to the AI so it can formulate a final sentence
        messages.append(response_message) # Add the AI's tool request to history
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "get_weather",
            "content": weather_data, # SNEAKING THE DATA BACK TO THE AI!
        })
        
        # Ask the AI to read the data and give a final answer
        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        print(f"\n🧠 [AI FINAL ANSWER]: {final_response.choices[0].message.content}")

else:
    print(f"\n🧠 [AI]: {response_message.content}")