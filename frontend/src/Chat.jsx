import { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Loader2 } from 'lucide-react';

export default function Chat() {
  // This array is the "Transcript"! It holds the whole conversation.
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // This helps us automatically scroll to the bottom when a new message appears
  const messagesEndRef = useRef(null);
  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => scrollToBottom(), [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // 1. Add the user's new message to the transcript
    const userMessage = { role: 'user', content: input };
    const updatedTranscript = [...messages, userMessage];
    
    setMessages(updatedTranscript); // Update the screen immediately
    setInput(''); // Clear the input box
    setIsLoading(true);

    try {
      // 2. Send the ENTIRE transcript array to our new FastAPI /chat endpoint
      const response = await fetch('https://ai-agent-backend-l44r.onrender.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: updatedTranscript }),
      });

      const data = await response.json();

      // 3. Add the AI's reply to the transcript
      const aiMessage = { role: 'assistant', content: data.reply };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);

    } catch (error) {
      console.error("Chat error:", error);
      alert("Backend connection failed!");
    }

    setIsLoading(false);
  };

  return (
    <div className="flex flex-col h-[80vh] w-full max-w-3xl mx-auto bg-gray-900 border border-gray-800 rounded-2xl overflow-hidden shadow-2xl mt-10">
      
      {/* 1. The Scrolling Chat Window */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Say hello! I will remember everything we talk about.</p>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            
            {/* AI Avatar */}
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
            )}

            {/* The Message Bubble */}
            <div className={`max-w-[80%] p-4 rounded-2xl ${
              msg.role === 'user' 
                ? 'bg-cyan-600 text-white rounded-tr-none' 
                : 'bg-gray-800 text-gray-200 rounded-tl-none border border-gray-700'
            }`}>
              {msg.content}
            </div>

            {/* User Avatar */}
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-full bg-cyan-800 flex items-center justify-center shrink-0">
                <User className="w-5 h-5 text-cyan-200" />
              </div>
            )}
          </div>
        ))}
        
        {/* Loading Bubble */}
        {isLoading && (
          <div className="flex gap-4 justify-start">
            <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center shrink-0">
              <Loader2 className="w-5 h-5 text-white animate-spin" />
            </div>
            <div className="p-4 bg-gray-800 rounded-2xl rounded-tl-none border border-gray-700">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-75"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-150"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 2. The Input Box Area */}
      <div className="p-4 bg-gray-950 border-t border-gray-800">
        <form onSubmit={handleSendMessage} className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Message the AI..."
            className="flex-1 bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 transition-colors"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 text-white px-5 rounded-xl transition-colors flex items-center justify-center"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>

    </div>
  );
}