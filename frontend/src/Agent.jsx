import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Globe, Sparkles, Loader2 } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://ai-agent-backend-l44r.onrender.com';

export default function App() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAskAgent = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/agent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error connecting to Agent:", error);
      alert("Backend connection failed. Check VITE_API_URL or the FastAPI server.");
    }

    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col items-center pt-20 px-4 font-sans">
      
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center p-3 bg-indigo-500/10 rounded-2xl mb-4 text-indigo-400">
          <Globe className="w-8 h-8" />
        </div>
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4">
          Live Web <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">Agent</span>
        </h1>
        <p className="text-gray-400 max-w-lg mx-auto text-lg">
          Ask any factual question. The AI will autonomously search the live internet to find the answer.
        </p>
      </div>

      {/* Input Form */}
      <form onSubmit={handleAskAgent} className="w-full max-w-2xl relative mb-12">
        <div className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
          <div className="relative flex items-center bg-gray-900 rounded-2xl border border-gray-800 focus-within:border-indigo-500/50 shadow-2xl transition-all">
            <Search className="w-6 h-6 text-gray-500 ml-4" />
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="E.g., Who won the Super Bowl in 2024?"
              className="w-full bg-transparent border-none text-white px-4 py-5 focus:outline-none placeholder-gray-600 text-lg"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !question}
              className="mr-2 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-800 disabled:text-gray-500 text-white font-semibold rounded-xl transition-colors flex items-center gap-2"
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Sparkles className="w-5 h-5" />}
              {isLoading ? 'Searching...' : 'Ask Agent'}
            </button>
          </div>
        </div>
      </form>

      {/* Loading State Animation */}
      {isLoading && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-2xl bg-gray-900/50 border border-gray-800 rounded-2xl p-6"
        >
          <div className="flex items-center gap-4 text-indigo-400 mb-4">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="font-medium animate-pulse">Agent is deciding which tool to use...</span>
          </div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-800 rounded animate-pulse w-3/4"></div>
            <div className="h-4 bg-gray-800 rounded animate-pulse w-full"></div>
            <div className="h-4 bg-gray-800 rounded animate-pulse w-5/6"></div>
          </div>
        </motion.div>
      )}

      {/* Result Card */}
      {result && !isLoading && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-2xl bg-gray-900 border border-gray-800 rounded-2xl p-8 shadow-2xl relative overflow-hidden"
        >
          {/* Subtle background glow */}
          <div className="absolute top-0 right-0 -mt-4 -mr-4 w-32 h-32 bg-indigo-500/10 blur-3xl rounded-full"></div>

          {/* If the tool was used, show the badge! */}
          {result.used_tool && (
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 rounded-full text-sm font-medium mb-6">
              <Globe className="w-4 h-4" />
              Searched Wikipedia for: "{result.search_query}"
            </div>
          )}

          <div className="text-gray-200 text-lg leading-relaxed">
            {result.answer}
          </div>
        </motion.div>
      )}

    </div>
  );
}