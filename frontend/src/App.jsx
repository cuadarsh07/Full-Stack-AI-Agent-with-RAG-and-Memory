import { useState } from 'react';
import DocumentChat from './DocumentChat'; 
import Agent from './Agent';             
import Chat from './Chat'; 

export default function App() {
  // This state controls which tool is currently visible
  const [activeTab, setActiveTab] = useState('agent');

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans flex flex-col">
      
      {/* 1. The Navigation Bar */}
      <nav className="flex justify-center gap-4 p-6 border-b border-gray-800 bg-gray-900/50">
        <button 
          onClick={() => setActiveTab('docChat')}
          className={`px-6 py-2 rounded-lg font-semibold transition-all ${
            activeTab === 'docChat' 
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30' 
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          📄 Document Q&A
        </button>
        
        <button 
          onClick={() => setActiveTab('agent')}
          className={`px-6 py-2 rounded-lg font-semibold transition-all ${
            activeTab === 'agent' 
              ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-500/30' 
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          🌐 Web Agent
        </button>

        {/* --- ADDED THE 3RD BUTTON HERE --- */}
        <button 
          onClick={() => setActiveTab('chat')}
          className={`px-6 py-2 rounded-lg font-semibold transition-all ${
            activeTab === 'chat' 
              ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/30' 
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          💬 Smart Chat
        </button>

      </nav>

      {/* 2. The Active Tool */}
      <main className="flex-1 w-full h-full overflow-auto">
        {activeTab === 'docChat' && <DocumentChat />}
        {activeTab === 'agent' && <Agent />}
        {/* --- ADDED THE 3RD COMPONENT HERE --- */}
        {activeTab === 'chat' && <Chat />}
      </main>

    </div>
  );
}