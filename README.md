# 🤖 AI Text Summarizer

A modern full-stack web application that uses AI to transform lengthy content into concise, actionable summaries. Perfect for summarizing articles, documents, reports, and more.

![AI Text Summarizer](https://img.shields.io/badge/AI-Powered-blueviolet) ![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688) ![React](https://img.shields.io/badge/React-Frontend-61DAFB)

## ✨ Features

- 📝 **Multiple Summary Styles**
  - Bullet Points (quick overview)
  - Short Paragraph (2-sentence summary)
  - Detailed & Formal (comprehensive summary)

- 📊 **Real-time Analytics**
  - Word count tracking (input/output)
  - Character count monitoring
  - Compression percentage display

- 📋 **User-Friendly Interface**
  - One-click copy to clipboard
  - Instant summary generation
  - Loading states with smooth animations
  - Responsive design for all devices

- 🎨 **Modern UI/UX**
  - Beautiful purple gradient theme
  - Smooth transitions and animations
  - Professional card-based layout
  - Mobile-responsive design

## 🚀 Tech Stack

**Frontend:**
- React 18
- Vite (build tool)
- Custom CSS3 with animations

**Backend:**
- FastAPI (Python)
- Groq API integration
- Uvicorn server

**AI Model:**
- Groq API with Llama 3 (70B parameters)
- Fast inference times
- High-quality summaries

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Groq API key ([Get it free](https://console.groq.com))

### Backend Setup

1. **Install dependencies:**
```bash
pip install fastapi uvicorn groq python-dotenv
```

Or using Poetry:
```bash
poetry install
```

2. **Create `.env` file in root directory:**
```env
GROQ_API_KEY=your_groq_api_key_here
```

3. **Run the FastAPI server:**
```bash
python main.py
```

Server will start at `http://127.0.0.1:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Start development server:**
```bash
npm run dev
```

Frontend will start at `http://localhost:5173`

## 🎯 Usage

1. Open `http://localhost:5173` in your browser
2. Paste your long text into the input area
3. Select your preferred summary style
4. Click "Summarize Text"
5. View your summary with word count statistics
6. Click "Copy" to copy the summary to clipboard

## 🔑 Getting Your Groq API Key

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste it into your `.env` file

## 📁 Project Structure

```
fastapi/
├── main.py                 # FastAPI backend server
├── pyproject.toml         # Python dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore            # Git ignore rules
└── frontend/
    ├── src/
    │   ├── App.jsx       # Main React component
    │   ├── App.css       # Custom styles
    │   └── main.jsx      # React entry point
    ├── index.html        # HTML template
    ├── package.json      # Node dependencies
    └── vite.config.js    # Vite configuration
```

## 🌟 Features Showcase

### Smart Word Count
Both input and output text display real-time word and character counts, helping you understand the compression ratio.

### Copy to Clipboard
One-click copy functionality with visual feedback ("Copied!" message) makes it easy to use the summaries.

### Compression Statistics
See exactly how much your text was reduced by with percentage badges.

### Responsive Design
Works perfectly on desktop, tablet, and mobile devices.

## 🛠️ API Endpoints

### `POST /summarize`

**Request Body:**
```json
{
  "text": "Your long text here...",
  "style": "3 concise bullet points"
}
```

**Response:**
```json
{
  "title": "Generated Title",
  "content": "Summary content..."
}
```

## 📸 Screenshots

*Add your screenshots here after deployment*

## 🤝 Contributing

Feel free to fork this project and submit pull requests!

## 📄 License

MIT License - feel free to use this project for your portfolio or learning purposes.

## 👨‍💻 Author

Built with ❤️ using FastAPI, React, and Groq AI

---

⭐ Star this repo if you find it useful!
