# ☕ Spill the Tea - TMZ Celebrity Gossip Chatbot

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

> *"Your AI bestie for all the celebrity tea! ☕💅✨"*

A Retrieval-Augmented Generation (RAG) powered chatbot that scrapes TMZ for the latest celebrity news and serves it up with a sassy, gossip-lounge personality. Built with Streamlit, FAISS vector search, and Mistral AI.

## 🌟 Features

### 🔍 Smart Search & Retrieval
- **Semantic Search**: Uses sentence transformers and FAISS for intelligent content retrieval
- **TMZ Web Scraping**: Automatically scrapes and chunks celebrity news articles
- **Vector Embeddings**: Converts articles into searchable embeddings for accurate results
- **Source Attribution**: Links back to original TMZ articles

### 💬 Conversational AI
- **Mistral Integration**: Powered by Mistral AI for natural language responses
- **Sassy Personality**: Gossip-style responses with phrases like "Here's the tea ☕" and "Spill the gossip ✨"
- **Context-Aware**: Uses RAG to provide accurate, source-backed answers
- **Chat History**: Sidebar tracks all your gossip sessions

### 🎨 Beautiful UI
- **Glamorous Design**: Pink/purple gradients and elegant typography
- **Quick Topics**: Pre-configured questions for instant tea
- **Session Management**: Save and revisit past conversations
- **Responsive Layout**: Clean, modern interface with smooth animations

## 🚀 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Mistral AI API
- **Vector Search**: FAISS + Sentence Transformers
- **Web Scraping**: BeautifulSoup4, Requests
- **Data Processing**: NumPy, Python JSON

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Mistral API key ([Get one here](https://mistral.ai/))

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/spill-the-tea-chatbot.git
cd spill-the-tea-chatbot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create requirements.txt (if not exists)
```txt
streamlit>=1.28.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
numpy>=1.24.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

## ⚙️ Configuration

### 1. Set Up Mistral API Key

Open your main Streamlit app file and add your Mistral API key:

```python
# Backend configuration - Set your Mistral API key here
MISTRAL_API_KEY = "your_actual_mistral_api_key_here"
```

**Security Note**: For production, use environment variables:
```python
import os
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
```

### 2. Scrape TMZ Data

Run the scraper to collect celebrity news:

```python
from tmz_scraper import TMZChunkingScraper

scraper = TMZChunkingScraper()
scraper.scrape_all_pages(num_pages=5)  # Adjust number of pages
scraper.create_chunks()
scraper.save_data("tmz_data.json")
```

## 🎯 Usage

### Running the Chatbot

**Option 1: Streamlit Web App**
```bash
streamlit run app.py
```

**Option 2: Command Line Interface**
```python
from tmz_rag_bot import TMZRAGBot

bot = TMZRAGBot(data_file="tmz_data.json")
bot.interactive_chat()
```

### Using the Interface

1. **Quick Topics**: Click pre-configured questions for instant gossip
2. **Custom Questions**: Type your own celebrity questions in the chat input
3. **Chat History**: View and manage past conversations in the sidebar
4. **New Session**: Start fresh conversations with the "New Chat" button

### Example Queries

```
💅 "What's Taylor Swift's latest drama?"
☕ "Tell me about celebrity wedding gossip"
✨ "Any Hollywood breakup tea?"
💋 "What's happening with red carpet fashion?"
🔥 "Give me the latest celebrity baby news"
```

## 📁 Project Structure

```
spill-the-tea-chatbot/
│
├── app.py                    # Main Streamlit application
├── tmz_scraper.py           # Web scraping module
├── tmz_rag_bot.py           # RAG chatbot logic
├── tmz_data.json            # Scraped celebrity news data
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── .streamlit/              # Streamlit configuration (optional)
│   └── config.toml
│
└── assets/                  # Images and resources (optional)
    └── logo.png
```

## 🔧 Key Components

### TMZ Scraper (`tmz_scraper.py`)
- Scrapes TMZ articles with headlines and content
- Chunks content intelligently for better retrieval
- Saves structured JSON data

### RAG Bot (`tmz_rag_bot.py`)
- Loads and processes scraped data
- Creates FAISS vector index for semantic search
- Generates embeddings using sentence transformers
- Queries Mistral API for response generation

### Streamlit App (`app.py`)
- Beautiful, responsive UI with gossip-lounge aesthetic
- Session management and chat history
- Real-time chat interface
- Stats and RAG explanation

## 🎨 Customization

### Change Personality
Modify the system prompt in `tmz_rag_bot.py`:
```python
prompt = f"""You are a fun, sassy celebrity gossip expert...
# Customize the personality here!
"""
```

### Adjust Search Results
Change the number of retrieved articles:
```python
def search(self, query, top_k=3):  # Increase top_k for more context
```

### Modify UI Theme
Edit the CSS in the Streamlit app:
```python
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    /* Customize colors and styles */
</style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### "Bot isn't ready" Error
- Ensure `tmz_data.json` exists and contains valid data
- Run the scraper first to generate data

### Mistral API Errors
- Verify your API key is correct
- Check your API quota and rate limits
- Ensure you have internet connectivity

### FAISS Installation Issues
On some systems, you may need:
```bash
pip install faiss-cpu --no-cache-dir
# OR for GPU support:
pip install faiss-gpu
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 📊 Performance Tips

1. **Data Volume**: Start with 3-5 pages, expand as needed
2. **Chunk Size**: Adjust in scraper for better context (default: by heading)
3. **Model Selection**: Try different sentence transformer models:
   - `all-MiniLM-L6-v2` (default, fast)
   - `all-mpnet-base-v2` (more accurate)
4. **Caching**: Streamlit automatically caches the bot initialization

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add more celebrity news sources
- Improve the scraping logic
- Enhance the UI/UX
- Add sentiment analysis
- Create a mobile-responsive version
- Add voice interaction

## ⚠️ Disclaimer

This project is for educational purposes only. Respect TMZ's terms of service and copyright. The scraper should be used responsibly with appropriate rate limiting. Always verify information from official sources.

## 🙏 Acknowledgments

- **TMZ** for celebrity news content
- **Mistral AI** for the language model API
- **Sentence Transformers** for semantic search capabilities
- **Streamlit** for the amazing web framework
- **FAISS** for efficient vector search

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue
- Submit a pull request
- Reach out via email

---

<div align="center">

### ☕ Happy Tea Spilling! 💅✨

*Built with 💖 by [Sharmin Islam Disha]*

[⭐ Star this repo](https://github.com/dishaislam/spill-the-tea-chatbot) | [🐛 Report Bug](https://github.com/dishaislam/spill-the-tea-chatbot/issues) | [💡 Request Feature](https://github.com/dishaislam/spill-the-tea-chatbot/issues)

</div>
