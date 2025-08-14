import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import random
from datetime import datetime

# Backend configuration - Set your Mistral API key here
MISTRAL_API_KEY = "Xj8iRe2Mi6Uhe0BrfHRdL6Rw75q9gI7Q"  # Replace with your actual API key

class TMZRAGBot:
    def __init__(self, data_file="tmz_data.json", model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.index = None
        self.metadata = []
        
        # Mistral API configuration - using backend key
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        self.mistral_api_key = MISTRAL_API_KEY
        
        self.load_data(data_file)
        self.build_vector_index()
    
    def load_data(self, data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_articles = data.get('raw_articles', [])
            all_chunks = []
            
            for article in raw_articles:
                for chunk_data in article['structured_content']:
                    if chunk_data.get('has_content', False):
                        chunk = {
                            'text': chunk_data['combined_text'],
                            'heading': chunk_data['heading'],
                            'content': chunk_data['content'],
                            'article_id': article['article_id'],
                            'page': article['page'],
                            'url': article['url']
                        }
                        all_chunks.append(chunk)
            
            self.chunks = all_chunks
        except:
            self.chunks = []
    
    def build_vector_index(self):
        if not self.chunks:
            return
        
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype='float32'))
        
        self.metadata = self.chunks
    
    def retrieve_context(self, query, top_k=5):
        if not self.index:
            return []
        
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding, dtype='float32'), 
            k=min(top_k, self.index.ntotal)
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx].copy()
                chunk['similarity_score'] = float(distances[0][i])
                results.append(chunk)
        
        return results
    
    def format_context(self, retrieved_chunks):
        if not retrieved_chunks:
            return "No relevant information found."
        
        context = ""
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            context += f"Article {i}:\n"
            context += f"Headline: {chunk['heading']}\n"
            context += f"Content: {chunk['content']}\n"
            if chunk.get('url'):
                context += f"Source: {chunk['url']}\n"
            context += "\n"
        
        return context.strip()
    
    def create_rag_prompt(self, user_query, context):
        prompt = f"""You are a fun, sassy celebrity gossip assistant called "Tea Bot". Your job is to answer questions about celebrities using the provided context from TMZ articles.

CONTEXT FROM TMZ ARTICLES:
{context}

USER QUESTION: {user_query}

INSTRUCTIONS:
- Use ONLY the information provided in the context above
- Be conversational, fun, and slightly sassy (use emojis like ☕ 💅 ✨)
- If the context doesn't contain relevant information, say you don't have tea on that topic
- Include source references when mentioning specific details
- Keep responses engaging but factual based on the context
- Start responses with phrases like "Here's the tea ☕" or "Spilling the gossip ✨"

RESPONSE:"""
        
        return prompt
    
    def call_mistral_api(self, prompt):
        if not self.mistral_api_key or self.mistral_api_key == "your_mistral_api_key_here":
            return "☕ Hey bestie! The tea machine isn't properly connected to the Mistral API yet. Please configure the backend API key for the full gossip experience! 💅"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        payload = {
            "model": "mistral-small",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.mistral_api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except requests.exceptions.RequestException as e:
            return f"Oops! The tea machine had a hiccup ☕ Technical issue: {str(e)}"
        except Exception as e:
            return f"Something went wrong while brewing your tea! ☕ Error: {str(e)}"
    
    def chat(self, user_input):
        user_input = user_input.strip()
        
        if not user_input:
            return "Hun, you gotta ask me something! 💅"
        
        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'what\'s up', 'sup']
        if any(greeting in user_input.lower() for greeting in greetings):
            return "Hey hun! ☕ Ready to spill some celebrity tea? What's the gossip you're curious about?"
        
        # Step 1: Retrieve relevant context from vector database
        retrieved_chunks = self.retrieve_context(user_input, top_k=5)
        
        # Step 2: Format context for LLM
        context = self.format_context(retrieved_chunks)
        
        # Step 3: Create RAG prompt
        rag_prompt = self.create_rag_prompt(user_input, context)
        
        # Step 4: Generate response using Mistral LLM
        llm_response = self.call_mistral_api(rag_prompt)
        
        return llm_response
    
    def get_stats(self):
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'mistral_configured': self.mistral_api_key is not None and self.mistral_api_key != "your_mistral_api_key_here",
            'ready': self.index is not None and len(self.chunks) > 0
        }
    
    def is_ready(self):
        return self.index is not None and len(self.chunks) > 0

# Streamlit App Configuration
st.set_page_config(
    page_title="Spill The Tea ☕",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #ff6b9d, #c44569);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .chat-container {
        border: 2px solid #ff6b9d;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        background: linear-gradient(135deg, #ffeef7, #fff);
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background-color: #ff6b9d;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 5px 15px;
        margin: 5px 0;
        text-align: right;
    }
    
    .bot-message {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 5px;
        margin: 5px 0;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .history-item {
        background: rgba(255, 107, 157, 0.1);
        border: 1px solid #ff6b9d;
        border-radius: 10px;
        padding: 8px 12px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: rgba(255, 107, 157, 0.2);
        transform: translateX(5px);
    }
    
    .chat-session {
        border: 2px solid #ff69b4;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #ffeef7, #fff);
    }
    
    .session-title {
        font-weight: bold;
        color: #c44569;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot' not in st.session_state:
    # Auto-initialize bot at startup
    with st.spinner("Loading Tea Bot..."):
        st.session_state.bot = TMZRAGBot()
        st.session_state.bot_ready = st.session_state.bot.is_ready()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = []

if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = 0

# Helper functions for chat history management
def create_new_session():
    session_id = len(st.session_state.chat_sessions)
    new_session = {
        'id': session_id,
        'title': f"Chat Session {session_id + 1}",
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'messages': [],
        'preview': "New conversation..."
    }
    st.session_state.chat_sessions.append(new_session)
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = []

def save_current_session():
    if st.session_state.chat_history:
        if st.session_state.current_session_id < len(st.session_state.chat_sessions):
            # Update existing session
            st.session_state.chat_sessions[st.session_state.current_session_id]['messages'] = st.session_state.chat_history.copy()
            # Update preview with first user message
            user_messages = [msg for msg in st.session_state.chat_history if msg['role'] == 'user']
            if user_messages:
                preview = user_messages[0]['content'][:50] + "..." if len(user_messages[0]['content']) > 50 else user_messages[0]['content']
                st.session_state.chat_sessions[st.session_state.current_session_id]['preview'] = preview

def load_session(session_id):
    save_current_session()  # Save current before switching
    st.session_state.current_session_id = session_id
    if session_id < len(st.session_state.chat_sessions):
        st.session_state.chat_history = st.session_state.chat_sessions[session_id]['messages'].copy()
    else:
        st.session_state.chat_history = []

# Main header
st.markdown('<h1 class="main-header"> Spill The Tea ☕</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Your AI-powered celebrity gossip assistant with Mistral LLM!</p>', unsafe_allow_html=True)

# Sidebar for chat history and controls
with st.sidebar:
    st.header("☕ Tea Bot Controls")
    
    # Bot status
    if st.session_state.bot_ready:
        stats = st.session_state.bot.get_stats()
        mistral_status = "✅ Connected" if stats['mistral_configured'] else "⚠️ Not configured"
        st.markdown(f"""
        <div class="stats-card">
            <h4> Ready to spilll the tea!</h4>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stats-card">
            <h4>❌ Bot Not Ready</h4>
            <p>Check tmz_data.json file</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat session management
    st.subheader("💬 Chat History")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat", use_container_width=True):
            save_current_session()
            create_new_session()
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_sessions = []
            st.session_state.chat_history = []
            st.session_state.current_session_id = 0
            st.rerun()
    
    # Display chat sessions
    if st.session_state.chat_sessions:
        st.markdown("**Previous Conversations:**")
        for i, session in enumerate(reversed(st.session_state.chat_sessions)):
            session_idx = len(st.session_state.chat_sessions) - 1 - i
            is_current = session_idx == st.session_state.current_session_id
            
            # Session container
            session_style = "border: 3px solid #ff69b4;" if is_current else "border: 1px solid #ff69b4;"
            
            with st.container():
                st.markdown(f"""
                <div class="chat-session" style="{session_style}">
                    <div class="session-title">Chat {session['id'] + 1}</div>
                    <div style="font-size: 0.8em; color: #666;">{session['timestamp']}</div>
                    <div style="font-size: 0.9em; margin-top: 5px;">{session['preview']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Load Chat {session['id'] + 1}", key=f"load_{session_idx}", use_container_width=True):
                    load_session(session_idx)
                    st.rerun()
    else:
        st.info("No chat history yet. Start a conversation!")
    
    # Quick actions
    st.subheader("⚡ Quick Topics")
    sample_questions = [
        "Taylor Swift latest news",
        "Celebrity wedding gossip",
        "Hollywood breakup drama",
        "Red carpet fashion",
        "Celebrity baby news"
    ]
    
    for question in sample_questions:
        if st.button(f"💅 {question}", key=f"quick_{question}", use_container_width=True):
            if st.session_state.bot_ready:
                # Create new session if current is empty, otherwise continue
                if not st.session_state.chat_history:
                    create_new_session()
                
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = st.session_state.bot.chat(question)
                st.session_state.chat_history.append({"role": "bot", "content": response})
                save_current_session()
                st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Chat with Spill the Tea BOT")
    
    # Show RAG status
    if st.session_state.bot_ready:
        stats = st.session_state.bot.get_stats()
        if stats['mistral_configured']:
            st.success("🤖 **BOT Active**")
        else:
            st.warning("⚠️ **Limited Mode**: Configure Mistral API key in backend for full experience")
    
    # Chat container
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>Tea Bot ☕:</strong><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Welcome message
            st.markdown(f"""
            <div class="bot-message">
                <strong>Tea Bot ☕:</strong><br>
                Hey gorgeous! 💅 Welcome to the most exclusive celebrity tea lounge! I'm your fabulous Tea Bot, ready to spill all the latest gossip using RAG technology. What celebrity drama are you dying to know about? ✨
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Spill the tea bestie... what's the gossip? 💅☕")
    
    if user_input:
        if not st.session_state.bot_ready:
            st.error("Tea Bot isn't ready! Check your data file 💅")
        else:
            # Create new session if this is the first message
            if not st.session_state.chat_history and not st.session_state.chat_sessions:
                create_new_session()
            
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get bot response
            with st.spinner("🔍 Searching for the juiciest gossip... 🤖 Brewing response..."):
                response = st.session_state.bot.chat(user_input)
                st.session_state.chat_history.append({"role": "bot", "content": response})
            
            # Save session
            save_current_session()
            st.rerun()

with col2:
    st.subheader("📊 Session Stats")
    
    # Current session stats
    if st.session_state.chat_history:
        user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        bot_messages = len([m for m in st.session_state.chat_history if m["role"] == "bot"])
        
        st.metric("💬 Your Messages", user_messages)
        st.metric("🤖 Bot Responses", bot_messages)
        
        # Current session info
        current_session = st.session_state.current_session_id + 1
        total_sessions = len(st.session_state.chat_sessions)
        st.metric("📝 Current Session", f"{current_session}")
        st.metric("🗂️ Total Sessions", total_sessions)
    else:
        st.info("Start chatting to see stats! 📊")
    
    # RAG explanation
    st.subheader("🔍 How it Works")
    st.markdown("""
    **Celebrity Tea Process:**
    1. 🔍 **Scraping**: Find relevant articles from popular portals
    2. 📝 **Context**: Add articles to your question
    3. 🤖 **Generate**: LLM creates sassy response
    4. ☕ **Serve**: Fresh celebrity tea delivered!
    """)
    
    # Tips
    st.subheader("💡 Pro Tips")
    st.markdown("""
    - **Be specific**: "Taylor Swift drama" works better
    - **Ask follow-ups**: "Tell me more about that!"
    - **Use names**: Celebrity names get best results
    - **Try different angles**: Mix up your questions
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 TMZ RAG Tea Bot - Powered by Mistral LLM + Vector Search ☕</p>
    <p><em>Backend-configured for seamless celebrity gossip experience!</em></p>
    <p><small>Your chat history is saved automatically ✨</small></p>
</div>
""", unsafe_allow_html=True)