import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import random

class TMZRAGBot:
    def __init__(self, data_file="tmz_data.json", model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.index = None
        self.metadata = []
        
        # Mistral API configuration
        self.mistral_api_url = "https://api.mistral.ai/v1/chat/completions"
        self.mistral_api_key = None  # Set this with your API key
        
        self.load_data(data_file)
        self.build_vector_index()
    
    def set_mistral_api_key(self, api_key):
        """Set Mistral API key"""
        self.mistral_api_key = api_key
    
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
        """Retrieve relevant chunks from vector database"""
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
        """Format retrieved chunks into context for LLM"""
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
        """Create a proper RAG prompt for Mistral"""
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
        """Call Mistral API for text generation"""
        if not self.mistral_api_key:
            return "❌ Mistral API key not set! Please configure your API key first."
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        payload = {
            "model": "mistral-small",  # or "mistral-medium", "mistral-large"
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
            return f"❌ Error calling Mistral API: {str(e)}"
        except Exception as e:
            return f"❌ Error processing response: {str(e)}"
    
    def chat(self, user_input):
        """Main RAG chat method"""
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
    
    def chat_with_local_fallback(self, user_input):
        """RAG chat with local fallback if API fails"""
        user_input = user_input.strip()
        
        if not user_input:
            return "Hun, you gotta ask me something! 💅"
        
        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'what\'s up', 'sup']
        if any(greeting in user_input.lower() for greeting in greetings):
            return "Hey hun! ☕ Ready to spill some celebrity tea? What's the gossip you're curious about?"
        
        # Retrieve context
        retrieved_chunks = self.retrieve_context(user_input, top_k=5)
        
        if not retrieved_chunks:
            return "Oop, I don't have any tea on that topic right now! ☕ Try asking about something else, bestie!"
        
        # Try Mistral API first
        if self.mistral_api_key:
            context = self.format_context(retrieved_chunks)
            rag_prompt = self.create_rag_prompt(user_input, context)
            llm_response = self.call_mistral_api(rag_prompt)
            
            # Check if API call was successful
            if not llm_response.startswith("❌"):
                return llm_response
        
        # Fallback to simple template-based response
        return self.generate_fallback_response(retrieved_chunks)
    
    def generate_fallback_response(self, retrieved_chunks):
        """Fallback response generation if LLM API fails"""
        if not retrieved_chunks:
            return "Oop, I don't have any tea on that topic right now! ☕"
        
        response = "Here's what I found in the celebrity tea files ☕✨\n\n"
        
        for i, chunk in enumerate(retrieved_chunks[:3], 1):
            response += f"**{i}. {chunk['heading']}** 💅\n"
            
            content = chunk['content']
            if len(content) > 200:
                content = content[:200] + "..."
            
            response += f"{content}\n"
            
            if chunk.get('url'):
                response += f"🔗 [Read more]({chunk['url']})\n"
            
            response += "\n"
        
        footers = [
            "That's the tea I've got! ☕ Need more gossip?",
            "Spilled from the archives! ☕ What else you want to know?",
            "Tea served! ☕✨ Anything else on your mind?"
        ]
        
        response += random.choice(footers)
        return response
    
    def get_stats(self):
        return {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'mistral_configured': self.mistral_api_key is not None,
            'ready': self.index is not None and len(self.chunks) > 0
        }
    
    def is_ready(self):
        return self.index is not None and len(self.chunks) > 0

# Alternative implementation using Hugging Face Transformers (local Mistral)
class TMZRAGBotLocal:
    def __init__(self, data_file="tmz_data.json", model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.chunks = []
        self.index = None
        self.metadata = []
        self.llm_pipeline = None
        
        self.load_data(data_file)
        self.build_vector_index()
        self.load_local_llm()
    
    def load_local_llm(self):
        """Load local Mistral model using transformers"""
        try:
            from transformers import pipeline
            # Using a smaller model that works locally
            self.llm_pipeline = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",  # Smaller alternative
                tokenizer="microsoft/DialoGPT-medium",
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        except:
            self.llm_pipeline = None
    
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
        embeddings = self.embedding_model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings, dtype='float32'))
        
        self.metadata = self.chunks
    
    def retrieve_context(self, query, top_k=3):
        if not self.index:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(
            np.array(query_embedding, dtype='float32'), 
            k=min(top_k, self.index.ntotal)
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
        
        return results
    
    def chat(self, user_input):
        user_input = user_input.strip()
        
        if not user_input:
            return "Hun, you gotta ask me something! 💅"
        
        # Retrieve context
        retrieved_chunks = self.retrieve_context(user_input, top_k=3)
        
        if not retrieved_chunks:
            return "Oop, I don't have any tea on that topic right now! ☕"
        
        # Create context
        context = ""
        for chunk in retrieved_chunks:
            context += f"{chunk['heading']}: {chunk['content'][:200]}... "
        
        # Generate response
        if self.llm_pipeline:
            prompt = f"Context: {context}\nUser: {user_input}\nTea Bot:"
            response = self.llm_pipeline(prompt, max_length=200, num_return_sequences=1)
            return response[0]['generated_text'].split("Tea Bot:")[-1].strip()
        
        # Fallback
        return f"Here's the tea ☕ Based on what I found: {retrieved_chunks[0]['content'][:200]}..."
    
    def is_ready(self):
        return self.index is not None and len(self.chunks) > 0