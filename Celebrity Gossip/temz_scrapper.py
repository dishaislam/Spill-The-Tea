import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
import re

class TMZChunkingScraper:
    def __init__(self, max_pages=6, delay=1):
        self.raw_articles = []
        self.chunked_articles = []
        self.max_pages = max_pages
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_all_pages(self):
        self.raw_articles = []
        
        # First page
        self._scrape_page(0, 'https://www.tmz.com/')
        
        # Subsequent pages
        for i in range(1, self.max_pages):
            url = f'https://www.tmz.com/?page={i}&cursor=eyJvcmRlcl9kYXRlX2JlZm9yZSI6MTc1NTAzNjAxMH0&locale=en'
            self._scrape_page(i, url)
            time.sleep(self.delay)
        
        return self.raw_articles
    
    def _scrape_page(self, page_num, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            for main_idx, main in enumerate(soup.find_all('main')):
                articles = main.find_all('article')
                if not articles:
                    articles = [main]
                
                for idx, article_elem in enumerate(articles):
                    article_data = self._process_article(article_elem, page_num, url, idx)
                    if article_data:
                        self.raw_articles.append(article_data)
        except:
            pass
    
    def _process_article(self, article_elem, page_num, url, article_idx):
        article_id = article_elem.get('id', f'article_{page_num}_{article_idx}')
        structured_content = self._extract_content(article_elem)
        
        if not structured_content:
            return None
        
        return {
            'article_id': article_id,
            'page': page_num,
            'url': url,
            'structured_content': structured_content,
            'scraped_at': datetime.now().isoformat()
        }
    
    def _extract_content(self, element):
        structured_content = []
        elements = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div'])
        
        current_heading = None
        current_content = []
        heading_level = 0
        
        for elem in elements:
            tag = elem.name.lower()
            text = elem.get_text(strip=True)
            
            if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and len(text) > 3:
                if current_heading:
                    chunk = self._create_chunk(current_heading, current_content, heading_level)
                    if chunk:
                        structured_content.append(chunk)
                
                current_heading = text
                heading_level = int(tag[1])
                current_content = []
            
            elif tag in ['p', 'div'] and len(text) > 10:
                current_content.append(text)
        
        if current_heading:
            chunk = self._create_chunk(current_heading, current_content, heading_level)
            if chunk:
                structured_content.append(chunk)
        
        if not structured_content:
            structured_content = self._extract_fallback_content(element)
        
        return structured_content
    
    def _create_chunk(self, heading, content_list, heading_level):
        content_text = ' '.join(content_list)
        combined_text = f"Heading: {heading}"
        if content_text:
            combined_text += f" Content: {content_text}"
        
        return {
            'heading': heading,
            'heading_level': heading_level,
            'content': content_text,
            'combined_text': combined_text,
            'has_content': len(content_text) > 0
        }
    
    def _extract_fallback_content(self, element):
        all_text = element.get_text(strip=True)
        if len(all_text) < 50:
            return []
        
        sentences = [s.strip() for s in re.split(r'[.!?]+\s+', all_text) if len(s.strip()) > 10]
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(' '.join(current_chunk)) > 200:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'heading': f"Content Section {len(chunks) + 1}",
                    'heading_level': 0,
                    'content': chunk_text,
                    'combined_text': f"Content: {chunk_text}",
                    'has_content': True
                })
                current_chunk = []
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 50:
                chunks.append({
                    'heading': f"Content Section {len(chunks) + 1}",
                    'heading_level': 0,
                    'content': chunk_text,
                    'combined_text': f"Content: {chunk_text}",
                    'has_content': True
                })
        
        return chunks
    
    def create_chunks(self):
        self.chunked_articles = []
        chunk_id = 0
        
        for article in self.raw_articles:
            for chunk_data in article['structured_content']:
                self.chunked_articles.append({
                    'chunk_id': chunk_id,
                    'article_id': article['article_id'],
                    'page': article['page'],
                    'url': article['url'],
                    'heading': chunk_data['heading'],
                    'content': chunk_data['content'],
                    'combined_text': chunk_data['combined_text'],
                    'has_content': chunk_data['has_content']
                })
                chunk_id += 1
        
        return self.chunked_articles
    
    def get_chunks_with_content(self):
        return [chunk for chunk in self.chunked_articles if chunk['has_content']]
    
    def get_vector_ready_chunks(self):
        return [
            {
                'id': chunk['chunk_id'],
                'text': chunk['combined_text'],
                'metadata': {
                    'heading': chunk['heading'],
                    'article_id': chunk['article_id'],
                    'page': chunk['page'],
                    'url': chunk['url']
                }
            }
            for chunk in self.get_chunks_with_content()
        ]
    
    def save_data(self, filename="tmz_data.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'raw_articles': self.raw_articles,
                'chunked_articles': self.chunked_articles
            }, f, indent=2, ensure_ascii=False)
    
    def load_data(self, filename="tmz_data.json"):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.raw_articles = data.get('raw_articles', [])
            self.chunked_articles = data.get('chunked_articles', [])
            return True
        except FileNotFoundError:
            return False

