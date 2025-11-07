"""
🔍 Vietnamese Football Search API
Backend for Google-like search interface
Tech: FastAPI + Neural Ranking Models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
from typing import List, Optional
from collections import Counter
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Vietnamese Football Search API", version="1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Vietnamese Text Processor
# ============================
class VietnameseTextProcessor:
    def __init__(self):
        self.stop_words = set([
            'và', 'của', 'trong', 'với', 'là', 'có', 'được', 'cho', 'từ', 'một', 'các',
            'để', 'không', 'sẽ', 'đã', 'về', 'hay', 'theo', 'như', 'cũng', 'này', 'đó',
            'khi', 'những', 'tại', 'sau', 'bị', 'giữa', 'trên', 'dưới', 'ngoài',
            'thì', 'nhưng', 'mà', 'hoặc', 'nếu', 'vì', 'do', 'nên', 'rồi', 'còn', 'đều',
            'chỉ', 'việc', 'người', 'lại', 'đây', 'đấy', 'ở', 'ra', 'vào', 'lên', 'xuống'
        ])
    
    def clean_text(self, text):
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ]', ' ', text)
        text = text.lower().strip()
        return text
    
    def tokenize(self, text):
        return text.split()
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 1]
    
    def preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        filtered = self.remove_stopwords(tokens)
        return filtered

# ============================
# BM25 Implementation
# ============================
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        
    def fit(self, corpus):
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N if self.N > 0 else 1
        
        self.df = {}
        for doc in corpus:
            for token in set(doc):
                self.df[token] = self.df.get(token, 0) + 1
                
    def get_scores(self, query):
        scores = []
        for doc in self.corpus:
            score = 0
            dl = len(doc)
            
            for token in query:
                if token in self.df:
                    tf = doc.count(token)
                    idf = np.log((self.N - self.df[token] + 0.5) / (self.df[token] + 0.5))
                    score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            
            scores.append(max(0, score))
        return np.array(scores)

# ============================
# Neural Models
# ============================
class ImprovedConvKNRM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_kernels=11):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        self.query_conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.doc_conv = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        
        self.mu = nn.Parameter(torch.linspace(-1, 1, n_kernels))
        self.sigma = nn.Parameter(torch.ones(n_kernels) * 0.1)
        
        self.dense = nn.Linear(n_kernels, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, query, doc):
        q_embed = self.embedding(query).transpose(1, 2)
        d_embed = self.embedding(doc).transpose(1, 2)
        
        q_conv = F.relu(self.query_conv(q_embed)).transpose(1, 2)
        d_conv = F.relu(self.doc_conv(d_embed)).transpose(1, 2)
        
        q_norm = F.normalize(q_conv, p=2, dim=2)
        d_norm = F.normalize(d_conv, p=2, dim=2)
        sim_matrix = torch.bmm(q_norm, d_norm.transpose(1, 2))
        
        kernels = []
        for i, (mu, sigma) in enumerate(zip(self.mu, self.sigma)):
            kernel = torch.exp(-((sim_matrix - mu) ** 2) / (2 * sigma ** 2))
            kernel_pooled = torch.sum(kernel, dim=2)
            kernel_max = torch.max(kernel_pooled, dim=1)[0].unsqueeze(1)
            kernels.append(kernel_max)
        
        kernel_features = torch.cat(kernels, dim=1)
        features = self.dropout(torch.tanh(self.dense(kernel_features)))
        output = torch.sigmoid(self.output(features)) * 2
        
        return output

class ImprovedDeepCT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.term_weight = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, query, doc):
        q_embed = self.embedding(query)
        q_lstm, _ = self.lstm(q_embed)
        q_weights = torch.sigmoid(self.term_weight(q_lstm))
        q_weighted = q_lstm * q_weights
        
        q_mask = (query != 0).unsqueeze(2).float()
        q_pooled = torch.sum(q_weighted * q_mask, dim=1) / (torch.sum(q_mask, dim=1) + 1e-8)
        
        d_embed = self.embedding(doc)
        d_lstm, _ = self.lstm(d_embed)
        d_weights = torch.sigmoid(self.term_weight(d_lstm))
        d_weighted = d_lstm * d_weights
        
        d_mask = (doc != 0).unsqueeze(2).float()
        d_pooled = torch.sum(d_weighted * d_mask, dim=1) / (torch.sum(d_mask, dim=1) + 1e-8)
        
        interaction = q_pooled * d_pooled
        score = torch.sigmoid(torch.mean(q_pooled + d_pooled + interaction, dim=1, keepdim=True)) * 2
        
        return score

# ============================
# Search Engine
# ============================
class SearchEngine:
    def __init__(self):
        self.processor = VietnameseTextProcessor()
        self.articles = []
        self.vocab = {}
        self.word2idx = {}
        self.bm25 = None
        self.convknrm = None
        self.deepct = None
        self.corpus = []
        
    def load_data(self, data_paths):
        """Load articles from JSON files"""
        for path in data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.articles.extend(data)
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        print(f"✅ Loaded {len(self.articles)} articles")
        
    def build_vocab(self):
        """Build vocabulary"""
        word_freq = Counter()
        
        for article in self.articles:
            text = f"{article.get('title', '')} {article.get('content', '')}"
            tokens = self.processor.preprocess(text)
            word_freq.update(tokens)
            self.corpus.append(tokens)
        
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        
        for word, freq in word_freq.items():
            if freq >= 2:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.word2idx[word] = idx
        
        print(f"✅ Built vocabulary: {len(self.vocab)} words")
        
    def initialize_models(self):
        """Initialize search models"""
        # BM25
        self.bm25 = BM25()
        self.bm25.fit(self.corpus)
        
        # Neural models
        vocab_size = len(self.vocab)
        self.convknrm = ImprovedConvKNRM(vocab_size)
        self.deepct = ImprovedDeepCT(vocab_size)
        
        self.convknrm.eval()
        self.deepct.eval()
        
        print("✅ Models initialized")
    
    def search(self, query: str, method: str = "all", top_k: int = 10):
        """Search with specified method"""
        query_tokens = self.processor.preprocess(query)
        
        if not query_tokens:
            return []
        
        results = []
        
        if method in ["bm25", "all"]:
            results.extend(self._search_bm25(query_tokens, top_k))
        
        if method in ["convknrm", "all"]:
            results.extend(self._search_convknrm(query_tokens, top_k))
            
        if method in ["deepct", "all"]:
            results.extend(self._search_deepct(query_tokens, top_k))
          # Remove duplicates and sort by score
        seen = set()
        unique_results = []
        for r in results:
            if r['doc_id'] not in seen:
                seen.add(r['doc_id'])
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        return unique_results[:top_k]
    
    def _search_bm25(self, query_tokens, top_k):
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                article = self.articles[int(idx)]
                results.append({
                    'doc_id': int(idx),
                    'score': float(scores[idx]),
                    'method': 'BM25',
                    'title': article.get('title', ''),
                    'content': article.get('content', '')[:200] + '...',
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'date': article.get('date', '')                })
        
        return results
    
    def _search_convknrm(self, query_tokens, top_k):
        query_indices = [self.word2idx.get(token, 1) for token in query_tokens[:20]]
        while len(query_indices) < 20:
            query_indices.append(0)
        query_tensor = torch.tensor([query_indices])
        
        scores = []
        
        with torch.no_grad():
            for i in range(min(300, len(self.articles))):
                doc_tokens = self.corpus[i]
                doc_indices = [self.word2idx.get(token, 1) for token in doc_tokens[:200]]
                while len(doc_indices) < 200:
                    doc_indices.append(0)
                doc_tensor = torch.tensor([doc_indices])
                
                try:
                    score = self.convknrm(query_tensor, doc_tensor).item()
                except:
                    score = 0.0
                
                scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0.01:
                article = self.articles[int(doc_id)]
                results.append({
                    'doc_id': int(doc_id),
                    'score': float(score),
                    'method': 'Conv-KNRM',
                    'title': article.get('title', ''),
                    'content': article.get('content', '')[:200] + '...',
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'date': article.get('date', '')
                })
        
        return results
    
    def _search_deepct(self, query_tokens, top_k):
        query_indices = [self.word2idx.get(token, 1) for token in query_tokens[:20]]
        while len(query_indices) < 20:
            query_indices.append(0)
        query_tensor = torch.tensor([query_indices])
        
        scores = []
        
        with torch.no_grad():
            for i in range(min(300, len(self.articles))):
                doc_tokens = self.corpus[i]
                doc_indices = [self.word2idx.get(token, 1) for token in doc_tokens[:200]]
                while len(doc_indices) < 200:
                    doc_indices.append(0)
                doc_tensor = torch.tensor([doc_indices])
                
                try:
                    score = self.deepct(query_tensor, doc_tensor).item()
                      # Relevance boost
                    query_set = set(query_indices[:10])
                    doc_set = set(doc_indices[:50])
                    overlap = len(query_set.intersection(doc_set))
                    relevance_boost = overlap / max(len(query_set), 1) * 0.5
                    score = score + relevance_boost
                except:
                    score = 0.0
                
                scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0.01:
                article = self.articles[int(doc_id)]
                results.append({
                    'doc_id': int(doc_id),
                    'score': float(score),
                    'method': 'DeepCT',
                    'title': article.get('title', ''),
                    'content': article.get('content', '')[:200] + '...',
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'date': article.get('date', '')
                })
        
        return results
# ============================
# Initialize Search Engine
# ============================
search_engine = SearchEngine()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    print("🚀 Starting Vietnamese Football Search API...")
    
    # Load data
    data_paths = [
        "../../vnexpress_bongda_part1.json",
        "../../vnexpress_bongda_part2.json",
        "../../vnexpress_bongda_part3.json",
        "../../vnexpress_bongda_part4.json"
    ]
    
    search_engine.load_data(data_paths)
    search_engine.build_vocab()
    search_engine.initialize_models()
    
    print("✅ Search engine ready!")

# ============================
# API Models
# ============================
class SearchRequest(BaseModel):
    query: str
    method: str = "all"  # "all", "bm25", "convknrm", "deepct"
    top_k: int = 10

class SearchResponse(BaseModel):
    query: str
    method: str
    total_results: int
    results: List[dict]
    took_ms: float

# ============================
# API Endpoints
# ============================
@app.get("/")
async def root():
    return {
        "message": "Vietnamese Football Search API",
        "version": "1.0",
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "articles": len(search_engine.articles),
        "vocab_size": len(search_engine.vocab)
    }

@app.get("/stats")
async def stats():
    return {
        "total_articles": len(search_engine.articles),
        "vocab_size": len(search_engine.vocab),
        "methods": ["BM25", "Conv-KNRM", "DeepCT"]
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint"""
    import time
    start = time.time()
    
    try:
        results = search_engine.search(
            query=request.query,
            method=request.method,
            top_k=request.top_k
        )
        
        took_ms = (time.time() - start) * 1000
        
        return SearchResponse(
            query=request.query,
            method=request.method,
            total_results=len(results),
            results=results,
            took_ms=took_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suggestions")
async def suggestions(q: str):
    """Get search suggestions"""
    if not q:
        return {"suggestions": []}
    
    # Simple suggestions based on popular queries
    popular_queries = [
        "Đội tuyển Việt Nam",
        "Park Hang Seo",
        "Quang Hải",
        "V-League",
        "Công Phượng",
        "World Cup",
        "AFF Cup",
        "Thái Lan",
        "bóng đá việt nam"
    ]
    
    suggestions = [query for query in popular_queries if q.lower() in query.lower()]
    
    return {"suggestions": suggestions[:5]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
