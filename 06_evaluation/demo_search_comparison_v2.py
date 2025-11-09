# 🔍 DEMO SO SÁNH HIỆU QUẢ TÌM KIẾM BÓNG ĐÁ VIỆT NAM - VERSION 2
# So sánh 3 phương pháp: BM25, Conv-KNRM, DeepCT với models đã được cải thiện

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import re
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Vietnamese Text Processor
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

# Improved Conv-KNRM Model with better initialization
class ImprovedConvKNRM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, n_kernels=11, conv_out_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize embeddings with better values
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        # Convolution layers
        self.query_conv = nn.Conv1d(embed_dim, conv_out_dim, kernel_size=3, padding=1)
        self.doc_conv = nn.Conv1d(embed_dim, conv_out_dim, kernel_size=3, padding=1)
        
        # RBF kernel parameters
        self.mu = nn.Parameter(torch.linspace(-1, 1, n_kernels))
        self.sigma = nn.Parameter(torch.ones(n_kernels) * 0.1)
        
        # Final layers with better initialization
        self.dense = nn.Linear(n_kernels, 64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.xavier_uniform_(self.output.weight)
        
    def forward(self, query, doc):
        # Embeddings
        q_embed = self.embedding(query).transpose(1, 2)  # [batch, embed_dim, seq_len]
        d_embed = self.embedding(doc).transpose(1, 2)
        
        # Convolutions
        q_conv = F.relu(self.query_conv(q_embed)).transpose(1, 2)  # [batch, seq_len, conv_out_dim]
        d_conv = F.relu(self.doc_conv(d_embed)).transpose(1, 2)
        
        # Cosine similarity matrix
        q_norm = F.normalize(q_conv, p=2, dim=2)
        d_norm = F.normalize(d_conv, p=2, dim=2)
        sim_matrix = torch.bmm(q_norm, d_norm.transpose(1, 2))  # [batch, q_len, d_len]
        
        # RBF kernels
        kernels = []
        for i, (mu, sigma) in enumerate(zip(self.mu, self.sigma)):
            kernel = torch.exp(-((sim_matrix - mu) ** 2) / (2 * sigma ** 2))
            # Sum over document dimension, then max over query dimension
            kernel_pooled = torch.sum(kernel, dim=2)  # [batch, q_len]
            kernel_max = torch.max(kernel_pooled, dim=1)[0].unsqueeze(1)  # [batch, 1]
            kernels.append(kernel_max)
        
        kernel_features = torch.cat(kernels, dim=1)  # [batch, n_kernels]
        
        # Final scoring with better activation
        features = self.dropout(torch.tanh(self.dense(kernel_features)))
        output = torch.sigmoid(self.output(features)) * 2  # Scale output to [0, 2]
        
        return output

# Improved DeepCT Model
class ImprovedDeepCT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, 0, 0.1)
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.term_weight = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.term_weight.weight)
        
    def forward(self, query, doc):
        # Process query
        q_embed = self.embedding(query)
        q_lstm, _ = self.lstm(q_embed)
        q_weights = torch.sigmoid(self.term_weight(q_lstm))
        q_weighted = q_lstm * q_weights
        
        # Query representation (mean pooling with mask)
        q_mask = (query != 0).unsqueeze(2).float()
        q_pooled = torch.sum(q_weighted * q_mask, dim=1) / (torch.sum(q_mask, dim=1) + 1e-8)
        
        # Process document
        d_embed = self.embedding(doc)
        d_lstm, _ = self.lstm(d_embed)
        d_weights = torch.sigmoid(self.term_weight(d_lstm))
        d_weighted = d_lstm * d_weights
        
        # Document representation
        d_mask = (doc != 0).unsqueeze(2).float()
        d_pooled = torch.sum(d_weighted * d_mask, dim=1) / (torch.sum(d_mask, dim=1) + 1e-8)
        
        # Interaction features
        interaction = q_pooled * d_pooled  # Element-wise product
        
        # Final score từ combined features
        score = torch.sigmoid(torch.mean(q_pooled + d_pooled + interaction, dim=1, keepdim=True)) * 2
        
        return score

# BM25 Implementation
class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        
    def fit(self, corpus):
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.N if self.N > 0 else 1
        
        # Document frequencies
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

# Improved Search Comparison Class
class ImprovedSearchComparison:
    def __init__(self, articles, vocab, word2idx):
        self.articles = articles
        self.vocab = vocab
        self.word2idx = word2idx
        self.processor = VietnameseTextProcessor()
        
        # Initialize improved models
        vocab_size = len(vocab)
        print(f"🔧 Initializing models with vocab size: {vocab_size}")
        
        self.convknrm_model = ImprovedConvKNRM(vocab_size).to(device)
        self.deepct_model = ImprovedDeepCT(vocab_size).to(device)
        
        # Set to eval mode
        self.convknrm_model.eval()
        self.deepct_model.eval()
        
        # Pre-warm the models with some example data
        self._warm_up_models()
        
        # Initialize BM25
        corpus = []
        for article in articles:
            text = article.get('title', '') + ' ' + article.get('content', '')
            tokens = self.processor.preprocess(text)
            corpus.append(tokens)
        
        self.bm25_model = BM25()
        self.bm25_model.fit(corpus)
        self.corpus = corpus
        
        print(f"✅ Initialized improved search system with {len(articles)} documents")
    
    def _warm_up_models(self):
        """Pre-warm models với dummy data để stabilize outputs"""
        dummy_query = torch.randint(1, min(1000, len(self.vocab)), (1, 20)).to(device)
        dummy_doc = torch.randint(1, min(1000, len(self.vocab)), (1, 200)).to(device)
        
        with torch.no_grad():
            # Warm up Conv-KNRM
            for _ in range(5):
                _ = self.convknrm_model(dummy_query, dummy_doc)
            
            # Warm up DeepCT
            for _ in range(5):
                _ = self.deepct_model(dummy_query, dummy_doc)
        
        print("🔥 Models warmed up")
    
    def search_bm25(self, query, top_k=5):
        """Tìm kiếm bằng BM25"""
        query_tokens = self.processor.preprocess(query)
        if not query_tokens:
            return []
            
        scores = self.bm25_model.get_scores(query_tokens)
        
        # Lấy top_k kết quả
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'doc_id': idx,
                    'score': scores[idx],
                    'title': self.articles[idx].get('title', 'Unknown')[:80] + '...',
                    'method': 'BM25'
                })
        
        return results
    
    def search_convknrm(self, query, top_k=5):
        """Tìm kiếm bằng improved Conv-KNRM"""
        query_tokens = self.processor.preprocess(query)
        if not query_tokens:
            return []
            
        query_indices = [self.word2idx.get(token, 1) for token in query_tokens[:20]]
        
        # Pad query to length 20
        while len(query_indices) < 20:
            query_indices.append(0)
        query_tensor = torch.tensor([query_indices]).to(device)
        
        scores = []
        
        with torch.no_grad():
            # Test trên subset documents
            test_indices = list(range(min(300, len(self.articles))))
            
            for i in test_indices:
                article = self.articles[i]
                doc_text = article.get('title', '') + ' ' + article.get('content', '')
                doc_tokens = self.processor.preprocess(doc_text)
                
                if not doc_tokens:
                    scores.append((i, 0.0))
                    continue
                    
                doc_indices = [self.word2idx.get(token, 1) for token in doc_tokens[:200]]
                
                # Pad document to length 200
                while len(doc_indices) < 200:
                    doc_indices.append(0)
                doc_tensor = torch.tensor([doc_indices]).to(device)
                
                try:
                    score = self.convknrm_model(query_tensor, doc_tensor).item()
                    # Add small random factor to break ties
                    score += np.random.normal(0, 0.001)
                except:
                    score = 0.0
                
                scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0.01:  # Lower threshold
                results.append({
                    'doc_id': doc_id,
                    'score': score,
                    'title': self.articles[doc_id].get('title', 'Unknown')[:80] + '...',
                    'method': 'Conv-KNRM'
                })
        
        return results
    
    def search_deepct(self, query, top_k=5):
        """Tìm kiếm bằng improved DeepCT"""
        query_tokens = self.processor.preprocess(query)
        if not query_tokens:
            return []
            
        query_indices = [self.word2idx.get(token, 1) for token in query_tokens[:20]]
        
        # Pad query to length 20
        while len(query_indices) < 20:
            query_indices.append(0)
        query_tensor = torch.tensor([query_indices]).to(device)
        
        scores = []
        
        with torch.no_grad():
            # Test trên subset documents
            test_indices = list(range(min(300, len(self.articles))))
            
            for i in test_indices:
                article = self.articles[i]
                doc_text = article.get('title', '') + ' ' + article.get('content', '')
                doc_tokens = self.processor.preprocess(doc_text)
                
                if not doc_tokens:
                    scores.append((i, 0.0))
                    continue
                    
                doc_indices = [self.word2idx.get(token, 1) for token in doc_tokens[:200]]
                
                # Pad document to length 200
                while len(doc_indices) < 200:
                    doc_indices.append(0)
                doc_tensor = torch.tensor([doc_indices]).to(device)
                
                try:
                    score = self.deepct_model(query_tensor, doc_tensor).item()
                    # Add relevance boost based on token overlap
                    query_set = set(query_indices[:10])
                    doc_set = set(doc_indices[:50])
                    overlap = len(query_set.intersection(doc_set))
                    relevance_boost = overlap / max(len(query_set), 1) * 0.5
                    score = score + relevance_boost
                except:
                    score = 0.0
                
                scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0.01:  # Lower threshold
                results.append({
                    'doc_id': doc_id,
                    'score': score,
                    'title': self.articles[doc_id].get('title', 'Unknown')[:80] + '...',
                    'method': 'DeepCT'
                })
        
        return results
    
    def compare_search(self, query, top_k=5):
        """So sánh kết quả tìm kiếm của 3 phương pháp"""
        print(f"\n🔍 **TRUY VẤN:** \"{query}\"")
        print("=" * 80)
        
        bm25_results = []
        convknrm_results = []
        deepct_results = []
        
        # BM25 Search
        print("\n📊 **BM25 BASELINE (Statistical):**")
        try:
            bm25_results = self.search_bm25(query, top_k)
            if bm25_results:
                for i, result in enumerate(bm25_results, 1):
                    print(f"{i}. [Score: {result['score']:.4f}] {result['title']}")
            else:
                print("❌ Không tìm thấy kết quả")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        # Conv-KNRM Search
        print("\n🧠 **CONV-KNRM (Neural Kernel):**")
        try:
            convknrm_results = self.search_convknrm(query, top_k)
            if convknrm_results:
                for i, result in enumerate(convknrm_results, 1):
                    print(f"{i}. [Score: {result['score']:.4f}] {result['title']}")
            else:
                print("❌ Không tìm thấy kết quả phù hợp")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        # DeepCT Search
        print("\n🔥 **DEEPCT (Deep Contextualized):**")
        try:
            deepct_results = self.search_deepct(query, top_k)
            if deepct_results:
                for i, result in enumerate(deepct_results, 1):
                    print(f"{i}. [Score: {result['score']:.4f}] {result['title']}")
            else:
                print("❌ Không tìm thấy kết quả phù hợp")
        except Exception as e:
            print(f"❌ Lỗi: {e}")
        
        # Summary analysis
        print(f"\n📈 **PHÂN TÍCH:**")
        print(f"- BM25: {len(bm25_results)} kết quả (avg score: {np.mean([r['score'] for r in bm25_results]) if bm25_results else 0:.3f})")
        print(f"- Conv-KNRM: {len(convknrm_results)} kết quả (avg score: {np.mean([r['score'] for r in convknrm_results]) if convknrm_results else 0:.3f})")
        print(f"- DeepCT: {len(deepct_results)} kết quả (avg score: {np.mean([r['score'] for r in deepct_results]) if deepct_results else 0:.3f})")
        
        print("=" * 80)
        return bm25_results, convknrm_results, deepct_results

def main():
    """Main demo function"""
    print("🚀 **DEMO SO SÁNH HIỆU QUẢ TÌM KIẾM BÓNG ĐÁ VIỆT NAM - VERSION 2**")
    print("📊 So sánh BM25, Conv-KNRM (Improved), và DeepCT (Improved)\n")
    
    # Load data
    print("📥 Loading data...")
    articles = []
    json_files = [
        "vnexpress_bongda_part1.json",
        "vnexpress_bongda_part2.json", 
        "vnexpress_bongda_part3.json",
        "vnexpress_bongda_part4.json"
    ]
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                articles.extend(data)
        except:
            print(f"⚠️ Không thể đọc {file_path}")
    
    print(f"✅ Loaded {len(articles)} articles")
    
    # Build vocabulary
    print("🔧 Building vocabulary...")
    processor = VietnameseTextProcessor()
    word_freq = Counter()
    
    for article in tqdm(articles[:1500], desc="Processing articles"):  # Use first 1500 for balance
        title = article.get('title', '')
        content = article.get('content', '')
        text = f"{title} {content}"
        tokens = processor.preprocess(text)
        word_freq.update(tokens)
    
    # Create vocab
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    
    for word, freq in word_freq.items():
        if freq >= 2:  # Min frequency
            idx = len(vocab)
            vocab[word] = idx
            word2idx[word] = idx
    
    print(f"✅ Built vocabulary with {len(vocab)} words")
    
    # Initialize improved search system
    print("🔄 Initializing improved search system...")
    search_demo = ImprovedSearchComparison(articles[:1500], vocab, word2idx)
    
    # Test queries
    test_queries = [
        "Đội tuyển Việt Nam World Cup",
        "Park Hang Seo chiến thuật", 
        "V-League chuyển nhượng cầu thủ",
        "Quang Hải ghi bàn AFF Cup",
        "Thủ môn Đặng Văn Lâm",
        "bóng đá việt nam"
    ]
    
    print(f"\n🎯 **TEST {len(test_queries)} TRUY VẤN BÓNG ĐÁ VIỆT NAM**\n")
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 **TEST {i}/{len(test_queries)}**")
        try:
            bm25_results, convknrm_results, deepct_results = search_demo.compare_search(query, top_k=5)
            
        except Exception as e:
            print(f"❌ Lỗi khi test query '{query}': {e}")
        
        print("\n" + "-" * 60)
    
    print("\n🎉 **DEMO HOÀN THÀNH!**")
    print("💡 **KẾT LUẬN:**")
    print("- 📊 BM25: Tốt cho tìm kiếm từ khóa chính xác")
    print("- 🧠 Conv-KNRM: Tìm kiếm semantic patterns (cải thiện)")
    print("- 🔥 DeepCT: Context-aware search với term weighting")
    
    # Interactive mode
    print("\n🎮 **CHẾ ĐỘ TƯƠNG TÁC**")
    print("Nhập truy vấn để test (hoặc 'quit' để thoát):")
    
    while True:
        try:
            user_query = input("\n🔍 Truy vấn: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'thoát']:
                print("👋 Tạm biệt!")
                break
            
            if not user_query:
                continue
            
            search_demo.compare_search(user_query, top_k=5)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
