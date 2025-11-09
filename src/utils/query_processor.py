"""
QUERY PROCESSING & ACCENT RESTORATION

Xử lý query từ user với các tính năng:
1. Vietnamese accent restoration (bong da → bóng đá)
2. Spell checking and correction
3. Query normalization
4. Synonym expansion
5. Query suggestion

Features:
- Automatic accent restoration using vocabulary
- Fuzzy matching for typos
- Smart query preprocessing
"""

import os
import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
from difflib import SequenceMatcher
import unicodedata

# Import từ modules
import sys
sys.path.append(os.path.dirname(__file__))


class VietnameseAccentRestorer:
    """
    Tự động restore dấu tiếng Việt
    
    Ví dụ:
    - "bong da" → "bóng đá"
    - "viet nam" → "việt nam"
    - "hlv" → "HLV" (giữ nguyên abbreviation)
    """
    
    def __init__(self, vocab_file: str = None):
        """
        Args:
            vocab_file: File chứa vocabulary (words có dấu)
        """
        self.vocab_file = vocab_file or "data/vocab/vocab.txt"
        self.vocabulary = set()  # Words có dấu
        self.accent_map = defaultdict(list)  # {word_no_accent: [word_with_accent1, word_with_accent2, ...]}
        self.word_freq = Counter()  # Frequency của mỗi word
        
        self.load_vocabulary()
        self.build_accent_map()
    
    def remove_accents(self, text: str) -> str:
        """
        Remove Vietnamese accents
        
        ví dụ: "bóng đá" → "bong da"
        """
        # Normalize Unicode
        text = unicodedata.normalize('NFD', text)
        # Remove combining characters
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        # Normalize lại
        text = unicodedata.normalize('NFC', text)
        
        # Replace đ/Đ
        text = text.replace('đ', 'd').replace('Đ', 'D')
        
        return text
    
    def load_vocabulary(self):
        """Load vocabulary từ file"""
        if not os.path.exists(self.vocab_file):
            print(f"⚠️ Vocabulary file not found: {self.vocab_file}")
            print("   Using fallback vocabulary...")
            # Fallback vocabulary cho football
            self.vocabulary = {
                'bóng_đá', 'việt_nam', 'thái_lan', 'huấn_luyện_viên',
                'cầu_thủ', 'trận_đấu', 'giải_đấu', 'vô_địch',
                'thắng', 'thua', 'hòa', 'bàn_thắng', 'penalty',
                'thủ_môn', 'hậu_vệ', 'tiền_đạo', 'tiền_vệ'
            }
            return
        
        try:
            with open(self.vocab_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.vocabulary.add(word)
                        self.word_freq[word] += 1
            
            print(f"✓ Loaded vocabulary: {len(self.vocabulary):,} words")
        except Exception as e:
            print(f"✗ Error loading vocabulary: {e}")
    
    def build_accent_map(self):
        """
        Xây dựng map: word không dấu → list words có dấu
        
        Ví dụ:
        {
            'bong_da': ['bóng_đá', 'bông_da'],
            'viet_nam': ['việt_nam', 'viết_nam']
        }
        """
        for word in self.vocabulary:
            word_no_accent = self.remove_accents(word.lower())
            self.accent_map[word_no_accent].append(word)
        
        print(f"✓ Built accent map: {len(self.accent_map):,} entries")
    
    def restore_word(self, word: str, context_words: List[str] = None) -> str:
        """
        Restore dấu cho một word
        
        Args:
            word: Word không dấu (ví dụ: "bong")
            context_words: Context words để chọn candidate tốt nhất
            
        Returns:
            Word có dấu (ví dụ: "bóng")
        """
        word_lower = word.lower()
        word_no_accent = self.remove_accents(word_lower)
        
        # Nếu word đã có trong vocabulary → return nguyên
        if word_lower in self.vocabulary:
            return word_lower
        
        # Tìm candidates trong accent_map
        candidates = self.accent_map.get(word_no_accent, [])
        
        if not candidates:
            # Không tìm thấy → return nguyên
            return word_lower
        
        if len(candidates) == 1:
            # Chỉ có 1 candidate → return luôn
            return candidates[0]
        
        # Nhiều candidates → chọn theo frequency hoặc context
        if context_words:
            # TODO: Implement context-based selection
            pass
        
        # Chọn candidate phổ biến nhất
        best_candidate = max(candidates, key=lambda w: self.word_freq.get(w, 0))
        return best_candidate
    
    def restore_text(self, text: str) -> str:
        """
        Restore dấu cho toàn bộ text
        
        Args:
            text: Text không dấu (ví dụ: "bong da viet nam")
            
        Returns:
            Text có dấu (ví dụ: "bóng đá việt nam")
        """
        # Tokenize
        words = text.lower().split()
        
        # Restore từng word
        restored_words = []
        for i, word in enumerate(words):
            # Lấy context (2 words trước + 2 words sau)
            context = words[max(0, i-2):i] + words[i+1:min(len(words), i+3)]
            restored = self.restore_word(word, context)
            restored_words.append(restored)
        
        return ' '.join(restored_words)
    
    def get_suggestions(self, word: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Gợi ý các từ tương tự (cho autocomplete)
        
        Returns:
            List[(word, similarity_score)]
        """
        word_no_accent = self.remove_accents(word.lower())
        
        # Tìm tất cả words có prefix giống
        suggestions = []
        
        for vocab_word in self.vocabulary:
            vocab_no_accent = self.remove_accents(vocab_word)
            
            # Check prefix
            if vocab_no_accent.startswith(word_no_accent):
                similarity = 1.0
                suggestions.append((vocab_word, similarity))
            else:
                # Check fuzzy match
                similarity = SequenceMatcher(None, word_no_accent, vocab_no_accent).ratio()
                if similarity > 0.7:
                    suggestions.append((vocab_word, similarity))
        
        # Sort by similarity + frequency
        suggestions.sort(key=lambda x: (x[1], self.word_freq.get(x[0], 0)), reverse=True)
        
        return suggestions[:max_suggestions]


class QueryProcessor:
    """
    Query Processing với accent restoration
    """
    
    def __init__(self, vocab_file: str = None):
        self.accent_restorer = VietnameseAccentRestorer(vocab_file)
        
        # Vietnamese stopwords (optional, để filter query)
        self.stopwords = {
            'của', 'và', 'có', 'được', 'trong', 'ở', 'tại',
            'với', 'để', 'cho', 'từ', 'về', 'theo', 'như'
        }
        
        # Common Vietnamese abbreviations
        self.abbreviations = {
            'hlv': 'huấn_luyện_viên',
            'slna': 'sông_lam_nghệ_an',
            'hagl': 'hoàng_anh_gia_lai',
            'vff': 'liên_đoàn_bóng_đá_việt_nam',
            'aff': 'asean_football_federation',
            'vl': 'vòng_loại'
        }
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query:
        1. Lowercase
        2. Remove extra spaces
        3. Remove special chars
        """
        # Lowercase
        query = query.lower().strip()
        
        # Remove special characters (keep spaces and underscore)
        query = re.sub(r'[^\w\s_]', ' ', query)
        
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        
        return query
    
    def expand_abbreviations(self, query: str) -> str:
        """
        Expand abbreviations
        
        Ví dụ: "hlv park" → "huấn luyện viên park"
        """
        words = query.split()
        expanded_words = []
        
        for word in words:
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def process_query(self, 
                     query: str, 
                     restore_accents: bool = True,
                     expand_abbr: bool = True,
                     remove_stopwords: bool = False) -> Dict:
        """
        Process query từ user
        
        Args:
            query: Raw query từ user
            restore_accents: Có restore dấu không
            expand_abbr: Có expand abbreviations không
            remove_stopwords: Có remove stopwords không
            
        Returns:
            Dict {
                'original': query gốc,
                'normalized': query đã normalize,
                'restored': query đã restore dấu,
                'expanded': query đã expand,
                'tokens': list tokens,
                'suggestions': list gợi ý
            }
        """
        result = {
            'original': query,
            'normalized': None,
            'restored': None,
            'expanded': None,
            'tokens': [],
            'suggestions': []
        }
        
        # Step 1: Normalize
        normalized = self.normalize_query(query)
        result['normalized'] = normalized
        
        # Step 2: Expand abbreviations
        if expand_abbr:
            expanded = self.expand_abbreviations(normalized)
        else:
            expanded = normalized
        result['expanded'] = expanded
        
        # Step 3: Restore accents
        if restore_accents:
            restored = self.accent_restorer.restore_text(expanded)
        else:
            restored = expanded
        result['restored'] = restored
        
        # Step 4: Tokenize
        tokens = restored.split()
        
        # Step 5: Remove stopwords (optional)
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        result['tokens'] = tokens
        
        # Step 6: Get suggestions cho từng token
        for token in tokens:
            suggestions = self.accent_restorer.get_suggestions(token, max_suggestions=3)
            if suggestions:
                result['suggestions'].append({
                    'token': token,
                    'alternatives': [s[0] for s in suggestions]
                })
        
        return result


class QueryIndexInterface:
    """
    Interface để query vào inverted index với accent restoration
    """
    
    def __init__(self, index_file: str = None, vocab_file: str = None):
        """
        Args:
            index_file: Path to inverted index pickle file
            vocab_file: Path to vocabulary file
        """
        self.query_processor = QueryProcessor(vocab_file)
        self.index = None
        self.ranker = None
        
        if index_file:
            self.load_index(index_file)
    
    def load_index(self, index_file: str):
        """Load inverted index"""
        try:
            from src.indexing.inverted_index import IndexBuilder
            from src.ranking.rankers import CombinedRanker
            
            self.index = IndexBuilder.load_index_from_pickle(index_file)
            self.ranker = CombinedRanker(self.index)
            
            print(f"✓ Loaded index: {self.index.total_docs:,} documents")
        except Exception as e:
            print(f"✗ Error loading index: {e}")
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               method: str = 'bm25',
               restore_accents: bool = True,
               verbose: bool = True) -> List[Tuple[str, float]]:
        """
        Search với query processing tự động
        
        Args:
            query: Raw query từ user (có thể không dấu)
            top_k: Số kết quả
            method: Ranking method ('bm25', 'tfidf', 'combined')
            restore_accents: Tự động restore dấu
            verbose: In chi tiết query processing
            
        Returns:
            List[(doc_id, score)]
        """
        if not self.ranker:
            print("⚠️ Index chưa được load. Call load_index() trước.")
            return []
        
        # Process query
        processed = self.query_processor.process_query(
            query,
            restore_accents=restore_accents,
            expand_abbr=True,
            remove_stopwords=False
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"QUERY PROCESSING")
            print(f"{'='*80}")
            print(f"Original:   {processed['original']}")
            print(f"Normalized: {processed['normalized']}")
            print(f"Expanded:   {processed['expanded']}")
            print(f"Restored:   {processed['restored']}")
            print(f"Tokens:     {processed['tokens']}")
            
            if processed['suggestions']:
                print(f"\nSuggestions:")
                for sugg in processed['suggestions']:
                    print(f"  '{sugg['token']}' → {sugg['alternatives']}")
            print(f"{'='*80}\n")
        
        # Search với restored tokens
        query_terms = processed['tokens']
        results = self.ranker.search(query_terms, top_k=top_k, method=method)
        
        return results
    
    def interactive_search(self):
        """Interactive search interface"""
        print(f"\n{'='*80}")
        print(f"VIETNAMESE SEARCH ENGINE - INTERACTIVE MODE")
        print(f"{'='*80}")
        print(f"Features:")
        print(f"  ✓ Automatic accent restoration (bong da → bóng đá)")
        print(f"  ✓ Abbreviation expansion (hlv → huấn luyện viên)")
        print(f"  ✓ Smart query processing")
        print(f"\nCommands:")
        print(f"  - Type your query to search")
        print(f"  - Type 'exit' or 'quit' to exit")
        print(f"  - Type 'help' for more commands")
        print(f"{'='*80}\n")
        
        while True:
            try:
                query = input("🔍 Search: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if query.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  search <query>  - Search with accent restoration")
                    print("  raw <query>     - Search without accent restoration")
                    print("  suggest <word>  - Get suggestions for a word")
                    print("  exit/quit       - Exit")
                    continue
                
                if query.startswith('raw '):
                    # Search without accent restoration
                    raw_query = query[4:].strip()
                    results = self.search(raw_query, restore_accents=False, verbose=True)
                elif query.startswith('suggest '):
                    # Get suggestions
                    word = query[8:].strip()
                    suggestions = self.query_processor.accent_restorer.get_suggestions(word)
                    print(f"\nSuggestions for '{word}':")
                    for i, (sugg, score) in enumerate(suggestions, 1):
                        print(f"  {i}. {sugg} (score: {score:.2f})")
                    continue
                else:
                    # Normal search with accent restoration
                    results = self.search(query, restore_accents=True, verbose=True)
                
                # Display results
                if results:
                    print(f"\n📊 RESULTS (Top {len(results)}):")
                    print(f"{'='*80}")
                    for rank, (doc_id, score) in enumerate(results, 1):
                        print(f"{rank:2d}. Doc: {doc_id[:60]:<60} | Score: {score:8.4f}")
                    print(f"{'='*80}\n")
                else:
                    print("❌ No results found.\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def demo():
    """Demo query processing"""
    print("="*80)
    print("QUERY PROCESSING & ACCENT RESTORATION - DEMO")
    print("="*80)
    
    # Initialize
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "bong da viet nam",
        "hlv park hang seo",
        "doi tuyen thai lan",
        "tran chung ket aff cup",
        "cau thu xuat sac nhat"
    ]
    
    print("\n📝 TEST QUERIES:\n")
    
    for query in test_queries:
        result = processor.process_query(query)
        
        print(f"Original:  '{result['original']}'")
        print(f"Restored:  '{result['restored']}'")
        print(f"Tokens:    {result['tokens']}")
        
        if result['suggestions']:
            print(f"Suggestions:")
            for sugg in result['suggestions'][:2]:  # Top 2
                print(f"  - {sugg['token']} → {sugg['alternatives'][:2]}")
        
        print("-" * 80)
    
    print("\n✓ DEMO COMPLETED!")


if __name__ == "__main__":
    demo()
