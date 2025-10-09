"""
RANKING & INDEXING SYSTEM FOR SEARCH ENGINE

Xây dựng hệ thống ranking và indexing với nhiều thuật toán:
- TF-IDF
- BM25
- PageRank-like (based on link analysis)
- Learning to Rank features

Features:
1. Multiple ranking algorithms (TF-IDF, BM25, BM25+, etc.)
2. Document ranking với query expansion
3. Relevance feedback
4. Phrase matching với proximity scoring
5. Field-based scoring (title, content, metadata)
6. Integration với Inverted Index
"""

import os
import json
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
from datetime import datetime
from tqdm import tqdm

# Import InvertedIndex từ indexing module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from indexing.inverted_index import InvertedIndex, IndexBuilder


class RankingScorer:
    """
    Base class cho các scoring algorithms
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
        
    def score(self, query_terms: List[str], doc_id: str) -> float:
        """Tính score cho document với query"""
        raise NotImplementedError


class TFIDFScorer(RankingScorer):
    """
    TF-IDF Scoring
    
    TF-IDF = TF * IDF
    TF = term frequency trong document
    IDF = log(N / df)
    """
    
    def score(self, query_terms: List[str], doc_id: str) -> float:
        score = 0.0
        
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]['postings']:
                tf = self.index.index[term]['postings'][doc_id]['tf']
                df = self.index.index[term]['df']
                idf = math.log(self.index.total_docs / df) if df > 0 else 0
                
                score += tf * idf
        
        return score


class BM25Scorer(RankingScorer):
    """
    BM25 Scoring (Okapi BM25)
    
    BM25 = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))
    
    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls length normalization (default: 0.75)
    """
    
    def __init__(self, inverted_index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        super().__init__(inverted_index)
        self.k1 = k1
        self.b = b
        
        # Tính average document length
        if self.index.avg_doc_length == 0 and self.index.doc_lengths:
            self.index.avg_doc_length = sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)
    
    def score(self, query_terms: List[str], doc_id: str) -> float:
        score = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        
        if doc_len == 0:
            return 0.0
        
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]['postings']:
                tf = self.index.index[term]['postings'][doc_id]['tf']
                df = self.index.index[term]['df']
                idf = math.log((self.index.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.index.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score


class BM25PlusScorer(RankingScorer):
    """
    BM25+ (Improved BM25)
    
    BM25+ = Σ IDF(qi) * (δ + (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))
    
    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls length normalization (default: 0.75)
        delta: Lower bound parameter (default: 1.0)
    """
    
    def __init__(self, inverted_index: InvertedIndex, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        super().__init__(inverted_index)
        self.k1 = k1
        self.b = b
        self.delta = delta
        
        if self.index.avg_doc_length == 0 and self.index.doc_lengths:
            self.index.avg_doc_length = sum(self.index.doc_lengths.values()) / len(self.index.doc_lengths)
    
    def score(self, query_terms: List[str], doc_id: str) -> float:
        score = 0.0
        doc_len = self.index.doc_lengths.get(doc_id, 0)
        
        if doc_len == 0:
            return 0.0
        
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]['postings']:
                tf = self.index.index[term]['postings'][doc_id]['tf']
                df = self.index.index[term]['df']
                idf = math.log((self.index.total_docs - df + 0.5) / (df + 0.5) + 1.0)
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.index.avg_doc_length))
                
                score += idf * (self.delta + (numerator / denominator))
        
        return score


class PhraseScorer(RankingScorer):
    """
    Phrase Proximity Scoring
    
    Tính score dựa trên proximity của các terms trong query.
    Nếu các terms xuất hiện gần nhau → score cao hơn
    """
    
    def __init__(self, inverted_index: InvertedIndex, max_distance: int = 10):
        super().__init__(inverted_index)
        self.max_distance = max_distance
    
    def score(self, query_terms: List[str], doc_id: str) -> float:
        if len(query_terms) < 2:
            return 0.0
        
        score = 0.0
        
        # Lấy positions của tất cả query terms trong document
        term_positions = {}
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]['postings']:
                term_positions[term] = self.index.index[term]['postings'][doc_id]['positions']
        
        # Nếu không có đủ terms
        if len(term_positions) < 2:
            return 0.0
        
        # Tính proximity score cho mỗi cặp terms liên tiếp
        for i in range(len(query_terms) - 1):
            term1 = query_terms[i]
            term2 = query_terms[i + 1]
            
            if term1 not in term_positions or term2 not in term_positions:
                continue
            
            pos1_list = term_positions[term1]
            pos2_list = term_positions[term2]
            
            # Tìm cặp positions gần nhau nhất
            min_distance = float('inf')
            for pos1 in pos1_list:
                for pos2 in pos2_list:
                    distance = abs(pos2 - pos1)
                    if distance < min_distance:
                        min_distance = distance
            
            # Score cao hơn nếu distance nhỏ hơn
            if min_distance <= self.max_distance:
                score += 1.0 / (1.0 + min_distance)
        
        return score


class FieldBasedScorer(RankingScorer):
    """
    Field-Based Scoring
    
    Tính score khác nhau cho các fields:
    - Title: weight cao nhất
    - Content: weight trung bình
    - Metadata: weight thấp
    """
    
    def __init__(self, 
                 inverted_index: InvertedIndex, 
                 field_weights: Dict[str, float] = None):
        super().__init__(inverted_index)
        
        # Default weights
        self.field_weights = field_weights or {
            'title': 2.0,
            'content': 1.0,
            'description': 1.5,
            'tags': 1.2
        }
    
    def score(self, query_terms: List[str], doc_id: str, doc_fields: Dict[str, List[str]]) -> float:
        """
        Tính score dựa trên multiple fields
        
        Args:
            query_terms: List query terms
            doc_id: Document ID
            doc_fields: Dict {field_name: tokens_list}
        """
        total_score = 0.0
        
        for field_name, tokens in doc_fields.items():
            weight = self.field_weights.get(field_name, 1.0)
            
            # Tính term frequency trong field này
            term_freq = Counter(tokens)
            
            field_score = 0.0
            for term in query_terms:
                if term in term_freq:
                    tf = term_freq[term]
                    # Simple TF scoring cho field
                    field_score += tf
            
            total_score += weight * field_score
        
        return total_score


class CombinedRanker:
    """
    Combined Ranking System
    
    Kết hợp nhiều scoring algorithms với weights
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
        
        # Khởi tạo các scorers
        self.scorers = {
            'tfidf': TFIDFScorer(inverted_index),
            'bm25': BM25Scorer(inverted_index),
            'bm25plus': BM25PlusScorer(inverted_index),
            'phrase': PhraseScorer(inverted_index)
        }
        
        # Default weights cho mỗi scorer
        self.scorer_weights = {
            'tfidf': 0.2,
            'bm25': 0.4,
            'bm25plus': 0.3,
            'phrase': 0.1
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Cập nhật weights cho các scorers"""
        self.scorer_weights.update(weights)
    
    def rank_documents(self, 
                       query_terms: List[str], 
                       candidate_docs: Set[str] = None,
                       top_k: int = 10,
                       use_scorers: List[str] = None) -> List[Tuple[str, float, Dict]]:
        """
        Rank documents cho query
        
        Args:
            query_terms: List query terms (đã preprocessing)
            candidate_docs: Set document IDs để xét (None = all docs)
            top_k: Số documents trả về
            use_scorers: List scorer names sử dụng (None = all)
            
        Returns:
            List[(doc_id, total_score, score_breakdown)]
        """
        # Nếu không chỉ định candidate docs, lấy tất cả docs có ít nhất 1 query term
        if candidate_docs is None:
            candidate_docs = set()
            for term in query_terms:
                if term in self.index.index:
                    candidate_docs.update(self.index.index[term]['postings'].keys())
        
        if not candidate_docs:
            return []
        
        # Chọn scorers
        if use_scorers is None:
            use_scorers = list(self.scorers.keys())
        
        # Tính score cho từng document
        doc_scores = []
        
        for doc_id in candidate_docs:
            score_breakdown = {}
            total_score = 0.0
            
            for scorer_name in use_scorers:
                if scorer_name not in self.scorers:
                    continue
                
                scorer = self.scorers[scorer_name]
                weight = self.scorer_weights.get(scorer_name, 1.0)
                
                score = scorer.score(query_terms, doc_id)
                score_breakdown[scorer_name] = score
                
                total_score += weight * score
            
            doc_scores.append((doc_id, total_score, score_breakdown))
        
        # Sắp xếp theo total_score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]
    
    def search(self, 
               query_terms: List[str], 
               top_k: int = 10,
               method: str = 'combined') -> List[Tuple[str, float]]:
        """
        Search interface đơn giản
        
        Args:
            query_terms: List query terms
            top_k: Số kết quả
            method: 'combined', 'bm25', 'tfidf', 'bm25plus'
            
        Returns:
            List[(doc_id, score)]
        """
        if method == 'combined':
            results = self.rank_documents(query_terms, top_k=top_k)
            return [(doc_id, score) for doc_id, score, _ in results]
        else:
            # Sử dụng single scorer
            if method not in self.scorers:
                raise ValueError(f"Unknown method: {method}")
            
            scorer = self.scorers[method]
            
            # Lấy candidate docs
            candidate_docs = set()
            for term in query_terms:
                if term in self.index.index:
                    candidate_docs.update(self.index.index[term]['postings'].keys())
            
            doc_scores = []
            for doc_id in candidate_docs:
                score = scorer.score(query_terms, doc_id)
                doc_scores.append((doc_id, score))
            
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores[:top_k]


class QueryExpander:
    """
    Query Expansion để cải thiện recall
    
    Methods:
    - Pseudo Relevance Feedback (PRF)
    - Synonym expansion
    """
    
    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
    
    def expand_with_prf(self, 
                        query_terms: List[str], 
                        ranker: CombinedRanker,
                        num_docs: int = 5,
                        num_terms: int = 3) -> List[str]:
        """
        Pseudo Relevance Feedback
        
        1. Retrieve top-k documents
        2. Lấy top terms từ documents đó
        3. Add vào query
        """
        # Retrieve top documents
        top_docs = ranker.search(query_terms, top_k=num_docs, method='bm25')
        
        if not top_docs:
            return query_terms
        
        # Đếm term frequency trong top documents
        term_freq = Counter()
        
        for doc_id, _ in top_docs:
            doc_len = self.index.doc_lengths.get(doc_id, 0)
            
            # Lấy tất cả terms trong document
            for term in self.index.vocabulary:
                if term in self.index.index and doc_id in self.index.index[term]['postings']:
                    tf = self.index.index[term]['postings'][doc_id]['tf']
                    term_freq[term] += tf
        
        # Loại bỏ query terms đã có
        for term in query_terms:
            if term in term_freq:
                del term_freq[term]
        
        # Lấy top terms
        expansion_terms = [term for term, _ in term_freq.most_common(num_terms)]
        
        # Kết hợp với query gốc
        expanded_query = query_terms + expansion_terms
        
        return expanded_query
    
    def expand_with_synonyms(self, 
                            query_terms: List[str], 
                            synonym_dict: Dict[str, List[str]]) -> List[str]:
        """
        Expand query với synonyms
        
        Args:
            query_terms: Original query terms
            synonym_dict: {term: [synonym1, synonym2, ...]}
        """
        expanded = query_terms.copy()
        
        for term in query_terms:
            if term in synonym_dict:
                synonyms = synonym_dict[term]
                # Thêm synonyms có trong vocabulary
                for syn in synonyms:
                    if syn in self.index.vocabulary and syn not in expanded:
                        expanded.append(syn)
        
        return expanded


class RankingEvaluator:
    """
    Evaluation metrics cho ranking
    """
    
    @staticmethod
    def precision_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """
        Precision@K = (số relevant docs trong top-K) / K
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_count / k
    
    @staticmethod
    def recall_at_k(relevant_docs: Set[str], retrieved_docs: List[str], k: int) -> float:
        """
        Recall@K = (số relevant docs trong top-K) / (tổng số relevant docs)
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_count = sum(1 for doc in top_k if doc in relevant_docs)
        
        return relevant_count / len(relevant_docs)
    
    @staticmethod
    def average_precision(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """
        Average Precision (AP)
        
        AP = (1/|relevant|) * Σ (Precision@k * rel(k))
        """
        if len(relevant_docs) == 0:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for k, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                relevant_count += 1
                precision_at_k = relevant_count / k
                precisions.append(precision_at_k)
        
        if not precisions:
            return 0.0
        
        return sum(precisions) / len(relevant_docs)
    
    @staticmethod
    def ndcg_at_k(relevance_scores: Dict[str, float], retrieved_docs: List[str], k: int) -> float:
        """
        Normalized Discounted Cumulative Gain (NDCG@K)
        
        DCG@K = Σ (2^rel(i) - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K
        """
        if k == 0:
            return 0.0
        
        top_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc in enumerate(top_k, 1):
            rel = relevance_scores.get(doc, 0.0)
            dcg += (2 ** rel - 1) / math.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        idcg = 0.0
        for i, (doc, rel) in enumerate(ideal_docs[:k], 1):
            idcg += (2 ** rel - 1) / math.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_reciprocal_rank(relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """
        Mean Reciprocal Rank (MRR)
        
        MRR = 1 / rank(first_relevant_doc)
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        
        return 0.0


def demo_ranking():
    """Demo ranking system"""
    print("="*80)
    print("RANKING & INDEXING DEMO")
    print("="*80)
    
    # 1. Load inverted index (từ file pickle hoặc build mới)
    print("\n1. Loading Inverted Index...")
    
    try:
        # Try load từ pickle
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from indexing.inverted_index import IndexBuilder
        inv_index = IndexBuilder.load_index_from_pickle("outputs/inverted_index_latest.pkl")
    except:
        print("⚠️ Không tìm thấy index file. Cần build index trước!")
        return
    
    # 2. Khởi tạo ranker
    print("\n2. Initializing Ranker...")
    ranker = CombinedRanker(inv_index)
    
    # 3. Demo search với query
    print("\n3. Testing Search...")
    
    # Query example (đã preprocessing)
    query_terms = ['bóng_đá', 'việt_nam', 'vô_địch']
    print(f"\nQuery: {' '.join(query_terms)}")
    
    # Search với BM25
    print("\n--- BM25 Results ---")
    results_bm25 = ranker.search(query_terms, top_k=5, method='bm25')
    for rank, (doc_id, score) in enumerate(results_bm25, 1):
        print(f"{rank}. Doc: {doc_id[:20]}... | Score: {score:.4f}")
    
    # Search với Combined
    print("\n--- Combined Results ---")
    results_combined = ranker.rank_documents(query_terms, top_k=5)
    for rank, (doc_id, total_score, breakdown) in enumerate(results_combined, 1):
        print(f"{rank}. Doc: {doc_id[:20]}... | Total: {total_score:.4f}")
        print(f"    Breakdown: {breakdown}")
    
    # 4. Query Expansion
    print("\n4. Testing Query Expansion...")
    expander = QueryExpander(inv_index)
    expanded_query = expander.expand_with_prf(query_terms, ranker, num_docs=3, num_terms=2)
    print(f"Original query: {query_terms}")
    print(f"Expanded query: {expanded_query}")
    
    # 5. Evaluation example
    print("\n5. Evaluation Metrics Example...")
    relevant_docs = {results_bm25[0][0], results_bm25[1][0]}  # Giả sử 2 docs đầu là relevant
    retrieved_docs = [doc_id for doc_id, _ in results_bm25]
    
    evaluator = RankingEvaluator()
    p_at_3 = evaluator.precision_at_k(relevant_docs, retrieved_docs, k=3)
    r_at_3 = evaluator.recall_at_k(relevant_docs, retrieved_docs, k=3)
    mrr = evaluator.mean_reciprocal_rank(relevant_docs, retrieved_docs)
    
    print(f"Precision@3: {p_at_3:.4f}")
    print(f"Recall@3: {r_at_3:.4f}")
    print(f"MRR: {mrr:.4f}")
    
    print("\n✓ DEMO COMPLETED!")


if __name__ == "__main__":
    demo_ranking()
