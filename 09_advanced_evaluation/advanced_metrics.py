"""
Topic 9: Advanced Evaluation Metrics
NDCG, MAP, MRR Implementation
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple

class AdvancedEvaluator:
    """Advanced metrics for search engine evaluation"""
    
    def __init__(self):
        self.results = {}
    
    def dcg_at_k(self, relevances: List[float], k: int) -> float:
        """
        Compute Discounted Cumulative Gain at k
        
        Formula: DCG@k = sum(rel_i / log2(i+1)) for i=1 to k
        """
        relevances = np.array(relevances)[:k]
        if relevances.size == 0:
            return 0.0
        
        # Positions start at 1
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)
    
    def ndcg_at_k(self, relevances: List[float], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k
        
        Formula: NDCG@k = DCG@k / IDCG@k
        where IDCG is DCG for perfect ranking
        """
        dcg = self.dcg_at_k(relevances, k)
        
        # Ideal DCG (sorted descending)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = self.dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def average_precision(self, relevances: List[int]) -> float:
        """
        Compute Average Precision
        
        Formula: AP = (sum of P@k * rel(k)) / total relevant docs
        """
        relevances = np.array(relevances)
        if np.sum(relevances) == 0:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for k, rel in enumerate(relevances, 1):
            if rel > 0:
                num_relevant += 1
                precision_at_k = num_relevant / k
                precisions.append(precision_at_k)
        
        if not precisions:
            return 0.0
        
        return np.mean(precisions)
    
    def mean_average_precision(self, all_relevances: List[List[int]]) -> float:
        """
        Compute Mean Average Precision across queries
        
        Formula: MAP = mean(AP for each query)
        """
        aps = [self.average_precision(rel) for rel in all_relevances]
        return np.mean(aps) if aps else 0.0
    
    def reciprocal_rank(self, relevances: List[int]) -> float:
        """
        Compute Reciprocal Rank (1/rank of first relevant doc)
        """
        for rank, rel in enumerate(relevances, 1):
            if rel > 0:
                return 1.0 / rank
        return 0.0
    
    def mean_reciprocal_rank(self, all_relevances: List[List[int]]) -> float:
        """
        Compute Mean Reciprocal Rank across queries
        
        Formula: MRR = mean(RR for each query)
        """
        rrs = [self.reciprocal_rank(rel) for rel in all_relevances]
        return np.mean(rrs) if rrs else 0.0
    
    def precision_at_k(self, relevances: List[int], k: int) -> float:
        """
        Compute Precision at k
        
        Formula: P@k = (relevant docs in top k) / k
        """
        relevances = np.array(relevances)[:k]
        if len(relevances) == 0:
            return 0.0
        return np.sum(relevances > 0) / k
    
    def recall_at_k(self, relevances: List[int], k: int, total_relevant: int) -> float:
        """
        Compute Recall at k
        
        Formula: R@k = (relevant docs in top k) / total relevant docs
        """
        if total_relevant == 0:
            return 0.0
        
        relevances = np.array(relevances)[:k]
        return np.sum(relevances > 0) / total_relevant
    
    def f1_at_k(self, relevances: List[int], k: int, total_relevant: int) -> float:
        """
        Compute F1 Score at k
        
        Formula: F1@k = 2 * (P@k * R@k) / (P@k + R@k)
        """
        p = self.precision_at_k(relevances, k)
        r = self.recall_at_k(relevances, k, total_relevant)
        
        if p + r == 0:
            return 0.0
        
        return 2 * (p * r) / (p + r)


def demo_evaluation():
    """Demo evaluation with sample queries"""
    
    evaluator = AdvancedEvaluator()
    
    # Sample: 3 queries with relevance judgments
    # 1 = relevant, 0 = not relevant
    # Format: [query1_results, query2_results, query3_results]
    sample_queries = [
        # Query 1: "Messi World Cup"
        [1, 1, 0, 1, 0, 0, 1, 0, 0, 0],  # 4 relevant in top 10
        
        # Query 2: "Manchester United"
        [1, 0, 1, 1, 0, 1, 0, 0, 0, 0],  # 4 relevant in top 10
        
        # Query 3: "V-League 2024"
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # 3 relevant in top 10
    ]
    
    print("\n" + "="*80)
    print("📊 ADVANCED EVALUATION METRICS DEMO")
    print("="*80)
    
    # Evaluate each query
    for i, relevances in enumerate(sample_queries, 1):
        print(f"\n📝 Query {i}:")
        print(f"  Relevance pattern: {relevances}")
        print(f"  NDCG@10: {evaluator.ndcg_at_k(relevances, 10):.4f}")
        print(f"  AP:      {evaluator.average_precision(relevances):.4f}")
        print(f"  RR:      {evaluator.reciprocal_rank(relevances):.4f}")
        print(f"  P@5:     {evaluator.precision_at_k(relevances, 5):.4f}")
        print(f"  R@5:     {evaluator.recall_at_k(relevances, 5, sum(relevances)):.4f}")
    
    # Overall metrics
    print(f"\n📈 Overall Metrics (across {len(sample_queries)} queries):")
    print(f"  MAP:     {evaluator.mean_average_precision(sample_queries):.4f}")
    print(f"  MRR:     {evaluator.mean_reciprocal_rank(sample_queries):.4f}")
    
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    demo_evaluation()
