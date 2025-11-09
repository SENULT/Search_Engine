# 📈 Topic 9: Advanced Evaluation

## Files
- `ndcg_map_evaluation.ipynb`: NDCG, MAP, MRR metrics
- `user_study.ipynb`: User study framework
- `ab_testing.ipynb`: A/B testing for ranking methods

## Metrics Implemented

### 1. NDCG (Normalized Discounted Cumulative Gain)
- Measures ranking quality
- Considers position of relevant documents
- Formula: $NDCG@k = \frac{DCG@k}{IDCG@k}$

### 2. MAP (Mean Average Precision)
- Average precision across multiple queries
- Formula: $MAP = \frac{1}{|Q|} \sum_{q=1}^{|Q|} AP(q)$

### 3. MRR (Mean Reciprocal Rank)
- Average of reciprocal ranks
- Formula: $MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$

### 4. User Study
- User satisfaction surveys
- Click-through rate (CTR)
- Time to first click
- Session duration

### 5. A/B Testing
- Compare BM25 vs Neural models
- Statistical significance tests
- Confidence intervals

## Usage

```bash
cd 09_advanced_evaluation
jupyter notebook ndcg_map_evaluation.ipynb
```

## Expected Results

| Metric | BM25 | Conv-KNRM | DeepCT |
|--------|------|-----------|--------|
| NDCG@10 | 0.72 | 0.78 | 0.81 |
| MAP | 0.68 | 0.74 | 0.77 |
| MRR | 0.75 | 0.80 | 0.83 |

## Output Files
- `evaluation_metrics.csv`
- `user_study_results.json`
- `ab_test_report.html`
