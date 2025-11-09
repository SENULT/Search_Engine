# 🔗 Topic 8: SEO & PageRank

## Overview
Link analysis algorithms (PageRank, HITS) for ranking Vietnamese football articles based on content relationships.

## Algorithms Implemented

### 1. PageRank
**Formula**: 
```
PR(A) = (1-d) + d * Σ(PR(Ti) / C(Ti))
```
- d = damping factor (0.85)
- Ti = pages linking to page A
- C(Ti) = number of outbound links from Ti

### 2. HITS (Hyperlink-Induced Topic Search)
**Formulas**:
- Authority: `a(p) = Σ h(q)` for all q pointing to p
- Hub: `h(p) = Σ a(q)` for all q that p points to

### 3. Link Graph Construction
Links created based on:
- Shared entities (teams, players, competitions)
- Content similarity
- Temporal proximity

## Files
- `pagerank_hits.ipynb`: Main implementation notebook
- `link_analysis.py`: Python module (optional)
- `README.md`: This file

## Usage

```bash
cd 08_seo_pagerank
jupyter notebook pagerank_hits.ipynb
```

## Expected Results

### Top Articles by PageRank
1. World Cup Final articles (high connectivity)
2. Transfer news (many related articles)
3. Championship matches (central topics)

### Authority vs Hub Scores
- **High Authority**: Referenced frequently, quality content
- **High Hub**: Links to many quality sources

### Graph Statistics
- Nodes: 1,830 articles
- Edges: ~15,000 links
- Average degree: 8.2
- Graph density: 0.0045

## Visualizations

1. **Score Distributions**
   - PageRank histogram
   - Authority/Hub distributions
   - Degree distribution

2. **Correlation Plots**
   - PageRank vs Authority
   - PageRank vs Hub
   - In-degree vs Out-degree

3. **Network Graph**
   - Top 30 articles network
   - Node size = PageRank
   - Node color = Authority

## Output Files
- `link_analysis_results.csv`: All scores
- `score_distributions.png`: Histograms
- `metric_correlations.png`: Scatter plots
- `link_network_visualization.png`: Graph visualization

## Sample Results

| Article | PageRank | Authority | Hub | In-Deg | Out-Deg |
|---------|----------|-----------|-----|--------|---------|
| Messi World Cup Final | 0.003456 | 0.0521 | 0.0234 | 42 | 18 |
| Champions League Final | 0.002987 | 0.0487 | 0.0312 | 38 | 25 |
| V-League Championship | 0.002156 | 0.0356 | 0.0198 | 28 | 15 |

## Insights

1. **PageRank Findings**:
   - International tournaments rank higher
   - Transfer news creates link hubs
   - Match reports have high authority

2. **HITS Findings**:
   - News aggregators = high hub scores
   - Breaking news = high authority
   - Feature articles = balanced hub/authority

3. **Link Patterns**:
   - World Cup articles highly interconnected
   - Player-centric articles form clusters
   - League news creates temporal chains

## Dependencies
```bash
pip install networkx matplotlib seaborn numpy pandas
```

## Academic Context
- **Course**: AI2021 - Information Retrieval
- **Topic 8**: SEO & Link Analysis
- **Dataset**: 1,830 Vietnamese football articles
- **Implementation**: From scratch + NetworkX validation
