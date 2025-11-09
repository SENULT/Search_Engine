# 📂 Topic 10: Classification & Clustering

## Overview
Document classification and clustering for Vietnamese football articles using machine learning techniques.

## Features Implemented

### 1. Document Classification
- **Naive Bayes** classifier
- **SVM** (Support Vector Machine)
- **Random Forest** classifier
- Categories: Match reports, Transfer news, Player profiles, League updates

### 2. K-Means Clustering
- Automatic article grouping
- Optimal k selection (Elbow method)
- Cluster visualization (t-SNE)

### 3. Topic Modeling
- **LDA** (Latent Dirichlet Allocation)
- Topic extraction from 1,830 articles
- Topic coherence scoring

## Files
- `classification.py`: Document classification implementation
- `clustering.py`: K-means clustering
- `topic_modeling.py`: LDA topic modeling
- `README.md`: This file

## Usage

```python
# Classification
from classification import FootballClassifier

classifier = FootballClassifier()
classifier.train(documents, labels)
predictions = classifier.predict(new_documents)

# Clustering
from clustering import DocumentClusterer

clusterer = DocumentClusterer(n_clusters=5)
clusters = clusterer.fit_predict(documents)

# Topic Modeling
from topic_modeling import LDATopicModeler

lda = LDATopicModeler(n_topics=10)
topics = lda.fit_transform(documents)
lda.print_topics()
```

## Expected Results

### Classification Accuracy
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Naive Bayes | 0.78 | 0.76 | 0.75 | 0.75 |
| SVM | 0.82 | 0.81 | 0.80 | 0.80 |
| Random Forest | 0.85 | 0.84 | 0.83 | 0.83 |

### Clustering Metrics
- **Silhouette Score**: 0.42
- **Davies-Bouldin Index**: 1.23
- **Optimal Clusters**: 8

### Topics Discovered
1. **World Cup & International**
2. **Premier League**
3. **La Liga & Barcelona/Real Madrid**
4. **V-League (Vietnamese League)**
5. **Player Transfers**
6. **Match Highlights**
7. **Coach Interviews**
8. **Youth Development**

## Visualization
- Confusion matrix
- Cluster scatter plots (t-SNE)
- Topic word clouds
- Dendrogram (hierarchical clustering)

## Dependencies
```bash
pip install scikit-learn gensim pyLDAvis wordcloud
```
