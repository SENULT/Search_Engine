# 🔍 Vietnamese Football Search Engine# Search_Engine



A comprehensive search engine system for Vietnamese text with advanced NLP preprocessing, inverted indexing, and multiple ranking algorithms.Project: Search Engine offline with topic Football in VietNam



## 📋 Features

# Create file .env

### 1. **Web Crawling**
- VnExpress news crawler
- MongoDB storage integration
- Configurable data extraction

### 2. **Text Preprocessing**
- Vietnamese tokenization (PyVi)
- Stopwords removal (45+ Vietnamese stopwords)
- POS tagging with Hidden Markov Model
- Stemming and lemmatization
- Advanced normalization:
  - Number normalization (text to digits)
  - Date normalization (standardized format)
  - Abbreviation expansion
- Spell checking with Levenshtein distance
- N-grams analysis (1-5 grams)

### 3. **Inverted Index**
- Efficient document indexing
- Term frequency (TF) calculation
- Document frequency (DF) tracking
- Position storage for phrase queries
- Multiple storage formats:
  - MongoDB (distributed)
  - JSON (human-readable)
  - Pickle (fast loading)

### 4. **Ranking Algorithms**
- **TF-IDF**: Classic term frequency-inverse document frequency
- **BM25**: Okapi BM25 with tunable parameters (k1, b)
- **BM25+**: Enhanced BM25 with delta parameter
- **Phrase Scoring**: Proximity-based ranking
- **Field-Based Scoring**: Multi-field weighting (title, content, etc.)
- **Combined Ranking**: Weighted ensemble of multiple algorithms

### 5. **Query Processing**
- Query expansion with Pseudo Relevance Feedback (PRF)
- Synonym expansion
- Multi-term query support

### 6. **Evaluation Metrics**
- Precision@K
- Recall@K
- Average Precision (AP)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Mean Reciprocal Rank (MRR)

---

## 📁 Project Structure

```
Search_Engine/
├── src/                           # Source code
│   ├── crawling/                 # Web crawling module
│   │   ├── __init__.py
│   │   └── crawler.py            # VnExpress crawler
│   ├── preprocessing/            # Text preprocessing
│   │   ├── __init__.py
│   │   └── processor.py          # Vietnamese text processor
│   ├── indexing/                 # Inverted index
│   │   ├── __init__.py
│   │   └── inverted_index.py     # Index builder & storage
│   ├── ranking/                  # Ranking algorithms
│   │   ├── __init__.py
│   │   └── rankers.py            # Multiple rankers
│   └── utils/                    # Utilities
│       ├── __init__.py
│       └── database.py           # MongoDB utilities
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_crawling_demo.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_indexing_demo.ipynb
│   └── 04_ranking_demo.ipynb
│
├── data/                          # Data storage
│   ├── raw/                      # Raw crawled data
│   │   ├── vnexpress_bongda_part1.json
│   │   ├── vnexpress_bongda_part2.json
│   │   ├── vnexpress_bongda_part3.json
│   │   └── vnexpress_bongda_part4.json
│   ├── processed/                # Preprocessed data
│   └── vocab/                    # Vocabularies & stopwords
│       └── vocab.txt
│
├── outputs/                       # Output files
│   ├── indexes/                  # Inverted indexes
│   ├── processed/                # Processing results
│   └── logs/                     # Log files
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_indexing.py
│   └── test_ranking.py
│
├── configs/                       # Configuration files
│   └── config.yaml
│
├── .env                          # Environment variables (not in git)
├── .env.example                  # Template for .env
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MongoDB 4.0+
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SENULT/Search_Engine.git
   cd Search_Engine
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy .env.example to .env
   cp .env.example .env
   
   # Edit .env and add your MongoDB connection string
   # MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/
   ```

### Quick Start

#### 1. Text Preprocessing

```python
from src.preprocessing.processor import VietnameseTextProcessor

# Initialize processor
processor = VietnameseTextProcessor()

# Process text
text = "Đội tuyển Việt Nam giành chiến thắng 3-0 trước Thái Lan"
result = processor.complete_text_processing_pipeline(text)

print(result['tokens'])          # Tokenized words
print(result['filtered_tokens']) # After stopwords removal
print(result['pos_tags'])        # POS tags
print(result['stemmed_tokens'])  # Stemmed tokens
```

#### 2. Build Inverted Index

```python
from src.indexing.inverted_index import IndexBuilder

# Initialize builder
builder = IndexBuilder()

# Build from MongoDB
builder.build_index_from_collection(
    collection_name="vnexpress_bongda",
    token_field="filtered_tokens",
    limit=1000
)

# Save index
builder.save_index_to_pickle("outputs/indexes/index.pkl")
builder.save_index_to_mongodb("inverted_index")
```

#### 3. Search & Ranking

```python
from src.ranking.rankers import CombinedRanker
from src.indexing.inverted_index import IndexBuilder

# Load index
inv_index = IndexBuilder.load_index_from_pickle("outputs/indexes/index.pkl")

# Initialize ranker
ranker = CombinedRanker(inv_index)

# Search with query
query_terms = ['bóng_đá', 'việt_nam', 'vô_địch']
results = ranker.search(query_terms, top_k=10, method='bm25')

for rank, (doc_id, score) in enumerate(results, 1):
    print(f"{rank}. Document: {doc_id} | Score: {score:.4f}")
```

#### 4. Query Expansion

```python
from src.ranking.rankers import QueryExpander

# Initialize expander
expander = QueryExpander(inv_index)

# Expand with PRF
expanded_query = expander.expand_with_prf(
    query_terms=['bóng_đá', 'việt_nam'],
    ranker=ranker,
    num_docs=5,
    num_terms=3
)

print(f"Original: {query_terms}")
print(f"Expanded: {expanded_query}")
```

---

## 📊 Ranking Algorithms Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **TF-IDF** | Simple, fast | No length normalization | Short documents |
| **BM25** | Industry standard, length normalization | Requires parameter tuning | General purpose |
| **BM25+** | Better than BM25, handles edge cases | Slightly slower | Long documents |
| **Phrase** | Context-aware | Requires positions | Phrase queries |
| **Combined** | Best of all | Complex | Production systems |

### BM25 Parameters

- **k1** (1.0-2.0): Controls term frequency saturation
  - Lower: More linear TF
  - Higher: More saturation
  - Default: **1.5**

- **b** (0.0-1.0): Controls length normalization
  - 0: No normalization
  - 1: Full normalization
  - Default: **0.75**

---

## 🧪 Running Notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open notebooks in order:
1. `01_crawling_demo.ipynb` - Data collection
2. `02_text_preprocessing.ipynb` - Text processing
3. `03_indexing_demo.ipynb` - Index building
4. `04_ranking_demo.ipynb` - Search & ranking

---

## 🧰 Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Ranking settings
ranking:
  default_method: bm25
  bm25:
    k1: 1.5
    b: 0.75
  top_k: 10
  query_expansion: true
  prf_docs: 5
  prf_terms: 3
```

---

## 🧪 Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_ranking.py
```

---

## 📈 Performance Tips

1. **Indexing**
   - Use pickle format for fastest loading
   - Enable batch processing for large datasets
   - Store positions only if needed for phrase queries

2. **Ranking**
   - Use BM25 for best accuracy/speed tradeoff
   - Enable query expansion for better recall
   - Tune k1, b parameters based on your data

3. **Preprocessing**
   - Disable spell checking for faster processing
   - Cache vocabulary for repeated use
   - Use batch processing for multiple documents

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **Search Engine Team** - FPT University

---

## 🙏 Acknowledgments

- **PyVi** - Vietnamese NLP toolkit
- **Underthesea** - Vietnamese NLP library
- **MongoDB** - Database storage
- **BM25** - Ranking algorithm by Stephen Robertson

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

## 🔗 References

1. Robertson, S. E., & Walker, S. (1994). "Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval"
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). "Introduction to Information Retrieval"
3. PyVi Documentation: https://github.com/trungtv/pyvi

---

**Happy Searching! 🔍**
