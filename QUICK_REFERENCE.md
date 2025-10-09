# 🚀 Quick Reference - Folder Structure

## 📍 Where to Find What

### 🐍 Python Code
```
src/indexing/inverted_index.py    →  Build inverted index
src/ranking/rankers.py            →  Ranking algorithms (BM25, TF-IDF, etc.)
src/crawling/crawler.py           →  Web crawler
src/utils/database.py             →  MongoDB utilities
```

### 📓 Notebooks
```
notebooks/01_crawling_demo.ipynb         →  Crawling demo
notebooks/02_text_preprocessing.ipynb    →  Text preprocessing (MAIN)
```

### 📦 Data
```
data/raw/vnexpress_bongda_part*.json    →  Raw Vietnamese football news
data/vocab/vocab.txt                     →  Vietnamese vocabulary
```

### 📤 Outputs
```
outputs/indexes/         →  Save index files here (.pkl, .json)
outputs/processed/       →  Save processing results
outputs/logs/           →  Log files
```

### ⚙️ Configuration
```
.env                  →  Environment variables (your MONGO_URI)
configs/config.yaml   →  Project configuration
requirements.txt      →  Python dependencies
```

---

## 💻 Common Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Then edit .env
```

### Run Notebooks
```bash
jupyter notebook
# Open: notebooks/02_text_preprocessing.ipynb
```

### Test Structure
```bash
python test_imports.py
```

### Check Files
```bash
tree /F /A
```

---

## 📝 Import Cheat Sheet

```python
# Indexing
from src.indexing.inverted_index import InvertedIndex, IndexBuilder

# Ranking
from src.ranking.rankers import (
    BM25Scorer,
    TFIDFScorer,
    BM25PlusScorer,
    CombinedRanker,
    QueryExpander
)

# Utils
from src.utils.database import DatabaseManager
```

---

## 🎯 Quick Tasks

### Build Index
```python
from src.indexing.inverted_index import IndexBuilder

builder = IndexBuilder()
builder.build_index_from_collection("vnexpress_bongda", limit=1000)
builder.save_index_to_pickle("outputs/indexes/index.pkl")
```

### Search
```python
from src.ranking.rankers import CombinedRanker
from src.indexing.inverted_index import IndexBuilder

index = IndexBuilder.load_index_from_pickle("outputs/indexes/index.pkl")
ranker = CombinedRanker(index)

results = ranker.search(['bóng_đá', 'việt_nam'], top_k=10, method='bm25')
```

---

## 📂 File Paths (Updated)

### Before → After
```
indexing.py              →  src/indexing/inverted_index.py
ranking_indexing.py      →  src/ranking/rankers.py
updata.py                →  src/crawling/crawler.py
crawlcode.ipynb          →  notebooks/01_crawling_demo.ipynb
textprocessing.ipynb     →  notebooks/02_text_preprocessing.ipynb
vnexpress_*.json         →  data/raw/vnexpress_*.json
vocab.txt                →  data/vocab/vocab.txt
```

---

## 🔗 Documentation

- **README.md** - Full documentation
- **MIGRATION.md** - Migration guide
- **STRUCTURE_SUMMARY.md** - Detailed summary
- **This file** - Quick reference

---

**Keep this file handy for quick lookups! 📌**
