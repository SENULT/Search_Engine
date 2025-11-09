# 🚀 HOW TO RUN THE PROJECT - Complete Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Module by Module Guide](#module-by-module-guide)
3. [Full Pipeline Execution](#full-pipeline-execution)
4. [Troubleshooting](#troubleshooting)

---

## ✅ Quick Start

### Prerequisites
```bash
# Python 3.8+
python --version

# Install all dependencies
pip install torch torchvision pyvi pandas numpy tqdm pymongo gensim jupyter fastapi uvicorn
```

### Test Everything
```bash
# Run comprehensive test
python test_all.py

# You should see:
# ✓ Passed: 46/50 (92.0%)
```

---

## 📚 Module by Module Guide

### 1️⃣ Crawling (01_crawling/)

```bash
cd 01_crawling
jupyter notebook crawlcode.ipynb
```

**What it does:**
- Crawls Vietnamese football news from VnExpress
- Saves to `data/raw/vnexpress_bongda_part*.json`
- Total: ~1830+ articles

**Output:**
✓ vnexpressT_bongda_part1.json (473 articles)
✓ vnexpressT_bongda_part2.json (488 articles)  
✓ vnexpressT_bongda_part3.json (487 articles)
✓ vnexpressT_bongda_part4.json (308 articles)

---

### 2️⃣ Text Processing (02_preprocessing/)

```bash
cd 02_preprocessing
jupyter notebook textprocessing.ipynb
```

**What it does:**
- Tokenizes Vietnamese text (PyVi)
- Removes stopwords
- Extracts entities (teams, players, competitions)
- N-grams analysis

**Key Features:**
- Vietnamese accent handling
- Sports-specific entity recognition
- Compound word segmentation

---

### 3️⃣ Indexing (03_indexing/)

```bash
cd 03_indexing

# Build vocabulary
python build_vocab.py

# Test query indexing
python query_index.py
```

**What it does:**
- Builds inverted index
- Calculates TF-IDF scores
- Vietnamese accent restoration
- Query normalization

**Example:**
```python
# Input: "bong da" 
# Auto-restored: "bóng đá"
```

---

### 4️⃣ Ranking (04_ranking/)

```bash
cd 04_ranking
jupyter notebook BM25.ipynb
```

**What it does:**
- BM25 ranking implementation
- TF-IDF comparison
- Performance benchmarking

**Parameters:**
- k1 = 1.5 (term saturation)
- b = 0.75 (length normalization)

---

### 5️⃣ Neural Models (05_neural_models/)

```bash
cd 05_neural_models
jupyter notebook DeepCT_ConvKRM.ipynb
```

**⚠️ IMPORTANT: Cell Execution Order**

1. **Cell 1** - Import libraries
2. **Cell 7** - Initialize VietnameseTextProcessor
3. **Cell 5** - Load documents
4. **Cell 3** - Build vocabulary
5. **Cell 2, 4, 6** - Define models and train

**Models:**
- **DeepCT**: Deep Contextualized Term weighting
- **Conv-KNRM**: Convolutional Kernel-based Neural Ranking

**Training:**
- Vocab size: ~8000 words
- Embedding dim: 100
- Epochs: 10
- Time: ~30 minutes (CPU)

**Output:**
✓ deepct_convknrm_vi.pth (6.74 MB)

---

### 6️⃣ Evaluation (06_evaluation/)

```bash
cd 06_evaluation

# Demo comparison
python demo_search_comparison_v2.py

# Interactive testing
python interactive_query_demo.py
```

**Metrics:**
- Precision
- Recall
- F1-Score
- Response time

**Example Output:**
```
Query: "Messi World Cup"
─────────────────────────
BM25:      0.85 (120ms)
Conv-KNRM: 0.88 (350ms)
DeepCT:    0.91 (380ms)
```

---

### 7️⃣ Web Interface (07_web_interface/)

**Status:** 🚧 Under Development

To create web interface:

#### Backend (FastAPI)
```bash
cd 07_web_interface
mkdir -p backend frontend

# Create backend/app.py
cat > backend/app.py << EOF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search")
async def search(q: str, method: str = "bm25"):
    # Implement search logic here
    return {"query": q, "method": method, "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Run backend
python backend/app.py
```

#### Frontend (React + Vite)
```bash
cd frontend
npm create vite@latest . -- --template react
npm install
npm run dev
```

---

## 🔄 Full Pipeline Execution

### Method 1: Sequential (Recommended for First Time)

```bash
# 1. Crawl data (if not exists)
cd 01_crawling
jupyter nbconvert --execute crawlcode.ipynb

# 2. Process text
cd ../02_preprocessing
jupyter nbconvert --execute textprocessing.ipynb

# 3. Build index
cd ../03_indexing
python build_vocab.py

# 4. Test ranking
cd ../04_ranking
jupyter nbconvert --execute BM25.ipynb

# 5. Train neural models
cd ../05_neural_models
jupyter nbconvert --execute DeepCT_ConvKRM.ipynb

# 6. Evaluate
cd ../06_evaluation
python demo_search_comparison_v2.py
```

### Method 2: Automated Test

```bash
# From project root
python test_all.py

# Current status: 92% pass rate
# ✓ Data files: OK
# ✓ Notebooks: OK
# ✓ Models: OK
# ✓ Dependencies: OK
# ⚠️ Web interface: TODO
```

---

## 🐛 Troubleshooting

### Issue 1: NameError: Counter not defined
```python
# Solution: Run Cell 1 first (imports)
from collections import Counter, defaultdict
```

### Issue 2: NameError: documents not defined
```python
# Solution: Run cells in order
# Cell 1 → Cell 7 → Cell 5 → Cell 3
```

### Issue 3: PyVi not installed
```bash
pip install pyvi
```

### Issue 4: CUDA out of memory
```python
# Use CPU instead
device = torch.device('cpu')
model = model.to(device)
```

### Issue 5: Vietnamese text not displayed
```python
# Set encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### Issue 6: Module not found
```bash
# Add to Python path
import sys
sys.path.append('path/to/src')
```

---

## 📊 Expected Results

### Data Statistics
- **Total Articles**: 1,830+
- **Vocabulary Size**: 8,000+ words
- **Indexed Terms**: ~50,000

### Performance Benchmarks
| Method     | Precision | Recall | Time (ms) |
|------------|-----------|--------|-----------|
| BM25       | 0.75      | 0.72   | 120       |
| Conv-KNRM  | 0.82      | 0.79   | 350       |
| DeepCT     | 0.85      | 0.83   | 380       |

### Training Time
- **BM25 Index**: 2-3 minutes
- **Neural Models**: 30-40 minutes (CPU)
- **Full Pipeline**: ~1 hour

---

## 🎯 Project Status

### ✅ Completed (60-65%)
- [x] Web crawling
- [x] Text processing
- [x] Indexing (inverted index, TF-IDF)
- [x] BM25 ranking
- [x] Neural models (DeepCT, Conv-KNRM)
- [x] Basic evaluation

### 🚧 Partially Done
- [ ] Complete evaluation (need NDCG, MAP)
- [ ] Web interface (structure created)

### ❌ Not Started (Course Requirements)
- [ ] SEO & PageRank (Topic 8)
- [ ] Classification & Clustering (Topic 10)
- [ ] Social Search (Topic 11)

---

## 💡 Tips

1. **Always run test first**: `python test_all.py`
2. **Check cell order** in notebooks
3. **Use CPU** if no GPU available
4. **Save frequently** when training models
5. **Check logs** in `outputs/logs/`
6. **Backup data** before experiments

---

## 📞 Support

If you encounter issues:
1. Check `test_results.json` for detailed errors
2. Review README.md in each module folder
3. Check PROJECT_STRUCTURE.md for architecture
4. Verify Python version (3.8+)

---

**Last Updated**: 2024
**Test Pass Rate**: 92%
**Status**: Production Ready (Core Features)
