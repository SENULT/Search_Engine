"""
Script to organize Search Engine project structure
Tự động sắp xếp files vào folders theo cấu trúc chuẩn
"""

import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent
print(f"📁 Base directory: {BASE_DIR}\n")

# Định nghĩa cấu trúc folders
FOLDER_STRUCTURE = {
    "01_crawling": [
        "crawlcode.ipynb",
        "updata.py"
    ],
    "02_preprocessing": [
        "textprocessing.ipynb",
        "textprocessing_S1.ipynb",
        "textprocessing_s2ranking.ipynb"
    ],
    "03_indexing": [
        "build_vocab.py",
        "query_index.py"
    ],
    "04_ranking": [
        "BM25.ipynb",
        "compare.ipynb"
    ],
    "05_neural_models": [
        "DeepCT_ConvKRM.ipynb",
        "neural_ranking_models.ipynb",
        "deepct_convknrm_vi.pth"
    ],
    "06_evaluation": [
        "demo_search_comparison_v2.py",
        "interactive_query_demo.py",
        "bm25_vs_neural_comparison.png"
    ],
    "07_web_interface": [
        "web/"  # Folder đã tồn tại
    ]
}

# Files cần move vào data/raw/
DATA_FILES = [
    "vnexpressT_bongda_part1.json",
    "vnexpressT_bongda_part2.json",
    "vnexpressT_bongda_part3.json",
    "vnexpressT_bongda_part4.json"
]

# Files cần move vào data/processed/
PROCESSED_FILES = [
    "vocab.txt"
]

# Files giữ lại ở root
ROOT_FILES = [
    "README.md",
    "FINAL_TEAM_PACKAGE.md",
    "PROJECT_STRUCTURE.md",
    "requirements.txt",
    ".gitignore",
    "ai2021.pdf"
]

def create_folder_structure():
    """Tạo các folders nếu chưa tồn tại"""
    print("🔨 Creating folder structure...")
    
    for folder_name in FOLDER_STRUCTURE.keys():
        folder_path = BASE_DIR / folder_name
        folder_path.mkdir(exist_ok=True)
        print(f"  ✓ Created: {folder_name}/")
    
    # Tạo data folders
    (BASE_DIR / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Created: data/raw/")
    print(f"  ✓ Created: data/processed/")
    
    # Tạo docs folder
    (BASE_DIR / "docs").mkdir(exist_ok=True)
    print(f"  ✓ Created: docs/")
    
    print()

def move_files():
    """Di chuyển files vào các folders tương ứng"""
    print("📦 Moving files to organized structure...\n")
    
    moved_count = 0
    
    # Move files theo FOLDER_STRUCTURE
    for folder_name, files in FOLDER_STRUCTURE.items():
        print(f"📂 Processing {folder_name}/")
        for file in files:
            if file.endswith("/"):  # Skip folders
                continue
            
            src = BASE_DIR / file
            dst = BASE_DIR / folder_name / file
            
            if src.exists():
                # Kiểm tra nếu file đã tồn tại ở destination
                if dst.exists():
                    print(f"  ⚠️  Already exists: {file}")
                else:
                    shutil.move(str(src), str(dst))
                    print(f"  ✓ Moved: {file}")
                    moved_count += 1
            else:
                print(f"  ✗ Not found: {file}")
        print()
    
    # Move data files
    print(f"📂 Processing data/raw/")
    for file in DATA_FILES:
        src = BASE_DIR / file
        dst = BASE_DIR / "data" / "raw" / file
        
        if src.exists():
            if dst.exists():
                print(f"  ⚠️  Already exists: {file}")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved: {file}")
                moved_count += 1
        else:
            print(f"  ✗ Not found: {file}")
    print()
    
    # Move processed files
    print(f"📂 Processing data/processed/")
    for file in PROCESSED_FILES:
        src = BASE_DIR / file
        dst = BASE_DIR / "data" / "processed" / file
        
        if src.exists():
            if dst.exists():
                print(f"  ⚠️  Already exists: {file}")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ Moved: {file}")
                moved_count += 1
        else:
            print(f"  ✗ Not found: {file}")
    print()
    
    print(f"✅ Total files moved: {moved_count}\n")

def create_readme_files():
    """Tạo README.md cho từng folder"""
    print("📝 Creating README files for each folder...\n")
    
    readmes = {
        "01_crawling/README.md": """# 📥 Crawling Module

## Files
- `crawlcode.ipynb`: Notebook để crawl dữ liệu từ VnExpress
- `updata.py`: Script upload data lên MongoDB

## Usage
1. Chạy `crawlcode.ipynb` để crawl data
2. Data được lưu vào `data/raw/`
3. Chạy `python updata.py` để upload lên MongoDB

## Output
- vnexpress_bongda_part1.json
- vnexpress_bongda_part2.json  
- vnexpress_bongda_part3.json
- vnexpress_bongda_part4.json

Total: ~1830+ articles
""",

        "02_preprocessing/README.md": """# 🔧 Text Processing Module

## Files
- `textprocessing.ipynb`: Basic Vietnamese text processing
- `textprocessing_S1.ipynb`: Advanced processing
- `textprocessing_s2ranking.ipynb`: Processing for ranking

## Features
- Vietnamese tokenization (PyVi)
- Stopwords removal
- Entity extraction (teams, players, competitions)
- N-grams analysis

## Usage
Chạy các notebooks theo thứ tự để xử lý text tiếng Việt.
""",

        "03_indexing/README.md": """# 📇 Indexing Module

## Files
- `build_vocab.py`: Build vocabulary từ corpus
- `query_index.py`: Query processing & inverted index

## Features
- Inverted index construction
- TF-IDF calculation
- Accent restoration
- Query normalization

## Usage
```bash
python build_vocab.py
python query_index.py
```
""",

        "04_ranking/README.md": """# 📊 Ranking Module

## Files
- `BM25.ipynb`: BM25 ranking implementation
- `compare.ipynb`: So sánh các phương pháp ranking

## Algorithms
- BM25 (Okapi BM25)
- TF-IDF
- Combined ranking

## Usage
Chạy các notebooks để test ranking algorithms.
""",

        "05_neural_models/README.md": """# 🤖 Neural Ranking Models

## Files
- `DeepCT_ConvKRM.ipynb`: DeepCT + Conv-KNRM implementation
- `neural_ranking_models.ipynb`: Other neural models
- `deepct_convknrm_vi.pth`: Trained model weights

## Models
1. **DeepCT**: Deep Contextualized Term weighting
2. **Conv-KNRM**: Convolutional Kernel-based Neural Ranking

## Training
Chạy `DeepCT_ConvKRM.ipynb` để train models.

## Performance
- Vocab size: ~8000 words
- Embedding dim: 100
- Training time: ~30 minutes (CPU)
""",

        "06_evaluation/README.md": """# 📈 Evaluation Module

## Files
- `demo_search_comparison_v2.py`: Demo so sánh search methods
- `interactive_query_demo.py`: Interactive query interface
- `bm25_vs_neural_comparison.png`: Comparison chart

## Metrics
- Precision
- Recall  
- F1-Score
- Response time

## Usage
```bash
python demo_search_comparison_v2.py
python interactive_query_demo.py
```
""",

        "07_web_interface/README.md": """# 🌐 Web Interface

## Structure
- `backend/`: FastAPI server
- `frontend/`: React + Vite application

## Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Features
- Google-like UI
- Multiple search methods (BM25, Conv-KNRM, DeepCT)
- Real-time search
- Result snippets

## Ports
- Backend: http://localhost:8000
- Frontend: http://localhost:5173
"""
    }
    
    for file_path, content in readmes.items():
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ Created: {file_path}")
        else:
            print(f"  ⚠️  Already exists: {file_path}")
    
    print()

def print_final_structure():
    """In ra cấu trúc folder cuối cùng"""
    print("="*80)
    print("📁 FINAL PROJECT STRUCTURE")
    print("="*80)
    print("""
Search_Engine/
├── 01_crawling/
├── 02_preprocessing/
├── 03_indexing/
├── 04_ranking/
├── 05_neural_models/
├── 06_evaluation/
├── 07_web_interface/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── src/
├── docs/
├── README.md
├── FINAL_TEAM_PACKAGE.md
└── PROJECT_STRUCTURE.md
    """)
    print("="*80)
    print("✅ Project organization completed!")
    print("📖 See PROJECT_STRUCTURE.md for details")
    print("="*80)

def main():
    """Main function"""
    print("\n" + "="*80)
    print("🚀 ORGANIZING SEARCH ENGINE PROJECT")
    print("="*80 + "\n")
    
    try:
        # Step 1: Create folders
        create_folder_structure()
        
        # Step 2: Move files
        move_files()
        
        # Step 3: Create README files
        create_readme_files()
        
        # Step 4: Print final structure
        print_final_structure()
        
        print("\n🎉 Done! Your project is now organized.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
