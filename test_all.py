"""
Master Testing Script - Test toàn bộ Search Engine Project
Chạy tất cả modules và báo cáo kết quả
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

BASE_DIR = Path(__file__).parent

class ProjectTester:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        self.passed = 0
        self.failed = 0
        self.skipped = 0
    
    def print_header(self, text):
        print(f"\n{'='*80}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{'='*80}\n")
    
    def print_test(self, name, status, message=""):
        symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⊘"
        color = GREEN if status == "PASS" else RED if status == "FAIL" else YELLOW
        
        print(f"{color}{symbol} {name}{RESET}")
        if message:
            print(f"  {message}")
        
        self.results["tests"].append({
            "name": name,
            "status": status,
            "message": message
        })
        
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1
    
    def test_file_exists(self, filepath, description):
        """Test if a file exists"""
        path = BASE_DIR / filepath
        if path.exists():
            self.print_test(description, "PASS", f"File: {filepath}")
            return True
        else:
            self.print_test(description, "FAIL", f"Missing: {filepath}")
            return False
    
    def test_folder_exists(self, folderpath, description):
        """Test if a folder exists"""
        path = BASE_DIR / folderpath
        if path.exists() and path.is_dir():
            self.print_test(description, "PASS", f"Folder: {folderpath}")
            return True
        else:
            self.print_test(description, "FAIL", f"Missing folder: {folderpath}")
            return False
    
    def test_python_imports(self, module_name, description):
        """Test if Python modules can be imported"""
        try:
            __import__(module_name)
            self.print_test(description, "PASS", f"Module: {module_name}")
            return True
        except ImportError as e:
            self.print_test(description, "FAIL", f"Cannot import {module_name}: {e}")
            return False
    
    def test_data_files(self):
        """Test data files"""
        self.print_header("📥 TESTING DATA FILES")
        
        data_files = [
            "vnexpressT_bongda_part1.json",
            "vnexpressT_bongda_part2.json",
            "vnexpressT_bongda_part3.json",
            "vnexpressT_bongda_part4.json"
        ]
        
        for file in data_files:
            path = BASE_DIR / "data" / "raw" / file
            if self.test_file_exists(f"data/raw/{file}", f"Data file: {file}"):
                # Check if file is valid JSON
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.print_test(f"Valid JSON: {file}", "PASS", f"Items: {len(data)}")
                except Exception as e:
                    self.print_test(f"Valid JSON: {file}", "FAIL", str(e))
    
    def test_vocabulary(self):
        """Test vocabulary file"""
        self.print_header("📚 TESTING VOCABULARY")
        
        vocab_path = BASE_DIR / "data" / "processed" / "vocab.txt"
        if self.test_file_exists("data/processed/vocab.txt", "Vocabulary file"):
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = f.readlines()
                self.print_test("Vocabulary size", "PASS", f"Words: {len(vocab)}")
            except Exception as e:
                self.print_test("Read vocabulary", "FAIL", str(e))
    
    def test_python_dependencies(self):
        """Test Python dependencies"""
        self.print_header("🔧 TESTING PYTHON DEPENDENCIES")
        
        dependencies = [
            ("torch", "PyTorch for neural models"),
            ("fastapi", "FastAPI for web backend"),
            ("pyvi", "PyVi for Vietnamese NLP"),
            ("pandas", "Pandas for data processing"),
            ("numpy", "NumPy for numerical operations"),
            ("tqdm", "TQDM for progress bars"),
            ("pymongo", "PyMongo for MongoDB")
        ]
        
        for module, desc in dependencies:
            self.test_python_imports(module, desc)
    
    def test_modules(self):
        """Test each module folder"""
        self.print_header("📦 TESTING PROJECT MODULES")
        
        modules = [
            ("01_crawling", "Crawling module"),
            ("02_preprocessing", "Preprocessing module"),
            ("03_indexing", "Indexing module"),
            ("04_ranking", "Ranking module"),
            ("05_neural_models", "Neural models module"),
            ("06_evaluation", "Evaluation module"),
            ("07_web_interface", "Web interface module")
        ]
        
        for folder, desc in modules:
            if self.test_folder_exists(folder, desc):
                # Check if README exists
                readme_path = BASE_DIR / folder / "README.md"
                if readme_path.exists():
                    self.print_test(f"{folder} README", "PASS", "Documentation exists")
                else:
                    self.print_test(f"{folder} README", "SKIP", "No documentation")
    
    def test_notebooks(self):
        """Test Jupyter notebooks"""
        self.print_header("📓 TESTING JUPYTER NOTEBOOKS")
        
        notebooks = [
            ("01_crawling/crawlcode.ipynb", "Crawling notebook"),
            ("02_preprocessing/textprocessing.ipynb", "Text processing notebook"),
            ("04_ranking/BM25.ipynb", "BM25 ranking notebook"),
            ("05_neural_models/DeepCT_ConvKRM.ipynb", "DeepCT Conv-KNRM notebook")
        ]
        
        for notebook, desc in notebooks:
            path = BASE_DIR / notebook
            if self.test_file_exists(notebook, desc):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        nb_data = json.load(f)
                    cells = len(nb_data.get('cells', []))
                    self.print_test(f"{notebook} structure", "PASS", f"Cells: {cells}")
                except Exception as e:
                    self.print_test(f"{notebook} structure", "FAIL", str(e))
    
    def test_model_weights(self):
        """Test trained model weights"""
        self.print_header("🤖 TESTING MODEL WEIGHTS")
        
        model_path = BASE_DIR / "05_neural_models" / "deepct_convknrm_vi.pth"
        if self.test_file_exists("05_neural_models/deepct_convknrm_vi.pth", "Trained model weights"):
            size_mb = model_path.stat().st_size / (1024 * 1024)
            self.print_test("Model size", "PASS", f"{size_mb:.2f} MB")
    
    def test_web_interface(self):
        """Test web interface structure"""
        self.print_header("🌐 TESTING WEB INTERFACE")
        
        # Check backend
        backend_files = [
            ("07_web_interface/web/backend/app.py", "FastAPI backend"),
            ("07_web_interface/web/backend/requirements.txt", "Backend requirements")
        ]
        
        for file, desc in backend_files:
            self.test_file_exists(file, desc)
        
        # Check frontend
        frontend_files = [
            ("07_web_interface/web/frontend/package.json", "Frontend package.json"),
            ("07_web_interface/web/frontend/vite.config.js", "Vite configuration")
        ]
        
        for file, desc in frontend_files:
            self.test_file_exists(file, desc)
    
    def test_src_modules(self):
        """Test src/ modules"""
        self.print_header("📂 TESTING SRC MODULES")
        
        src_folders = [
            ("src/crawling", "Crawling utilities"),
            ("src/preprocessing", "Preprocessing utilities"),
            ("src/indexing", "Indexing utilities"),
            ("src/ranking", "Ranking utilities"),
            ("src/query", "Query utilities")
        ]
        
        for folder, desc in src_folders:
            self.test_folder_exists(folder, desc)
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed + self.skipped
        
        self.print_header("📊 TEST SUMMARY")
        
        print(f"{GREEN}✓ Passed:  {self.passed}/{total}{RESET}")
        print(f"{RED}✗ Failed:  {self.failed}/{total}{RESET}")
        print(f"{YELLOW}⊘ Skipped: {self.skipped}/{total}{RESET}")
        
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"\n📈 Pass Rate: {pass_rate:.1f}%\n")
        
        if self.failed == 0:
            print(f"{GREEN}🎉 ALL CRITICAL TESTS PASSED!{RESET}")
        else:
            print(f"{RED}⚠️  SOME TESTS FAILED - Check details above{RESET}")
        
        # Save results to file
        results_path = BASE_DIR / "test_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detailed results saved to: test_results.json\n")
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_header("🚀 STARTING PROJECT TESTS")
        
        # Test structure
        self.test_modules()
        
        # Test data
        self.test_data_files()
        self.test_vocabulary()
        
        # Test notebooks
        self.test_notebooks()
        
        # Test models
        self.test_model_weights()
        
        # Test dependencies
        self.test_python_dependencies()
        
        # Test src modules
        self.test_src_modules()
        
        # Test web interface
        self.test_web_interface()
        
        # Print summary
        self.print_summary()

def main():
    print("\n" + "="*80)
    print(f"{BLUE}🔬 SEARCH ENGINE PROJECT - COMPREHENSIVE TESTING{RESET}")
    print("="*80)
    
    tester = ProjectTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
