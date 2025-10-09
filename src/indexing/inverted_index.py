"""
INVERTED INDEX BUILDER FOR SEARCH ENGINE

Xây dựng Inverted Index từ preprocessed text và lưu vào database/files.

Features:
1. Build Inverted Index from preprocessed documents
2. Calculate TF-IDF scores
3. Store index in MongoDB and JSON files
4. Support incremental indexing
5. Query optimization
"""

import os
import json
import math
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Set
from tqdm import tqdm
import pickle

# MongoDB
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

# Import từ text processing
import sys
sys.path.append(os.path.dirname(__file__))

load_dotenv()


class InvertedIndex:
    """
    Inverted Index với TF-IDF scoring
    
    Structure:
    {
        'term1': {
            'df': 10,  # document frequency
            'postings': {
                'doc_id1': {'tf': 5, 'positions': [1, 5, 10, 15, 20]},
                'doc_id2': {'tf': 3, 'positions': [2, 8, 15]},
                ...
            }
        },
        'term2': {...}
    }
    """
    
    def __init__(self):
        self.index = defaultdict(lambda: {'df': 0, 'postings': {}})
        self.doc_lengths = {}  # {doc_id: length}
        self.total_docs = 0
        self.avg_doc_length = 0
        self.vocabulary = set()
        
    def add_document(self, doc_id: str, tokens: List[str], store_positions: bool = True):
        """
        Thêm document vào inverted index
        
        Args:
            doc_id: ID của document
            tokens: List các tokens đã preprocessing
            store_positions: Có lưu vị trí của term không
        """
        # Tính term frequency và positions
        term_positions = defaultdict(list)
        
        for position, term in enumerate(tokens):
            term_positions[term].append(position)
        
        # Cập nhật index
        for term, positions in term_positions.items():
            tf = len(positions)
            
            # Nếu term chưa xuất hiện trong document này
            if doc_id not in self.index[term]['postings']:
                self.index[term]['df'] += 1  # Tăng document frequency
            
            # Lưu posting
            self.index[term]['postings'][doc_id] = {
                'tf': tf,
                'positions': positions if store_positions else []
            }
            
            self.vocabulary.add(term)
        
        # Lưu document length
        self.doc_lengths[doc_id] = len(tokens)
        self.total_docs += 1
        
    def calculate_idf(self, term: str) -> float:
        """
        Tính IDF (Inverse Document Frequency)
        IDF = log(N / df)
        """
        df = self.index[term]['df']
        if df == 0:
            return 0
        return math.log(self.total_docs / df)
    
    def calculate_tf_idf(self, term: str, doc_id: str) -> float:
        """
        Tính TF-IDF score
        TF-IDF = TF * IDF
        """
        if term not in self.index or doc_id not in self.index[term]['postings']:
            return 0
        
        tf = self.index[term]['postings'][doc_id]['tf']
        idf = self.calculate_idf(term)
        
        return tf * idf
    
    def calculate_bm25(self, term: str, doc_id: str, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Tính BM25 score (cải tiến của TF-IDF)
        
        BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        """
        if term not in self.index or doc_id not in self.index[term]['postings']:
            return 0
        
        tf = self.index[term]['postings'][doc_id]['tf']
        idf = self.calculate_idf(term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        
        if self.avg_doc_length == 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 1
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / self.avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def get_postings(self, term: str) -> Dict:
        """Lấy postings list của một term"""
        return self.index.get(term, {'df': 0, 'postings': {}})
    
    def search(self, query_terms: List[str], method: str = 'bm25', top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Tìm kiếm documents liên quan đến query
        
        Args:
            query_terms: List các terms trong query (đã preprocessing)
            method: 'tfidf' hoặc 'bm25'
            top_k: Số lượng kết quả trả về
            
        Returns:
            List[(doc_id, score)] sorted by score descending
        """
        # Tính score cho từng document
        doc_scores = defaultdict(float)
        
        for term in query_terms:
            if term not in self.index:
                continue
            
            postings = self.index[term]['postings']
            
            for doc_id in postings.keys():
                if method == 'bm25':
                    score = self.calculate_bm25(term, doc_id)
                else:  # tfidf
                    score = self.calculate_tf_idf(term, doc_id)
                
                doc_scores[doc_id] += score
        
        # Sắp xếp theo score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về index"""
        return {
            'total_documents': self.total_docs,
            'vocabulary_size': len(self.vocabulary),
            'avg_doc_length': self.avg_doc_length if self.avg_doc_length > 0 else sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0,
            'total_postings': sum(len(term_data['postings']) for term_data in self.index.values()),
            'index_size_terms': len(self.index)
        }
    
    def to_dict(self) -> Dict:
        """Convert index sang dictionary để lưu file"""
        return {
            'index': dict(self.index),
            'doc_lengths': self.doc_lengths,
            'total_docs': self.total_docs,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary': list(self.vocabulary)
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Load index từ dictionary"""
        inv_index = cls()
        inv_index.index = defaultdict(lambda: {'df': 0, 'postings': {}}, data['index'])
        inv_index.doc_lengths = data['doc_lengths']
        inv_index.total_docs = data['total_docs']
        inv_index.avg_doc_length = data['avg_doc_length']
        inv_index.vocabulary = set(data['vocabulary'])
        return inv_index


class IndexBuilder:
    """
    Builder để xây dựng và lưu Inverted Index
    """
    
    def __init__(self, mongo_uri: str = None, db_name: str = "vnexpress_db"):
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.db_name = db_name
        self.client = None
        self.db = None
        self.inverted_index = InvertedIndex()
        
    def connect_database(self):
        """Kết nối MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, tls=True, tlsCAFile=certifi.where())
            self.db = self.client[self.db_name]
            print(f"✓ Đã kết nối database: {self.db_name}")
            return True
        except Exception as e:
            print(f"✗ Lỗi kết nối database: {e}")
            return False
    
    def build_index_from_collection(self, collection_name: str, token_field: str = 'filtered_tokens', limit: int = None):
        """
        Xây dựng index từ MongoDB collection
        
        Args:
            collection_name: Tên collection chứa preprocessed data
            token_field: Field chứa tokens (filtered_tokens hoặc stemmed_tokens)
            limit: Giới hạn số documents (None = all)
        """
        if not self.db:
            if not self.connect_database():
                return False
        
        collection = self.db[collection_name]
        
        # Đếm documents
        total_docs = collection.count_documents({})
        if limit:
            total_docs = min(total_docs, limit)
        
        print(f"\n🔨 BẮT ĐẦU XÂY DỰNG INVERTED INDEX")
        print(f"Collection: {collection_name}")
        print(f"Total documents: {total_docs}")
        print("="*80)
        
        # Lấy documents
        query = collection.find().limit(limit) if limit else collection.find()
        
        processed_count = 0
        error_count = 0
        
        for doc in tqdm(query, desc="Building index", total=total_docs):
            try:
                doc_id = str(doc.get('_id', ''))
                tokens = doc.get(token_field, [])
                
                if not tokens:
                    # Nếu không có filtered_tokens, thử lấy tokens
                    tokens = doc.get('tokens', [])
                
                if tokens:
                    self.inverted_index.add_document(doc_id, tokens, store_positions=True)
                    processed_count += 1
                    
            except Exception as e:
                error_count += 1
                print(f"\nLỗi xử lý document {doc.get('_id', 'unknown')}: {e}")
        
        print(f"\n✓ Hoàn thành!")
        print(f"  - Processed: {processed_count}")
        print(f"  - Errors: {error_count}")
        
        # In thống kê
        self.print_statistics()
        
        return True
    
    def build_index_from_json_files(self, json_files: List[str], token_field: str = 'filtered_tokens'):
        """
        Xây dựng index từ các JSON files (preprocessed data)
        
        Args:
            json_files: List đường dẫn tới JSON files
            token_field: Field chứa tokens
        """
        print(f"\n🔨 BẮT ĐẦU XÂY DỰNG INVERTED INDEX TỪ JSON FILES")
        print(f"Số files: {len(json_files)}")
        print("="*80)
        
        total_docs = 0
        
        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"⚠️ File không tồn tại: {json_file}")
                continue
            
            print(f"\nĐang xử lý: {json_file}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Nếu data là list
            if isinstance(data, list):
                docs = data
            # Nếu data là dict với key 'documents' hoặc tương tự
            elif isinstance(data, dict):
                docs = data.get('documents', [data])
            else:
                docs = [data]
            
            for doc in tqdm(docs, desc=f"Processing {os.path.basename(json_file)}"):
                try:
                    doc_id = doc.get('doc_id', str(total_docs))
                    tokens = doc.get(token_field, [])
                    
                    if not tokens:
                        tokens = doc.get('tokens', [])
                    
                    if tokens:
                        self.inverted_index.add_document(doc_id, tokens, store_positions=True)
                        total_docs += 1
                        
                except Exception as e:
                    print(f"Lỗi xử lý document: {e}")
        
        print(f"\n✓ Hoàn thành! Đã xử lý {total_docs} documents")
        self.print_statistics()
        
        return True
    
    def save_index_to_mongodb(self, collection_name: str = "inverted_index"):
        """
        Lưu inverted index vào MongoDB
        
        Structure:
        - Collection 'inverted_index_terms': Lưu index terms
        - Collection 'inverted_index_meta': Lưu metadata
        """
        if not self.db:
            if not self.connect_database():
                return False
        
        print(f"\n💾 ĐANG LƯU INDEX VÀO MONGODB...")
        
        # 1. Lưu metadata
        meta_collection = self.db[f"{collection_name}_meta"]
        meta_collection.delete_many({})  # Xóa dữ liệu cũ
        
        metadata = {
            'created_at': datetime.now(),
            'total_docs': self.inverted_index.total_docs,
            'vocabulary_size': len(self.inverted_index.vocabulary),
            'avg_doc_length': self.inverted_index.avg_doc_length,
            'doc_lengths': self.inverted_index.doc_lengths
        }
        
        meta_collection.insert_one(metadata)
        print(f"✓ Đã lưu metadata vào {collection_name}_meta")
        
        # 2. Lưu index terms (batch insert)
        terms_collection = self.db[f"{collection_name}_terms"]
        terms_collection.delete_many({})  # Xóa dữ liệu cũ
        
        batch_size = 1000
        batch = []
        
        for term, term_data in tqdm(self.inverted_index.index.items(), desc="Saving terms"):
            batch.append({
                'term': term,
                'df': term_data['df'],
                'postings': term_data['postings']
            })
            
            if len(batch) >= batch_size:
                terms_collection.insert_many(batch)
                batch = []
        
        # Insert remaining
        if batch:
            terms_collection.insert_many(batch)
        
        # Tạo index cho term field
        terms_collection.create_index('term', unique=True)
        
        print(f"✓ Đã lưu {len(self.inverted_index.index)} terms vào {collection_name}_terms")
        print(f"✓ Đã tạo index cho field 'term'")
        
        return True
    
    def save_index_to_json(self, output_dir: str = "outputs", filename: str = None):
        """
        Lưu inverted index vào JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inverted_index_{timestamp}.json"
        
        filepath = os.path.join(output_dir, filename)
        
        print(f"\n💾 ĐANG LƯU INDEX VÀO JSON FILE...")
        
        index_data = self.inverted_index.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ Đã lưu index vào: {filepath}")
        print(f"  File size: {file_size:.2f} MB")
        
        return filepath
    
    def save_index_to_pickle(self, output_dir: str = "outputs", filename: str = None):
        """
        Lưu inverted index vào Pickle file (nhanh hơn JSON)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inverted_index_{timestamp}.pkl"
        
        filepath = os.path.join(output_dir, filename)
        
        print(f"\n💾 ĐANG LƯU INDEX VÀO PICKLE FILE...")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ Đã lưu index vào: {filepath}")
        print(f"  File size: {file_size:.2f} MB")
        
        return filepath
    
    @staticmethod
    def load_index_from_json(filepath: str) -> InvertedIndex:
        """Load inverted index từ JSON file"""
        print(f"📂 Đang load index từ: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        inv_index = InvertedIndex.from_dict(data)
        print(f"✓ Đã load index thành công!")
        return inv_index
    
    @staticmethod
    def load_index_from_pickle(filepath: str) -> InvertedIndex:
        """Load inverted index từ Pickle file"""
        print(f"📂 Đang load index từ: {filepath}")
        
        with open(filepath, 'rb') as f:
            inv_index = pickle.load(f)
        
        print(f"✓ Đã load index thành công!")
        return inv_index
    
    def print_statistics(self):
        """In thống kê về index"""
        stats = self.inverted_index.get_statistics()
        
        print("\n" + "="*80)
        print("📊 THỐNG KÊ INVERTED INDEX")
        print("="*80)
        print(f"Total documents: {stats['total_documents']:,}")
        print(f"Vocabulary size: {stats['vocabulary_size']:,}")
        print(f"Average document length: {stats['avg_doc_length']:.1f}")
        print(f"Total postings: {stats['total_postings']:,}")
        print(f"Index size (terms): {stats['index_size_terms']:,}")
        
        # Top 10 terms với df cao nhất
        top_terms = sorted(
            [(term, data['df']) for term, data in self.inverted_index.index.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print(f"\nTop 10 terms (highest document frequency):")
        for term, df in top_terms:
            print(f"  '{term}': {df} documents")


def main():
    """Main function để demo"""
    print("="*80)
    print("INVERTED INDEX BUILDER")
    print("="*80)
    
    # Khởi tạo builder
    builder = IndexBuilder()
    
    # Option 1: Build từ MongoDB collection (preprocessed data)
    # builder.build_index_from_collection(
    #     collection_name="preprocessed_documents",
    #     token_field="filtered_tokens",
    #     limit=1000
    # )
    
    # Option 2: Build từ JSON files
    json_files = [
        "outputs/processed_vnexpress_20241009_123456.json"
    ]
    # builder.build_index_from_json_files(json_files)
    
    # Save index
    # builder.save_index_to_mongodb("inverted_index")
    # builder.save_index_to_json("outputs")
    # builder.save_index_to_pickle("outputs")
    
    print("\n✓ HOÀN THÀNH!")


if __name__ == "__main__":
    main()
