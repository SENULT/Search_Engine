"""
Database Utilities

MongoDB connection and operations
"""

import os
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """MongoDB Database Manager"""
    
    def __init__(self, mongo_uri: str = None, db_name: str = "vnexpress_db"):
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.db_name = db_name
        self.client = None
        self.db = None
    
    def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, tls=True, tlsCAFile=certifi.where())
            self.db = self.client[self.db_name]
            print(f"✓ Connected to database: {self.db_name}")
            return True
        except Exception as e:
            print(f"✗ Database connection error: {e}")
            return False
    
    def get_collection(self, collection_name: str):
        """Get a collection"""
        if not self.db:
            self.connect()
        return self.db[collection_name]
    
    def close(self):
        """Close connection"""
        if self.client:
            self.client.close()
            print("✓ Database connection closed")
