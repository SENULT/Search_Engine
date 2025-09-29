import os
import json
from pymongo import MongoClient
import schedule
import time
from datetime import datetime
from tqdm import tqdm 
from datetime import datetime, timezone

# MongoDB Atlas URI
MONGO_URI = "mongodb+srv://@cluster0.orljj0v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "vnexpress_db"
COLLECTION_NAME = "vnexpress_bongda"  # Gom tất cả vào 1 collection

# Thư mục chứa các file JSON
DATA_DIR = r"C:\Users\admin\Downloads\data_files"
os.makedirs(DATA_DIR, exist_ok=True)

def upload_files():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    if not files:
        print(f"[{datetime.now()}] Không có file JSON nào trong {DATA_DIR}")
        return

    total_inserted = 0
    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            tqdm.write(f"[{datetime.now()}] Đang upload {len(data)} documents từ {filename} ...")
            for doc in tqdm(data, desc=f"Uploading {filename}", unit="doc", ncols=80, leave=False):
                if "url" in doc and collection.find_one({"url": doc["url"]}):
                    continue
                doc["_uploaded_at"] = datetime.now(timezone.utc)
                doc["source_file"] = filename
                collection.insert_one(doc)
                total_inserted += 1
        else:
            if "url" not in data or not collection.find_one({"url": data["url"]}):
                data["_uploaded_at"] = datetime.utcnow()
                data["source_file"] = filename
                collection.insert_one(data)
                total_inserted += 1

    print(f"\n[{datetime.now()}] ✅ Upload hoàn tất. Tổng số document mới: {total_inserted}")

    client.close()

# ---- Chạy ngay khi start script ----
upload_files()

# ---- Đặt lịch tự động mỗi tuần vào 22:00 thứ 7 ----
schedule.every().saturday.at("22:00").do(upload_files)

print("Scheduler đang chạy... (Ctrl+C để dừng)")
while True:
    schedule.run_pending()
    time.sleep(300)
