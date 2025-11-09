# 🎯 **WEB APP SETUP GUIDE**

## ⚡ QUICK START (5 phút)

### **Bước 1: Setup Backend (2 phút)**

```powershell
# Di chuyển vào folder backend
cd d:\data\Search_Engine\web\backend

# Install Python dependencies
pip install fastapi uvicorn torch numpy pydantic python-multipart

# Run backend server
python app.py
```

**Backend sẽ chạy ở:** `http://localhost:8000`

---

### **Bước 2: Setup Frontend (3 phút)**

**Mở terminal mới:**

```powershell
# Di chuyển vào folder frontend
cd d:\data\Search_Engine\web\frontend

# Install Node dependencies (nếu chưa có)
npm install

# Run frontend dev server
npm run dev
```

**Frontend sẽ chạy ở:** `http://localhost:5173`

---

## 🌐 **SỬ DỤNG WEB APP**

1. **Mở trình duyệt:** `http://localhost:5173`
2. **Nhập truy vấn:** Ví dụ "Park Hang Seo"
3. **Chọn method:** All / BM25 / Conv-KNRM / DeepCT
4. **Xem kết quả!** 🎉

---

## 🔥 **DEMO QUERIES**

Thử các truy vấn này:
- "Đội tuyển Việt Nam"
- "Park Hang Seo"
- "Quang Hải ghi bàn"
- "V-League chuyển nhượng"
- "bóng đá việt nam"

---

## 🎨 **FEATURES**

✅ **Google-like Interface**
- Animated colorful logo
- Clean search bar
- Beautiful results cards
- Method badges (BM25/Conv-KNRM/DeepCT)

✅ **Smart Search**
- 3 neural ranking methods
- Real-time search
- Relevance scoring
- Popular queries

✅ **Responsive Design**
- Desktop, tablet, mobile
- Smooth animations
- Professional UI

---

## 📊 **API ENDPOINTS**

Backend API có các endpoints:

- `GET /` - API info
- `GET /health` - Health check  
- `GET /stats` - System statistics
- `POST /search` - Search articles
- `GET /suggestions` - Query suggestions

**API Docs:** `http://localhost:8000/docs`

---

## 🐛 **TROUBLESHOOTING**

### **Backend không chạy:**
```powershell
# Check Python version
python --version  # Cần Python 3.8+

# Reinstall dependencies
pip install -r requirements.txt

# Check data files
dir d:\data\Search_Engine\vnexpress_bongda_part*.json
```

### **Frontend không chạy:**
```powershell
# Check Node version
node --version  # Cần Node 16+, khuyến nghị 20.16+ hoặc 22.x LTS

# Nếu gặp lỗi "Vite requires Node 20.19+"
# Project đã dùng Vite 5.4.21 (stable) tương thích Node 20.16+

# Clear cache và reinstall
Remove-Item -Recurse -Force node_modules, package-lock.json
npm install

# Run again
npm run dev
```

### **CORS Error:**
- Backend đã config CORS cho phép tất cả origins
- Nếu vẫn lỗi, check backend có đang chạy không

---

## 🚀 **PRODUCTION DEPLOYMENT**

### **Backend:**
```powershell
# Run with uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### **Frontend:**
```powershell
# Build for production
npm run build

# Files sẽ ở folder dist/
# Deploy folder dist/ lên web server
```

---

## 📦 **FILES CẦN THIẾT**

Đảm bảo có đủ files:

**Backend:**
- ✅ `web/backend/app.py`
- ✅ `web/backend/requirements.txt`
- ✅ `vnexpress_bongda_part1-4.json` (ở folder gốc)

**Frontend:**
- ✅ `web/frontend/src/App.jsx`
- ✅ `web/frontend/src/components/*.jsx`
- ✅ `web/frontend/src/*.css`

---

## 🎉 **READY TO GO!**

**Web app của bạn đã sẵn sàng!**

**Google-like search cho bóng đá Việt Nam! ⚽🇻🇳**

**Have fun! 🚀**
