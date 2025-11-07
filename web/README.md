# 🔍 Vietnamese Football Search Engine

## Google-like Search Interface for Vietnamese Football News

Powered by Neural Ranking Models (BM25 + Conv-KNRM + DeepCT)

---

## 🚀 **TECH STACK**

### **Frontend:**
- ⚛️ React 18
- ⚡ Vite
- 🎨 CSS3 (Google-inspired design)

### **Backend:**
- 🐍 Python FastAPI
- 🧠 PyTorch Neural Models
- 🔍 BM25, Conv-KNRM, DeepCT

### **Data:**
- 📊 2000 Vietnamese football articles
- 📰 VnExpress Bóng Đá

---

## ⚡ **QUICK START**

### **1. Backend Setup (Python)**

```bash
cd web/backend

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Backend will run on: `http://localhost:8000`

### **2. Frontend Setup (React)**

```bash
cd web/frontend

# Install dependencies (if not already done)
npm install

# Run dev server
npm run dev
```

Frontend will run on: `http://localhost:5173`

---

## 🎯 **USAGE**

1. Open `http://localhost:5173` in your browser
2. Enter search query (e.g., "Park Hang Seo", "Quang Hải")
3. Select search method:
   - **All Methods**: Combine all 3 methods
   - **BM25**: Traditional statistical ranking
   - **Conv-KNRM**: Convolutional neural ranking
   - **DeepCT**: Deep contextualized term weighting
4. View results with scores and rankings!

---

## 📊 **FEATURES**

✅ **Google-like UI/UX**
- Clean, modern interface
- Smooth animations
- Responsive design

✅ **Multiple Search Methods**
- BM25 baseline
- Conv-KNRM neural ranking
- DeepCT context-aware search

✅ **Rich Results**
- Article titles
- Content snippets
- Relevance scores
- Method badges
- Source URLs

✅ **Popular Queries**
- Quick search suggestions
- Common football topics

---

## 🎨 **UI FEATURES**

🎨 **Google-inspired Design:**
- Colorful logo animation
- Clean search bar
- Card-based results
- Method badges (BM25, Conv-KNRM, DeepCT)
- Smooth transitions

📱 **Responsive:**
- Works on desktop, tablet, mobile
- Adaptive layout
- Touch-friendly

---

## 🔍 **SEARCH EXAMPLES**

Try these queries:
- "Đội tuyển Việt Nam"
- "Park Hang Seo"
- "Quang Hải ghi bàn"
- "V-League chuyển nhượng"
- "World Cup 2022"
- "bóng đá việt nam"

---

## 📈 **SEARCH METHODS COMPARISON**

| Method | Best For | Score Range |
|--------|----------|-------------|
| **BM25** | Exact keyword matching | 0-15 |
| **Conv-KNRM** | Semantic similarity | 0.3-0.6 |
| **DeepCT** | Context understanding | 1.2-1.5 |

---

## 🛠️ **DEVELOPMENT**

### **Backend API Endpoints:**

```
GET  /              - API info
GET  /health        - Health check
GET  /stats         - System stats
POST /search        - Search articles
GET  /suggestions   - Search suggestions
```

### **Frontend Structure:**

```
src/
├── App.jsx              # Main app component
├── components/
│   ├── SearchBar.jsx    # Search input
│   ├── SearchResults.jsx # Results display
│   └── Logo.jsx         # Animated logo
├── App.css              # Global styles
└── components/*.css     # Component styles
```

---

## 🎯 **PROJECT STRUCTURE**

```
web/
├── backend/
│   ├── app.py              # FastAPI server
│   └── requirements.txt    # Python deps
│
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   └── *.css
    ├── package.json
    └── vite.config.js
```

---

## 🔥 **FEATURES DEMO**

**Homepage:**
- Large centered logo
- Search bar
- Popular query chips
- Stats display

**Search Results:**
- Compact header with logo
- Method badges
- Article titles (clickable)
- Content snippets
- Relevance scores
- Source URLs

---

## 🚀 **DEPLOYMENT**

### **Backend:**
```bash
# Production server
uvicorn app:app --host 0.0.0.0 --port 8000
```

### **Frontend:**
```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 🎉 **READY TO USE!**

**🔥 Google-like search interface for Vietnamese football!**

**⚽ Powered by neural ranking models! 🇻🇳**

---

## 📞 **SUPPORT**

For issues or questions, check:
- Backend API: `http://localhost:8000/docs`
- Frontend: `http://localhost:5173`

**Happy Searching! 🔍⚽🇻🇳**
