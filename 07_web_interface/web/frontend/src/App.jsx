import { useState, useEffect } from 'react'
import './App.css'
import SearchBar from './components/SearchBar'
import SearchResults from './components/SearchResults'
import Logo from './components/Logo'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [method, setMethod] = useState('all')
  const [stats, setStats] = useState(null)

  // API base URL
  const API_URL = 'http://localhost:8000'

  // Fetch stats on load
  useEffect(() => {
    fetch(`${API_URL}/stats`)
      .then(res => res.json())
      .then(data => setStats(data))
      .catch(err => console.error('Failed to fetch stats:', err))
  }, [])

  const handleSearch = async (searchQuery) => {
    if (!searchQuery.trim()) return

    setLoading(true)
    setSearched(true)
    setQuery(searchQuery)

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          method: method,
          top_k: 10
        })
      })

      const data = await response.json()
      setResults(data.results || [])
    } catch (error) {
      console.error('Search error:', error)
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      {/* Header with Logo and Search Bar */}
      <div className={`header ${searched ? 'header-small' : 'header-large'}`}>
        <Logo searched={searched} />
        
        <div className="search-container">
          <SearchBar 
            onSearch={handleSearch}
            initialValue={query}
            searched={searched}
          />
          
          {/* Search method selector */}
          <div className="method-selector">
            <label>
              <input
                type="radio"
                value="all"
                checked={method === 'all'}
                onChange={(e) => setMethod(e.target.value)}
              />
              All Methods
            </label>
            <label>
              <input
                type="radio"
                value="bm25"
                checked={method === 'bm25'}
                onChange={(e) => setMethod(e.target.value)}
              />
              BM25
            </label>
            <label>
              <input
                type="radio"
                value="neural"
                checked={method === 'neural'}
                onChange={(e) => setMethod(e.target.value)}
              />
              DeepCT+Conv-KNRM
            </label>
          </div>
        </div>

        {/* Stats */}
        {stats && !searched && (
          <div className="stats">
            <p>🗄️ {stats.total_articles.toLocaleString()} bài báo</p>
            <p>📚 {stats.vocab_size.toLocaleString()} từ vựng</p>
          </div>
        )}
      </div>

      {/* Popular Queries (only show before search) */}
      {!searched && (
        <div className="popular-queries">
          <h3>Truy vấn phổ biến:</h3>
          <div className="query-chips">
            {[
              'Đội tuyển Việt Nam',
              'Park Hang Seo',
              'Quang Hải',
              'V-League',
              'Công Phượng',
              'World Cup',
              'AFF Cup'
            ].map(q => (
              <button
                key={q}
                className="chip"
                onClick={() => handleSearch(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Search Results */}
      {searched && (
        <div className="results-container">
          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p>Đang tìm kiếm...</p>
            </div>
          ) : (
            <SearchResults 
              results={results}
              query={query}
            />
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>🔍 Vietnamese Football Search Engine</p>
          <p>Powered by Neural Ranking Models (BM25 + DeepCT+Conv-KNRM)</p>
          <p>⚽ Data: VnExpress Bóng Đá | 🇻🇳 Made with ❤️</p>
        </div>
      </footer>
    </div>
  )
}

export default App
