import { useState } from 'react'
import './SearchBar.css'

function SearchBar({ onSearch, initialValue = '', searched = false }) {
  const [query, setQuery] = useState(initialValue)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query)
    }
  }

  return (
    <form className={`search-bar ${searched ? 'search-bar-small' : ''}`} onSubmit={handleSubmit}>
      <div className="search-input-container">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="search-input"
          placeholder="Tìm kiếm bóng đá Việt Nam..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus={!searched}
        />
        {query && (
          <button
            type="button"
            className="clear-button"
            onClick={() => setQuery('')}
          >
            ✕
          </button>
        )}
      </div>
      <button type="submit" className="search-button">
        Tìm kiếm
      </button>
    </form>
  )
}

export default SearchBar
