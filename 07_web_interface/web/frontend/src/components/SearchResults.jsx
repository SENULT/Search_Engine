import './SearchResults.css'

function SearchResults({ results, query }) {
  if (results.length === 0) {
    return (
      <div className="no-results">
        <h2>Không tìm thấy kết quả cho "{query}"</h2>
        <p>Thử tìm kiếm với từ khóa khác hoặc kiểm tra chính tả.</p>
      </div>
    )
  }

  return (
    <div className="search-results">
      <div className="results-info">
        Khoảng {results.length} kết quả cho <strong>"{query}"</strong>
      </div>

      {results.map((result, index) => (
        <div key={index} className="result-item">
          {/* Method badge */}
          <span className={`method-badge ${result.method.toLowerCase().replace(/[- ]/g, '')}`}>
            {result.method}
          </span>

          {/* Title */}
          <a href={result.url || '#'} className="result-title" target="_blank" rel="noopener noreferrer">
            {result.title}
          </a>

          {/* URL */}
          {result.url && (
            <div className="result-url">
              {result.url}
            </div>
          )}

          {/* Content snippet with highlighted query terms */}
          <p 
            className="result-snippet"
            dangerouslySetInnerHTML={{ 
              __html: result.snippet || result.content || result.summary || 'Không có nội dung' 
            }}
          />

          {/* Meta info */}
          <div className="result-meta">
            {result.date && <span className="result-date">📅 {result.date}</span>}
            <span className="result-score">Score: {result.score.toFixed(3)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

export default SearchResults
