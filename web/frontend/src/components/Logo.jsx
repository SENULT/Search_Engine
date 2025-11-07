import './Logo.css'

function Logo({ searched }) {
  return (
    <div className={`logo ${searched ? 'logo-small' : 'logo-large'}`}>
      <h1 className="logo-text">
        <span className="logo-char logo-v">V</span>
        <span className="logo-char logo-n">N</span>
        <span className="logo-char logo-f">F</span>
        <span className="logo-char logo-o">o</span>
        <span className="logo-char logo-o2">o</span>
        <span className="logo-char logo-t">t</span>
        <span className="logo-char logo-b">b</span>
        <span className="logo-char logo-a">a</span>
        <span className="logo-char logo-l">l</span>
        <span className="logo-char logo-l2">l</span>
      </h1>
      {!searched && (
        <p className="logo-subtitle">⚽ Tìm kiếm bóng đá Việt Nam</p>
      )}
    </div>
  )
}

export default Logo
