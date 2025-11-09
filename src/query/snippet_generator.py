"""
RESULT SNIPPETS AND HIGHLIGHTING
Generate search result snippets with keyword highlighting

Features:
- Extract relevant snippets around query terms
- Highlight matched keywords in context
- Generate summaries with ellipsis
- Support multi-field snippets (title, content, url)
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict


class SnippetGenerator:
    """Generate highlighted snippets for search results"""
    
    def __init__(self, 
                 snippet_length: int = 200,
                 context_window: int = 50,
                 max_snippets: int = 3):
        self.snippet_length = snippet_length
        self.context_window = context_window
        self.max_snippets = max_snippets
    
    def _find_matches(self, text: str, query_terms: List[str]) -> List[Tuple[int, int, str]]:
        """Find all positions of query terms in text"""
        matches = []
        text_lower = text.lower()
        
        for term in query_terms:
            term_lower = term.lower().replace('_', ' ')
            
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(term_lower, start)
                if pos == -1:
                    break
                matches.append((pos, pos + len(term_lower), term))
                start = pos + 1
        
        # Sort by position
        matches.sort(key=lambda x: x[0])
        return matches
    
    def _merge_overlapping(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping ranges"""
        if not ranges:
            return []
        
        ranges = sorted(ranges)
        merged = [ranges[0]]
        
        for current in ranges[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                # Overlapping, merge
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def _extract_snippet_windows(self, text: str, matches: List[Tuple[int, int, str]]) -> List[Tuple[int, int]]:
        """Extract windows around matches"""
        if not matches:
            return [(0, min(self.snippet_length, len(text)))]
        
        windows = []
        for start, end, term in matches:
            # Create window around match
            window_start = max(0, start - self.context_window)
            window_end = min(len(text), end + self.context_window)
            
            # Expand to word boundaries
            while window_start > 0 and not text[window_start-1].isspace():
                window_start -= 1
            while window_end < len(text) and not text[window_end].isspace():
                window_end += 1
            
            windows.append((window_start, window_end))
        
        # Merge overlapping windows
        windows = self._merge_overlapping(windows)
        
        # Limit number of snippets
        return windows[:self.max_snippets]
    
    def highlight(self, text: str, query_terms: List[str], 
                  highlight_start: str = '<mark>',
                  highlight_end: str = '</mark>') -> str:
        """Highlight query terms in text"""
        if not query_terms:
            return text
        
        matches = self._find_matches(text, query_terms)
        if not matches:
            return text
        
        # Build highlighted text
        result = []
        last_pos = 0
        
        for start, end, term in matches:
            # Add text before match
            result.append(text[last_pos:start])
            # Add highlighted match
            result.append(highlight_start)
            result.append(text[start:end])
            result.append(highlight_end)
            last_pos = end
        
        # Add remaining text
        result.append(text[last_pos:])
        
        return ''.join(result)
    
    def generate(self, text: str, query_terms: List[str],
                 highlight_start: str = '<mark>',
                 highlight_end: str = '</mark>') -> List[Dict]:
        """Generate snippets with highlighting"""
        if not text:
            return []
        
        matches = self._find_matches(text, query_terms)
        windows = self._extract_snippet_windows(text, matches)
        
        snippets = []
        for i, (start, end) in enumerate(windows):
            snippet_text = text[start:end]
            
            # Highlight terms in snippet
            highlighted = self.highlight(snippet_text, query_terms, highlight_start, highlight_end)
            
            # Add ellipsis if needed
            prefix = '...' if start > 0 else ''
            suffix = '...' if end < len(text) else ''
            
            snippets.append({
                'text': snippet_text,
                'highlighted': f"{prefix}{highlighted}{suffix}",
                'position': (start, end),
                'match_count': len([m for m in matches if start <= m[0] < end])
            })
        
        return snippets
    
    def generate_multi_field(self, document: Dict, query_terms: List[str]) -> Dict:
        """Generate snippets for multiple fields"""
        result = {
            'doc_id': document.get('id'),
            'title': None,
            'url': document.get('url'),
            'snippets': []
        }
        
        # Title snippet (always full, highlighted)
        if 'title' in document:
            result['title'] = self.highlight(document['title'], query_terms)
        
        # Content snippets
        if 'content' in document:
            result['snippets'] = self.generate(document['content'], query_terms)
        
        return result


class ResultPageFormatter:
    """Format search results for display"""
    
    def __init__(self, results_per_page: int = 10):
        self.results_per_page = results_per_page
        self.snippet_generator = SnippetGenerator()
    
    def format_result(self, rank: int, document: Dict, query_terms: List[str], score: float = None) -> Dict:
        """Format a single search result"""
        snippets = self.snippet_generator.generate_multi_field(document, query_terms)
        
        return {
            'rank': rank,
            'score': score,
            'title': snippets['title'] or document.get('title', 'Untitled'),
            'url': snippets['url'],
            'snippets': snippets['snippets'][:3],  # Top 3 snippets
            'doc_id': snippets['doc_id']
        }
    
    def format_page(self, documents: List[Dict], query: str, query_terms: List[str],
                   scores: List[float] = None, page: int = 1) -> Dict:
        """Format a page of search results"""
        start_idx = (page - 1) * self.results_per_page
        end_idx = start_idx + self.results_per_page
        
        page_docs = documents[start_idx:end_idx]
        page_scores = scores[start_idx:end_idx] if scores else [None] * len(page_docs)
        
        results = []
        for i, (doc, score) in enumerate(zip(page_docs, page_scores)):
            rank = start_idx + i + 1
            results.append(self.format_result(rank, doc, query_terms, score))
        
        return {
            'query': query,
            'total_results': len(documents),
            'page': page,
            'results_per_page': self.results_per_page,
            'total_pages': (len(documents) + self.results_per_page - 1) // self.results_per_page,
            'results': results
        }
    
    def to_text(self, formatted_page: Dict) -> str:
        """Convert formatted page to plain text"""
        lines = []
        lines.append("="*80)
        lines.append(f"Search results for: {formatted_page['query']}")
        lines.append(f"Page {formatted_page['page']}/{formatted_page['total_pages']} " +
                    f"({formatted_page['total_results']} total results)")
        lines.append("="*80)
        
        for result in formatted_page['results']:
            lines.append(f"\n{result['rank']}. {result['title']}")
            if result.get('url'):
                lines.append(f"   URL: {result['url']}")
            if result.get('score') is not None:
                lines.append(f"   Score: {result['score']:.4f}")
            
            for snippet in result['snippets']:
                # Remove HTML tags for plain text
                text = snippet['highlighted']
                text = text.replace('<mark>', '**').replace('</mark>', '**')
                lines.append(f"   {text}")
        
        lines.append("\n" + "="*80)
        return '\n'.join(lines)
    
    def to_html(self, formatted_page: Dict) -> str:
        """Convert formatted page to HTML"""
        html = ['<!DOCTYPE html><html><head><meta charset="UTF-8">']
        html.append('<style>')
        html.append('body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; }')
        html.append('.result { margin: 20px 0; padding: 15px; border-left: 3px solid #4285f4; }')
        html.append('.title { font-size: 18px; color: #1a0dab; margin-bottom: 5px; }')
        html.append('.url { color: #006621; font-size: 14px; }')
        html.append('.snippet { color: #545454; font-size: 14px; line-height: 1.5; margin-top: 5px; }')
        html.append('mark { background-color: #ffff00; font-weight: bold; }')
        html.append('.score { color: #999; font-size: 12px; }')
        html.append('.header { border-bottom: 1px solid #ddd; padding-bottom: 10px; margin-bottom: 20px; }')
        html.append('</style></head><body>')
        
        # Header
        html.append(f'<div class="header">')
        html.append(f'<h2>Search results for: {formatted_page["query"]}</h2>')
        html.append(f'<p>Page {formatted_page["page"]}/{formatted_page["total_pages"]} ')
        html.append(f'({formatted_page["total_results"]} total results)</p>')
        html.append('</div>')
        
        # Results
        for result in formatted_page['results']:
            html.append('<div class="result">')
            html.append(f'<div class="title">{result["rank"]}. {result["title"]}</div>')
            if result.get('url'):
                html.append(f'<div class="url">{result["url"]}</div>')
            if result.get('score') is not None:
                html.append(f'<div class="score">Score: {result["score"]:.4f}</div>')
            
            for snippet in result['snippets']:
                html.append(f'<div class="snippet">{snippet["highlighted"]}</div>')
            
            html.append('</div>')
        
        html.append('</body></html>')
        return '\n'.join(html)


def demo():
    """Demo snippet generation"""
    print("="*80)
    print("📄 SNIPPET GENERATION DEMO")
    print("="*80)
    
    # Sample document
    document = {
        'id': 'doc1',
        'title': 'Đội tuyển Việt Nam vô địch AFF Cup 2018',
        'url': 'https://example.com/news/viet-nam-vo-dich-aff-cup-2018',
        'content': '''
        Đội tuyển bóng đá Việt Nam đã xuất sắc vô địch AFF Cup 2018 sau khi đánh bại 
        Malaysia với tổng tỷ số 3-2 sau hai lượt trận chung kết. Đây là lần thứ hai 
        Việt Nam đăng quang tại giải đấu này. Huấn luyện viên Park Hang-seo đã tạo 
        nên kỳ tích cùng các cầu thủ Việt Nam. Trận chung kết diễn ra vô cùng kịch tính 
        với bàn thắng quyết định của Nguyễn Anh Đức. Chiến thắng này đã mang lại niềm 
        vui cho hàng triệu người hâm mộ bóng đá Việt Nam.
        '''
    }
    
    query = "việt nam vô địch aff cup"
    query_terms = ['việt_nam', 'vô_địch', 'aff_cup']
    
    print(f"\n📝 Query: {query}")
    print(f"🔍 Query terms: {', '.join(query_terms)}")
    
    # Generate snippets
    generator = SnippetGenerator()
    snippets = generator.generate_multi_field(document, query_terms)
    
    print(f"\n📌 Title (highlighted):")
    print(f"  {snippets['title']}")
    
    print(f"\n📄 Content snippets ({len(snippets['snippets'])}):")
    for i, snippet in enumerate(snippets['snippets'], 1):
        print(f"\n  Snippet {i}:")
        print(f"  {snippet['highlighted']}")
        print(f"  (Matches: {snippet['match_count']}, Position: {snippet['position']})")
    
    # Format as result page
    print("\n" + "="*80)
    print("📊 FORMATTED RESULT PAGE")
    print("="*80)
    
    formatter = ResultPageFormatter()
    result_page = formatter.format_page(
        documents=[document],
        query=query,
        query_terms=query_terms,
        scores=[0.9542],
        page=1
    )
    
    print(formatter.to_text(result_page))
    
    # Save HTML version
    html_output = formatter.to_html(result_page)
    with open('search_results_demo.html', 'w', encoding='utf-8') as f:
        f.write(html_output)
    print("\n✓ HTML version saved to: search_results_demo.html")


if __name__ == '__main__':
    demo()
