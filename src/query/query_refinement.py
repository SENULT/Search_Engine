"""
QUERY REFINEMENT SYSTEM
Advanced Query Processing for Vietnamese Search Engine

Components:
1. Spell Checking - Correct typos
2. Query Expansion - Add synonyms and related terms
3. Stopping & Stemming - Remove stopwords, normalize
4. Suggestions - Recommend better queries
5. Context & Personalization - User history and preferences
"""

import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import unicodedata
from pyvi import ViTokenizer
import difflib


class SpellChecker:
    """Vietnamese spell checker using edit distance and vocabulary"""
    
    def __init__(self, vocabulary: Set[str] = None):
        self.vocabulary = vocabulary or set()
        self.word_freq = Counter()
        
    def add_vocabulary(self, words: List[str], frequencies: Dict[str, int] = None):
        """Add words to vocabulary with optional frequencies"""
        self.vocabulary.update(words)
        if frequencies:
            self.word_freq.update(frequencies)
    
    def edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein distance"""
        if len(word1) < len(word2):
            return self.edit_distance(word2, word1)
        
        if len(word2) == 0:
            return len(word1)
        
        previous_row = range(len(word2) + 1)
        for i, c1 in enumerate(word1):
            current_row = [i + 1]
            for j, c2 in enumerate(word2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def get_candidates(self, word: str, max_distance: int = 2) -> List[Tuple[str, int, int]]:
        """Get candidate corrections (word, distance, frequency)"""
        if word in self.vocabulary:
            return [(word, 0, self.word_freq.get(word, 0))]
        
        candidates = []
        for vocab_word in self.vocabulary:
            # Quick filter by length
            if abs(len(vocab_word) - len(word)) > max_distance:
                continue
            
            distance = self.edit_distance(word, vocab_word)
            if distance <= max_distance:
                freq = self.word_freq.get(vocab_word, 0)
                candidates.append((vocab_word, distance, freq))
        
        # Sort by distance (lower better), then frequency (higher better)
        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[:5]
    
    def correct(self, word: str) -> Dict:
        """Correct a single word"""
        if word in self.vocabulary:
            return {
                'original': word,
                'corrected': word,
                'confidence': 1.0,
                'suggestions': []
            }
        
        candidates = self.get_candidates(word)
        
        if not candidates:
            return {
                'original': word,
                'corrected': word,
                'confidence': 0.0,
                'suggestions': []
            }
        
        best_match = candidates[0]
        return {
            'original': word,
            'corrected': best_match[0],
            'confidence': 1.0 - (best_match[1] / max(len(word), len(best_match[0]))),
            'suggestions': [c[0] for c in candidates]
        }
    
    def correct_query(self, query: str) -> Dict:
        """Correct entire query"""
        words = query.split()
        corrections = []
        corrected_words = []
        has_corrections = False
        
        for word in words:
            result = self.correct(word)
            corrections.append(result)
            corrected_words.append(result['corrected'])
            if result['original'] != result['corrected']:
                has_corrections = True
        
        return {
            'original': query,
            'corrected': ' '.join(corrected_words),
            'has_corrections': has_corrections,
            'details': corrections
        }


class QueryExpander:
    """Expand queries with synonyms and related terms"""
    
    def __init__(self):
        # Vietnamese football domain synonyms
        self.synonyms = {
            'bóng_đá': ['túc_cầu', 'football', 'soccer'],
            'cầu_thủ': ['tuyển_thủ', 'player', 'cầu_thủ_viên'],
            'huấn_luyện_viên': ['hlv', 'coach', 'trainer'],
            'trận_đấu': ['trận', 'match', 'game', 'cuộc_đấu'],
            'chung_kết': ['final', 'trận_chung_kết'],
            'bán_kết': ['semi_final', 'semifinal'],
            'vô_địch': ['champion', 'championship', 'vô_địch_viên'],
            'đội_tuyển': ['đội', 'team', 'tuyển'],
            'bàn_thắng': ['goal', 'ghi_bàn'],
            'thủ_môn': ['goalkeeper', 'gk'],
            'hậu_vệ': ['defender', 'def'],
            'tiền_đạo': ['striker', 'forward', 'fw'],
            'tiền_vệ': ['midfielder', 'mid', 'mf'],
        }
        
        # Related terms (concepts that often appear together)
        self.related_terms = {
            'việt_nam': ['vn', 'việt', 'vnm'],
            'thái_lan': ['thái', 'thailand'],
            'sea_games': ['seagames', 'seag'],
            'world_cup': ['worldcup', 'wc'],
            'aff_cup': ['aff', 'asean'],
            'park_hang_seo': ['park', 'hlv_park'],
        }
        
        # Hypernyms (broader terms)
        self.hypernyms = {
            'việt_nam': ['đông_nam_á', 'châu_á'],
            'thái_lan': ['đông_nam_á', 'châu_á'],
            'sea_games': ['giải_đấu', 'tournament'],
            'world_cup': ['giải_đấu', 'tournament'],
        }
    
    def expand(self, query: str, max_expansions: int = 3) -> Dict:
        """Expand query with synonyms and related terms"""
        words = query.split()
        
        # Original query tokens
        expanded_terms = {
            'original': set(words),
            'synonyms': set(),
            'related': set(),
            'hypernyms': set()
        }
        
        for word in words:
            # Add synonyms
            if word in self.synonyms:
                expanded_terms['synonyms'].update(self.synonyms[word][:max_expansions])
            
            # Add related terms
            if word in self.related_terms:
                expanded_terms['related'].update(self.related_terms[word][:max_expansions])
            
            # Add hypernyms (broader context)
            if word in self.hypernyms:
                expanded_terms['hypernyms'].update(self.hypernyms[word][:max_expansions])
        
        # Build expanded query
        all_terms = (
            list(expanded_terms['original']) +
            list(expanded_terms['synonyms']) +
            list(expanded_terms['related'])
        )
        
        return {
            'original_query': query,
            'expanded_query': ' '.join(all_terms),
            'expansion_details': {
                'original': list(expanded_terms['original']),
                'synonyms': list(expanded_terms['synonyms']),
                'related': list(expanded_terms['related']),
                'hypernyms': list(expanded_terms['hypernyms'])
            }
        }


class QuerySuggester:
    """Generate query suggestions based on common patterns"""
    
    def __init__(self):
        self.query_log = Counter()  # Track query frequency
        self.query_patterns = defaultdict(list)  # Common patterns
        
        # Pre-populated common queries
        self.popular_queries = [
            'bóng_đá_việt_nam',
            'đội_tuyển_việt_nam',
            'park_hang_seo',
            'sea_games_2023',
            'vô_địch_aff_cup',
            'world_cup_2022',
            'cầu_thủ_xuất_sắc',
            'lịch_thi_đấu',
            'kết_quả_trận_đấu',
            'bảng_xếp_hạng'
        ]
    
    def add_query(self, query: str):
        """Add query to log"""
        self.query_log[query] += 1
    
    def suggest_by_prefix(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """Suggest queries by prefix matching"""
        suggestions = []
        
        # From popular queries
        for query in self.popular_queries:
            if query.startswith(prefix.lower()):
                suggestions.append(query)
        
        # From query log
        for query, freq in self.query_log.most_common(100):
            if query.startswith(prefix.lower()) and query not in suggestions:
                suggestions.append(query)
        
        return suggestions[:max_suggestions]
    
    def suggest_similar(self, query: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Suggest similar queries using fuzzy matching"""
        candidates = []
        
        all_queries = self.popular_queries + list(self.query_log.keys())
        
        for candidate in all_queries:
            similarity = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
            if similarity > 0.5 and query != candidate:
                freq = self.query_log.get(candidate, 1)
                score = similarity * 0.7 + (freq / max(self.query_log.values() or [1])) * 0.3
                candidates.append((candidate, score))
        
        candidates.sort(key=lambda x: -x[1])
        return candidates[:max_suggestions]
    
    def suggest_completions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Auto-complete suggestions"""
        if not partial_query:
            return []
        
        # Get prefix suggestions
        prefix_suggestions = self.suggest_by_prefix(partial_query, max_suggestions)
        
        # If not enough, add similar queries
        if len(prefix_suggestions) < max_suggestions:
            similar = self.suggest_similar(partial_query, max_suggestions - len(prefix_suggestions))
            prefix_suggestions.extend([s[0] for s in similar])
        
        return prefix_suggestions[:max_suggestions]


class StopwordRemover:
    """Remove Vietnamese stopwords"""
    
    def __init__(self):
        # Vietnamese stopwords
        self.stopwords = {
            'và', 'của', 'có', 'được', 'trong', 'từ', 'với', 'là', 'để',
            'một', 'các', 'những', 'này', 'đó', 'khi', 'như', 'sau', 'trước',
            'về', 'cho', 'nếu', 'thì', 'bởi', 'vì', 'nên', 'đã', 'sẽ',
            'đang', 'vẫn', 'còn', 'mà', 'ở', 'tại', 'bằng', 'theo'
        }
        
        # Context-aware: Don't remove these in certain contexts
        self.keep_in_context = {
            'có': ['có thể', 'có được'],
            'được': ['được ghi', 'được chọn'],
        }
    
    def should_remove(self, word: str, context: List[str] = None) -> bool:
        """Check if word should be removed"""
        if word not in self.stopwords:
            return False
        
        # Check context
        if context and word in self.keep_in_context:
            context_str = ' '.join(context)
            for phrase in self.keep_in_context[word]:
                if phrase in context_str:
                    return False
        
        return True
    
    def remove(self, tokens: List[str], keep_positions: bool = False) -> List[str]:
        """Remove stopwords from token list"""
        if keep_positions:
            # Keep positions but mark as removed
            return [token if not self.should_remove(token) else None for token in tokens]
        else:
            return [token for token in tokens if not self.should_remove(token)]


class QueryRefinementPipeline:
    """Complete query refinement pipeline"""
    
    def __init__(self, vocab_file: str = None):
        self.spell_checker = SpellChecker()
        self.query_expander = QueryExpander()
        self.suggester = QuerySuggester()
        self.stopword_remover = StopwordRemover()
        
        # Load vocabulary if provided
        if vocab_file:
            self._load_vocabulary(vocab_file)
    
    def _load_vocabulary(self, vocab_file: str):
        """Load vocabulary for spell checking"""
        try:
            with open(vocab_file, 'r', encoding='utf-8') as f:
                words = []
                freqs = {}
                for line in f:
                    parts = line.strip().split('\t')
                    if parts:
                        word = parts[0]
                        freq = int(parts[1]) if len(parts) > 1 else 1
                        words.append(word)
                        freqs[word] = freq
                
                self.spell_checker.add_vocabulary(words, freqs)
                print(f"✓ Loaded {len(words)} words from vocabulary")
        except Exception as e:
            print(f"⚠️ Could not load vocabulary: {e}")
    
    def refine(self, 
               query: str,
               correct_spelling: bool = True,
               expand_query: bool = False,
               remove_stopwords: bool = False,
               get_suggestions: bool = True) -> Dict:
        """
        Complete query refinement
        
        Args:
            query: Input query
            correct_spelling: Apply spell checking
            expand_query: Add synonyms and related terms
            remove_stopwords: Remove Vietnamese stopwords
            get_suggestions: Generate query suggestions
        
        Returns:
            Dict with refined query and details
        """
        result = {
            'original_query': query,
            'refined_query': query,
            'steps': []
        }
        
        current_query = query
        
        # Step 1: Tokenization
        tokens = ViTokenizer.tokenize(current_query).split()
        result['steps'].append({
            'step': 'tokenization',
            'input': current_query,
            'output': ' '.join(tokens),
            'tokens': tokens
        })
        current_query = ' '.join(tokens)
        
        # Step 2: Spell Checking
        if correct_spelling:
            spell_result = self.spell_checker.correct_query(current_query)
            result['steps'].append({
                'step': 'spell_checking',
                'input': current_query,
                'output': spell_result['corrected'],
                'has_corrections': spell_result['has_corrections'],
                'details': spell_result['details']
            })
            current_query = spell_result['corrected']
        
        # Step 3: Stopword Removal
        if remove_stopwords:
            tokens = current_query.split()
            filtered_tokens = self.stopword_remover.remove(tokens)
            result['steps'].append({
                'step': 'stopword_removal',
                'input': current_query,
                'output': ' '.join(filtered_tokens),
                'removed': [t for t in tokens if t not in filtered_tokens]
            })
            current_query = ' '.join(filtered_tokens)
        
        # Step 4: Query Expansion
        expansion_result = None
        if expand_query:
            expansion_result = self.query_expander.expand(current_query)
            result['steps'].append({
                'step': 'query_expansion',
                'input': current_query,
                'output': expansion_result['expanded_query'],
                'details': expansion_result['expansion_details']
            })
            # Note: We keep original query for search, but provide expanded version
        
        # Step 5: Query Suggestions
        suggestions = []
        if get_suggestions:
            suggestions = self.suggester.suggest_completions(query)
            similar = self.suggester.suggest_similar(query)
            result['steps'].append({
                'step': 'suggestions',
                'completions': suggestions,
                'similar_queries': [s[0] for s in similar]
            })
        
        # Final refined query
        result['refined_query'] = current_query
        result['expanded_query'] = expansion_result['expanded_query'] if expansion_result else None
        result['suggestions'] = suggestions
        
        # Track query for future suggestions
        self.suggester.add_query(query)
        
        return result
    
    def process(self, query: str, **options) -> str:
        """Simple interface - returns refined query string"""
        result = self.refine(query, **options)
        return result['refined_query']


def demo():
    """Demo the query refinement system"""
    print("="*80)
    print("🔍 QUERY REFINEMENT SYSTEM DEMO")
    print("="*80)
    
    pipeline = QueryRefinementPipeline()
    
    # Test queries
    test_queries = [
        ("bong da viet nam", "No accents - needs spell check"),
        ("hlv park hang seo", "Abbreviation + proper name"),
        ("tran chung ket aff cup", "Multiple terms needing correction"),
        ("cau thu xuat sac nhat", "Player query with adjectives"),
    ]
    
    for query, description in test_queries:
        print(f"\n{'─'*80}")
        print(f"📝 Test: {description}")
        print(f"{'─'*80}")
        print(f"Original: {query}")
        
        result = pipeline.refine(
            query,
            correct_spelling=True,
            expand_query=True,
            remove_stopwords=False,
            get_suggestions=True
        )
        
        print(f"\n🔧 Processing steps:")
        for step in result['steps']:
            if step['step'] == 'spell_checking' and step.get('has_corrections'):
                print(f"  • {step['step']}: {step['input']} → {step['output']}")
                for detail in step['details']:
                    if detail['original'] != detail['corrected']:
                        print(f"    ✓ '{detail['original']}' → '{detail['corrected']}' (confidence: {detail['confidence']:.2f})")
            elif step['step'] == 'query_expansion':
                print(f"  • {step['step']}:")
                for key, values in step['details'].items():
                    if values:
                        print(f"    {key}: {', '.join(values)}")
            elif step['step'] == 'suggestions':
                print(f"  • {step['step']}:")
                if step.get('completions'):
                    print(f"    Completions: {', '.join(step['completions'][:3])}")
        
        print(f"\n✨ Results:")
        print(f"  Refined: {result['refined_query']}")
        if result.get('expanded_query'):
            print(f"  Expanded: {result['expanded_query']}")


if __name__ == '__main__':
    demo()
