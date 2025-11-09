"""
Build Vocabulary từ Inverted Index

Extract tất cả terms từ inverted index để làm vocabulary cho accent restoration.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.indexing.inverted_index import IndexBuilder
from collections import Counter


def build_vocab_from_index(index_file: str, output_file: str = "data/vocab/vocab_from_index.txt"):
    """
    Extract vocabulary từ inverted index
    
    Args:
        index_file: Path to index pickle file
        output_file: Output vocabulary file
    """
    print("="*80)
    print("BUILDING VOCABULARY FROM INVERTED INDEX")
    print("="*80)
    
    # Load index
    print(f"\n1. Loading index from: {index_file}")
    try:
        index = IndexBuilder.load_index_from_pickle(index_file)
        print(f"   ✓ Loaded: {index.total_docs:,} documents")
        print(f"   ✓ Vocabulary size: {len(index.vocabulary):,} terms")
    except FileNotFoundError:
        print(f"   ✗ File not found: {index_file}")
        print("\n⚠️ Please build index first using:")
        print("   python -c \"from src.indexing.inverted_index import IndexBuilder; builder = IndexBuilder(); builder.build_index_from_collection('vnexpress_bongda', limit=1000); builder.save_index_to_pickle('outputs/indexes/index.pkl')\"")
        return
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # Extract terms with frequency
    print(f"\n2. Extracting terms with frequency...")
    term_freq = Counter()
    
    for term, data in index.index.items():
        df = data['df']  # document frequency
        term_freq[term] = df
    
    print(f"   ✓ Extracted: {len(term_freq):,} terms")
    
    # Sort by frequency
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Save to file
    print(f"\n3. Saving vocabulary to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for term, freq in sorted_terms:
            f.write(f"{term}\n")
    
    print(f"   ✓ Saved: {len(sorted_terms):,} terms")
    
    # Statistics
    print(f"\n4. Vocabulary Statistics:")
    print(f"   - Total terms: {len(sorted_terms):,}")
    print(f"   - Top 10 most frequent:")
    
    for i, (term, freq) in enumerate(sorted_terms[:10], 1):
        print(f"     {i:2d}. '{term}' → {freq} documents")
    
    print(f"\n   - Terms appearing in only 1 document: {sum(1 for _, freq in sorted_terms if freq == 1):,}")
    print(f"   - Terms appearing in >10 documents: {sum(1 for _, freq in sorted_terms if freq > 10):,}")
    print(f"   - Terms appearing in >100 documents: {sum(1 for _, freq in sorted_terms if freq > 100):,}")
    
    print(f"\n{'='*80}")
    print(f"✓ VOCABULARY BUILT SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n📍 Next step: Use this vocab for accent restoration:")
    print(f"   from src.utils.query_processor import QueryProcessor")
    print(f"   processor = QueryProcessor(vocab_file='{output_file}')")
    print(f"   result = processor.process_query('bong da viet nam')")
    print(f"   print(result['restored'])  # → 'bóng đá việt nam'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vocabulary from inverted index")
    parser.add_argument(
        '--index',
        default='outputs/indexes/index.pkl',
        help='Path to inverted index pickle file'
    )
    parser.add_argument(
        '--output',
        default='data/vocab/vocab_from_index.txt',
        help='Output vocabulary file'
    )
    
    args = parser.parse_args()
    
    build_vocab_from_index(args.index, args.output)
