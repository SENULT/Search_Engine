"""
INTERACTIVE QUERY PROCESSING DEMO
Nhập query để test toàn bộ pipeline: Spell Check, Expansion, Suggestions, Snippets

Bạn có thể test:
- Query CÓ DẤU vs KHÔNG DẤU
- Spell checking tự động
- Query expansion với synonyms
- Snippet generation với highlighting
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.query.query_refinement import QueryRefinementPipeline
from src.query.snippet_generator import SnippetGenerator, ResultPageFormatter


def print_banner():
    """Print welcome banner"""
    print("="*80)
    print("🔍 INTERACTIVE QUERY PROCESSING DEMO")
    print("="*80)
    print("\n✨ Features:")
    print("  1. ✓ Spell Checking (sửa lỗi chính tả)")
    print("  2. ✓ Accent Restoration (tự động thêm dấu)")
    print("  3. ✓ Query Expansion (mở rộng với synonyms)")
    print("  4. ✓ Suggestions (gợi ý queries)")
    print("  5. ✓ Snippet Generation (hiển thị kết quả với highlight)")
    print("\n📝 Commands:")
    print("  • Type your query → See processing steps")
    print("  • 'examples' → Show example queries")
    print("  • 'compare' → Compare WITH vs WITHOUT processing")
    print("  • 'search <query>' → Full search demo with snippets")
    print("  • 'help' → Show this help")
    print("  • 'exit' → Quit")
    print("="*80)


def show_examples():
    """Show example queries"""
    print("\n📚 EXAMPLE QUERIES:")
    print("─"*80)
    examples = [
        ("bong da viet nam", "No accents - will auto restore"),
        ("huan luyen vien park", "Missing accents + name"),
        ("tran chung ket aff cup", "Tournament query"),
        ("cau thu xuat sac nhat", "Player query with superlative"),
        ("lich thi dau world cup", "Schedule query"),
        ("bang xep hang doi tuyen", "Ranking query"),
    ]
    
    for query, description in examples:
        print(f"  • '{query}'")
        print(f"    → {description}")
    print("─"*80)


def process_query_interactive(pipeline, query):
    """Process query and show all steps"""
    print(f"\n{'='*80}")
    print(f"🔍 PROCESSING: '{query}'")
    print(f"{'='*80}")
    
    result = pipeline.refine(
        query,
        correct_spelling=True,
        expand_query=True,
        remove_stopwords=False,
        get_suggestions=True
    )
    
    # Show processing steps
    print(f"\n📊 PROCESSING PIPELINE:")
    print("─"*80)
    
    for i, step in enumerate(result['steps'], 1):
        step_name = step['step'].replace('_', ' ').title()
        print(f"\n{i}. {step_name}")
        
        if step['step'] == 'tokenization':
            print(f"   Input:  '{step['input']}'")
            print(f"   Output: {step['tokens']}")
        
        elif step['step'] == 'spell_checking':
            if step['has_corrections']:
                print(f"   ✓ Corrections found:")
                for detail in step['details']:
                    if detail['original'] != detail['corrected']:
                        print(f"     • '{detail['original']}' → '{detail['corrected']}' " +
                              f"(confidence: {detail['confidence']:.0%})")
                print(f"   Result: '{step['output']}'")
            else:
                print(f"   ✓ No corrections needed")
        
        elif step['step'] == 'stopword_removal':
            if step.get('removed'):
                print(f"   Removed: {', '.join(step['removed'])}")
            print(f"   Result: '{step['output']}'")
        
        elif step['step'] == 'query_expansion':
            details = step['details']
            print(f"   Original: {', '.join(details['original'])}")
            if details['synonyms']:
                print(f"   + Synonyms: {', '.join(details['synonyms'])}")
            if details['related']:
                print(f"   + Related: {', '.join(details['related'])}")
            if details['hypernyms']:
                print(f"   + Broader: {', '.join(details['hypernyms'])}")
        
        elif step['step'] == 'suggestions':
            if step.get('completions'):
                print(f"   Auto-complete: {', '.join(step['completions'][:3])}")
            if step.get('similar_queries'):
                print(f"   Similar: {', '.join(step['similar_queries'][:3])}")
    
    # Show final results
    print(f"\n{'─'*80}")
    print(f"✨ FINAL RESULTS:")
    print(f"{'─'*80}")
    print(f"Original Query:  '{query}'")
    print(f"Refined Query:   '{result['refined_query']}'")
    if result.get('expanded_query'):
        print(f"Expanded Query:  '{result['expanded_query']}'")
    
    if result.get('suggestions'):
        print(f"\n💡 Did you mean?")
        for i, sugg in enumerate(result['suggestions'][:5], 1):
            print(f"  {i}. {sugg}")
    
    print("="*80)
    
    return result


def compare_queries(pipeline):
    """Compare WITH vs WITHOUT processing"""
    print(f"\n{'='*80}")
    print("⚖️  COMPARISON: With vs Without Query Processing")
    print(f"{'='*80}")
    
    test_query = input("\n📝 Enter query to compare: ").strip()
    if not test_query:
        test_query = "bong da viet nam vo dich"
        print(f"Using default: '{test_query}'")
    
    print(f"\n🔍 Testing: '{test_query}'")
    
    # WITHOUT processing
    print(f"\n❌ WITHOUT Query Processing:")
    print("─"*40)
    print(f"  Raw terms: {test_query.split()}")
    print(f"  → Search exactly as typed")
    print(f"  → May miss results due to:")
    print(f"    • Missing accents")
    print(f"    • Typos")
    print(f"    • No synonyms")
    
    # WITH processing
    print(f"\n✅ WITH Query Processing:")
    print("─"*40)
    result = pipeline.refine(test_query, correct_spelling=True, expand_query=True)
    
    print(f"  Refined: {result['refined_query'].split()}")
    if result.get('expanded_query'):
        expanded = result['expanded_query'].split()
        print(f"  Expanded: {expanded}")
        print(f"  → {len(expanded)} terms (from {len(test_query.split())} original)")
    
    print(f"\n📊 Improvements:")
    corrections = sum(1 for s in result['steps'] 
                     if s['step'] == 'spell_checking' and s.get('has_corrections'))
    print(f"  ✓ Spelling corrections: {corrections} words")
    
    expansion = next((s for s in result['steps'] if s['step'] == 'query_expansion'), None)
    if expansion:
        added_terms = (len(expansion['details']['synonyms']) + 
                      len(expansion['details']['related']))
        print(f"  ✓ Terms added: {added_terms} (synonyms + related)")
    
    print(f"  ✓ Better recall expected!")
    print("="*80)


def search_demo(pipeline, query):
    """Demo full search with snippets"""
    print(f"\n{'='*80}")
    print(f"🔍 FULL SEARCH DEMO: '{query}'")
    print(f"{'='*80}")
    
    # Process query
    result = pipeline.refine(query, correct_spelling=True, expand_query=False)
    refined = result['refined_query']
    query_terms = refined.split()
    
    print(f"\n✓ Query processed: '{refined}'")
    print(f"✓ Search terms: {query_terms}")
    
    # Mock search results
    mock_documents = [
        {
            'id': 'doc1',
            'title': 'Đội tuyển Việt Nam vô địch AFF Cup 2018',
            'url': 'https://vnexpress.net/dtqg-viet-nam-vo-dich-aff-cup-2018',
            'content': '''Đội tuyển bóng đá Việt Nam đã xuất sắc vô địch AFF Cup 2018 
            sau khi đánh bại Malaysia với tổng tỷ số 3-2 sau hai lượt trận chung kết. 
            Đây là lần thứ hai Việt Nam đăng quang tại giải đấu này. Huấn luyện viên 
            Park Hang-seo đã tạo nên kỳ tích cùng các cầu thủ Việt Nam. Trận chung kết 
            diễn ra vô cùng kịch tính với bàn thắng quyết định của Nguyễn Anh Đức.'''
        },
        {
            'id': 'doc2',
            'title': 'Park Hang-seo gia hạn hợp đồng với VFF đến 2023',
            'url': 'https://vnexpress.net/park-hang-seo-gia-han-hop-dong',
            'content': '''Huấn luyện viên Park Hang-seo chính thức gia hạn hợp đồng 
            với Liên đoàn bóng đá Việt Nam (VFF) đến năm 2023. Ông sẽ tiếp tục dẫn dắt 
            đội tuyển Việt Nam tại các giải đấu quan trọng như vòng loại World Cup và 
            AFF Cup. Thành công của Park Hang-seo với bóng đá Việt Nam là không thể phủ nhận.'''
        },
        {
            'id': 'doc3',
            'title': 'Lịch thi đấu vòng loại World Cup 2022 - Đội tuyển Việt Nam',
            'url': 'https://vnexpress.net/lich-thi-dau-vl-world-cup-2022',
            'content': '''Đội tuyển Việt Nam sẽ có những trận đấu quan trọng trong vòng loại 
            World Cup 2022 khu vực châu Á. Huấn luyện viên Park Hang-seo và các cầu thủ đang 
            tích cực chuẩn bị. Trận đấu đầu tiên dự kiến diễn ra vào tháng 3 năm 2022.'''
        }
    ]
    
    # Format results with snippets
    formatter = ResultPageFormatter(results_per_page=10)
    scores = [0.9542, 0.8731, 0.7215]  # Mock relevance scores
    
    result_page = formatter.format_page(
        documents=mock_documents,
        query=refined,
        query_terms=query_terms,
        scores=scores,
        page=1
    )
    
    # Display results
    print(formatter.to_text(result_page))
    
    # Save HTML
    html = formatter.to_html(result_page)
    html_file = 'search_results.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ HTML version saved: {html_file}")
    print(f"  Open in browser to see highlighted results!")
    print("="*80)


def main():
    """Main interactive loop"""
    print_banner()
    
    # Initialize pipeline
    print("\n⚙️  Initializing query processing pipeline...")
    pipeline = QueryRefinementPipeline()
    print("✓ Ready!\n")
    
    # Pre-populate some queries for suggestions
    sample_queries = [
        'bóng_đá_việt_nam', 'đội_tuyển_việt_nam', 'park_hang_seo',
        'sea_games', 'aff_cup', 'world_cup', 'lịch_thi_đấu',
        'bảng_xếp_hạng', 'cầu_thủ_xuất_sắc', 'huấn_luyện_viên'
    ]
    for q in sample_queries:
        pipeline.suggester.add_query(q)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n🔍 Enter query (or command): ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            if user_input.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                print_banner()
            
            elif user_input.lower() == 'examples':
                show_examples()
            
            elif user_input.lower() == 'compare':
                compare_queries(pipeline)
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    search_demo(pipeline, query)
                else:
                    print("❌ Please provide a query after 'search'")
            
            else:
                # Regular query processing
                process_query_interactive(pipeline, user_input)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
