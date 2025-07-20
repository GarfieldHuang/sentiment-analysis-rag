#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試增強版RAG系統
"""

import os
from enhanced_news_rag import EnhancedNewsRAG

def test_system():
    """測試系統功能"""
    print("=== 測試增強版RAG系統 ===")
    
    # 檢查OpenAI API金鑰
    if not os.getenv("OPENAI_API_KEY"):
        print("請設定環境變數 OPENAI_API_KEY")
        print("Windows: set OPENAI_API_KEY=your_api_key_here")
        print("或者在程式中直接設定:")
        print("os.environ['OPENAI_API_KEY'] = 'your_api_key_here'")
        return
    
    # 建立RAG系統
    rag = EnhancedNewsRAG(top_k=3)
    
    # 測試款次整併
    print("\n1. 測試款次整併...")
    consolidated = rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
    if consolidated:
        print(f"✓ 成功整併 {len(consolidated)} 個款次")
        
        # 顯示前5個款次
        print("\n前5個款次:")
        for i, (clause_num, data) in enumerate(list(consolidated.items())[:5]):
            print(f"  款次 {clause_num}: {data['full_description'][:100]}...")
    else:
        print("✗ 款次整併失敗")
        return
    
    # 測試向量資料庫
    print("\n2. 測試向量資料庫建立...")
    if rag.build_vector_database():
        print("✓ 向量資料庫建立成功")
    else:
        print("✗ 向量資料庫建立失敗")
        return
    
    # 測試新聞擷取
    print("\n3. 測試新聞擷取...")
    test_url = "https://udn.com/news/story/7252/8683898"
    title, content = rag.get_news_content(test_url)
    if title and content:
        print(f"✓ 新聞擷取成功")
        print(f"  標題: {title}")
        print(f"  內容長度: {len(content)} 字元")
    else:
        print("✗ 新聞擷取失敗")
    
    # 測試向量搜索
    print("\n4. 測試向量搜索...")
    if title and content:
        candidates = rag.vector_search(f"{title} {content}")
        if candidates:
            print(f"✓ 找到 {len(candidates)} 個候選款次")
            for i, candidate in enumerate(candidates):
                print(f"  {i+1}. 款次 {candidate['clause_number']}: 相似度 {candidate['similarity']:.4f}")
        else:
            print("✗ 沒有找到候選款次")
    
    print("\n=== 測試完成 ===")
    print("如果要執行完整分析，請執行: python enhanced_news_rag.py")

if __name__ == "__main__":
    test_system()
