#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試OpenAI Embedding + FAISS向量資料庫的RAG系統
"""

from enhanced_news_rag import EnhancedNewsRAG
import os

def test_openai_faiss_system():
    """測試OpenAI + FAISS系統"""
    print("=== 測試OpenAI Embedding + FAISS向量資料庫 ===\n")
    
    # 檢查API金鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("請設定OPENAI_API_KEY環境變數")
        print("或執行: python setup_openai.py")
        return
    
    # 建立RAG系統
    rag = EnhancedNewsRAG(top_k=3, openai_api_key=api_key)
    
    # 測試載入現有向量資料庫
    print("1. 測試向量資料庫載入...")
    if rag.load_vector_database():
        print("✓ 成功載入現有向量資料庫")
    else:
        print("× 未找到現有向量資料庫，開始建立新的...")
        
        # 整併款次資料
        print("\n2. 測試款次整併...")
        if rag.consolidate_clauses("../data/重訊款次整理_20250715彙整提供.xlsx"):
            print(f"✓ 成功整併 {len(rag.consolidated_clauses)} 個款次")
        else:
            print("× 款次整併失敗")
            return
        
        # 建立向量資料庫
        print("\n3. 測試向量資料庫建立...")
        if rag.build_vector_database():
            print("✓ 成功建立FAISS向量資料庫")
            
            # 儲存向量資料庫
            if rag.save_vector_database():
                print("✓ 向量資料庫已儲存")
            else:
                print("× 向量資料庫儲存失敗")
        else:
            print("× 向量資料庫建立失敗")
            return
    
    # 測試向量搜索
    print("\n4. 測試向量搜索功能...")
    test_content = "台積電宣布投資新廠，預計投資金額達100億美元"
    
    candidates = rag.vector_search(test_content)
    if candidates:
        print(f"✓ 找到 {len(candidates)} 個候選款次:")
        for i, candidate in enumerate(candidates, 1):
            print(f"  {i}. 款次 {candidate['clause_number']}: 相似度 {candidate['similarity']:.4f}")
    else:
        print("× 向量搜索失敗")
    
    # 測試新聞處理
    print("\n5. 測試新聞處理...")
    results = rag.process_news("../data/新聞.csv")
    
    if results:
        print(f"✓ 成功處理 {len(results)} 則新聞")
        
        # 顯示第一則結果
        if results:
            result = results[0]
            print(f"\n範例結果:")
            print(f"  新聞標題: {result.news_title}")
            print(f"  符合款次: {result.primary_match.clause_number}")
            print(f"  信心分數: {result.primary_match.confidence_score:.4f}")
            print(f"  符合原因: {result.primary_match.reason}")
    else:
        print("× 新聞處理失敗")
    
    print("\n=== 測試完成 ===")

if __name__ == "__main__":
    test_openai_faiss_system()
