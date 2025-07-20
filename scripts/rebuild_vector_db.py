#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重建向量資料庫工具
"""

from enhanced_news_rag import EnhancedNewsRAG
import os

def rebuild_vector_database():
    """重建向量資料庫"""
    print("=== 重建向量資料庫 ===\n")
    
    # 檢查API金鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("錯誤：未設定OPENAI_API_KEY環境變數")
        print("請先設定API金鑰：")
        print("  Windows: set OPENAI_API_KEY=your_api_key")
        print("  或執行: python setup_openai.py")
        return
    
    # 建立RAG系統
    rag = EnhancedNewsRAG(top_k=5, openai_api_key=api_key)
    
    # 清空現有資料
    print("1. 清空現有向量資料庫...")
    rag.clear_vector_database()
    
    # 重新整併款次
    print("2. 重新整併重訊款次...")
    if not rag.consolidate_clauses("../data/重訊款次整理_20250715彙整提供.xlsx"):
        print("× 款次整併失敗")
        return
    
    # 重建向量資料庫
    print("3. 重建向量資料庫...")
    if not rag.build_vector_database():
        print("× 向量資料庫建立失敗")
        return
    
    # 儲存新的向量資料庫
    print("4. 儲存向量資料庫...")
    if rag.save_vector_database():
        print("✓ 向量資料庫重建完成")
        
        # 顯示統計資訊
        print(f"\n重建結果:")
        print(f"  整併款次數: {len(rag.consolidated_clauses)}")
        print(f"  向量數量: {len(rag.vector_db)}")
        
        # 測試載入
        print("\n5. 測試載入功能...")
        test_rag = EnhancedNewsRAG(top_k=3)
        if test_rag.load_vector_database():
            print("✓ 向量資料庫載入測試成功")
            print(f"  載入款次數: {len(test_rag.consolidated_clauses)}")
        else:
            print("× 向量資料庫載入測試失敗")
    else:
        print("× 向量資料庫儲存失敗")

if __name__ == "__main__":
    rebuild_vector_database()
