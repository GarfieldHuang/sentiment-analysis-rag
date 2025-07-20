#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試HTML內容清理和token限制功能
"""

from enhanced_news_rag import EnhancedNewsRAG
import pandas as pd
import os

def test_content_cleaning():
    """測試內容清理功能"""
    print("=== 測試HTML內容清理和Token限制 ===\n")
    
    # 建立RAG系統
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("請設定OPENAI_API_KEY環境變數")
        return
    
    rag = EnhancedNewsRAG(top_k=3, openai_api_key=api_key)
    
    # 讀取新聞CSV
    try:
        df = pd.read_csv("../data/新聞.csv")
        print(f"載入 {len(df)} 則新聞進行測試\n")
    except Exception as e:
        print(f"無法載入新聞檔案: {e}")
        return
    
    # 測試每則新聞的內容清理
    for i, row in df.iterrows():
        if i >= 3:  # 只測試前3則
            break
            
        url = row['網址']
        print(f"=== 測試新聞 {i+1} ===")
        print(f"網址: {url}")
        
        # 1. 擷取原始內容
        title, content = rag.get_news_content(url)
        print(f"標題: {title[:50]}...")
        print(f"原始內容長度: {len(content)} 字元")
        
        # 2. 測試HTML清理
        cleaned_content = rag.clean_html_content(content)
        print(f"清理後長度: {len(cleaned_content)} 字元")
        
        # 3. 測試token限制
        truncated_content = rag.truncate_text_for_embedding(cleaned_content)
        print(f"截斷後長度: {len(truncated_content)} 字元")
        
        # 4. 測試完整預處理
        processed_content = rag.preprocess_text(content)
        print(f"預處理後長度: {len(processed_content)} 字元")
        
        # 5. 測試Embedding
        try:
            embedding = rag.get_embeddings([processed_content])
            if embedding is not None:
                print(f"✓ Embedding成功，維度: {embedding.shape}")
            else:
                print("× Embedding失敗")
        except Exception as e:
            print(f"× Embedding錯誤: {e}")
        
        print(f"內容預覽: {processed_content[:100]}...\n")
        print("-" * 60)

def test_full_system():
    """測試完整系統"""
    print("\n=== 測試完整系統（含內容清理） ===\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("請設定OPENAI_API_KEY環境變數")
        return
    
    rag = EnhancedNewsRAG(top_k=3, openai_api_key=api_key)
    
    # 測試向量資料庫建立（使用清理後的內容）
    print("1. 整併款次資料...")
    if rag.consolidate_clauses("../data/重訊款次整理_20250715彙整提供.xlsx"):
        print("✓ 款次整併成功")
    else:
        print("× 款次整併失敗")
        return
    
    print("\n2. 建立向量資料庫（使用清理功能）...")
    if rag.build_vector_database():
        print("✓ 向量資料庫建立成功")
    else:
        print("× 向量資料庫建立失敗")
        return
    
    print("\n3. 測試新聞分析...")
    results = rag.process_news("../data/新聞.csv")
    
    if results:
        print(f"✓ 成功分析 {len(results)} 則新聞")
        
        # 顯示第一則結果
        if results:
            result = results[0]
            print(f"\n範例結果:")
            print(f"  新聞標題: {result.news_title}")
            print(f"  符合款次: {result.primary_match.clause_number}")
            print(f"  信心分數: {result.primary_match.confidence_score:.4f}")
    else:
        print("× 新聞分析失敗")

if __name__ == "__main__":
    test_content_cleaning()
    test_full_system()
