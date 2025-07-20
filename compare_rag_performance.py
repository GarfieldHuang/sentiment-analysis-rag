#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較TF-IDF與OpenAI Embedding效能
"""

import time
import numpy as np
from enhanced_news_rag import EnhancedNewsRAG
from simplified_enhanced_rag import NewsRAG  # 原本的TF-IDF版本
import os

def compare_performance():
    """比較TF-IDF與OpenAI Embedding的效能"""
    print("=== RAG系統效能比較 ===\n")
    
    # 檢查API金鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("警告：未設定OPENAI_API_KEY，將跳過OpenAI測試")
        test_openai = False
    else:
        test_openai = True
    
    # 測試新聞內容
    test_news = [
        "台積電宣布投資新廠房建設，預計投入資金100億美元",
        "上市公司發布年度財報，營收較去年同期成長15%",
        "公司董事會決議發放現金股利每股2元",
        "企業併購案完成，取得目標公司60%股權"
    ]
    
    # 1. 測試TF-IDF版本
    print("1. 測試TF-IDF版本...")
    tfidf_rag = NewsRAG(top_k=5)
    
    start_time = time.time()
    tfidf_rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
    tfidf_rag.build_vector_database()
    tfidf_build_time = time.time() - start_time
    
    # 測試搜索速度
    tfidf_search_times = []
    tfidf_results = []
    
    for news in test_news:
        start_time = time.time()
        candidates = tfidf_rag.vector_search(news)
        search_time = time.time() - start_time
        tfidf_search_times.append(search_time)
        tfidf_results.append(candidates)
    
    print(f"  ✓ 建立時間: {tfidf_build_time:.2f}秒")
    print(f"  ✓ 平均搜索時間: {np.mean(tfidf_search_times):.4f}秒")
    
    # 2. 測試OpenAI Embedding版本
    if test_openai:
        print("\n2. 測試OpenAI Embedding版本...")
        openai_rag = EnhancedNewsRAG(top_k=5, openai_api_key=api_key)
        
        start_time = time.time()
        openai_rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
        openai_rag.build_vector_database()
        openai_build_time = time.time() - start_time
        
        # 測試搜索速度
        openai_search_times = []
        openai_results = []
        
        for news in test_news:
            start_time = time.time()
            candidates = openai_rag.vector_search(news)
            search_time = time.time() - start_time
            openai_search_times.append(search_time)
            openai_results.append(candidates)
        
        print(f"  ✓ 建立時間: {openai_build_time:.2f}秒")
        print(f"  ✓ 平均搜索時間: {np.mean(openai_search_times):.4f}秒")
        
        # 3. 比較結果品質
        print("\n3. 結果品質比較:")
        for i, news in enumerate(test_news):
            print(f"\n測試新聞 {i+1}: {news[:30]}...")
            
            # TF-IDF結果
            if tfidf_results[i]:
                top_tfidf = tfidf_results[i][0]
                print(f"  TF-IDF: 款次{top_tfidf['clause_number']} (相似度: {top_tfidf['similarity']:.4f})")
            else:
                print(f"  TF-IDF: 無結果")
            
            # OpenAI結果
            if openai_results[i]:
                top_openai = openai_results[i][0]
                print(f"  OpenAI: 款次{top_openai['clause_number']} (相似度: {top_openai['similarity']:.4f})")
            else:
                print(f"  OpenAI: 無結果")
        
        # 4. 效能總結
        print(f"\n=== 效能總結 ===")
        print(f"建立時間比較:")
        print(f"  TF-IDF: {tfidf_build_time:.2f}秒")
        print(f"  OpenAI: {openai_build_time:.2f}秒")
        print(f"  速度差異: {(openai_build_time/tfidf_build_time):.1f}x")
        
        print(f"\n搜索時間比較:")
        print(f"  TF-IDF: {np.mean(tfidf_search_times):.4f}秒")
        print(f"  OpenAI: {np.mean(openai_search_times):.4f}秒")
        print(f"  速度差異: {(np.mean(openai_search_times)/np.mean(tfidf_search_times)):.1f}x")
        
        print(f"\n特點比較:")
        print(f"  TF-IDF: 快速、免費、本地運算")
        print(f"  OpenAI: 高品質、語意理解、需要API費用")
    
    else:
        print("\n跳過OpenAI測試（未設定API金鑰）")
    
    print("\n=== 比較完成 ===")

if __name__ == "__main__":
    compare_performance()
