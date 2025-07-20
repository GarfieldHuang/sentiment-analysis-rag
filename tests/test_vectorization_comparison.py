#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比較分詞與不分詞在向量化上的差異
"""

import os
import jieba
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def test_vectorization_methods():
    """測試不同向量化方法的差異"""
    
    print("=== 向量化方法比較測試 ===\n")
    
    # 測試文本
    test_texts = [
        "台積電宣布投資新廠房建設，預計投入資金100億美元用於擴大半導體產能",
        "上市公司發布年度財務報告，營收較去年同期大幅成長15%",
        "董事會決議發放現金股利每股2元，股東可於除息日後領取"
    ]
    
    query_text = "台積電投資新廠擴產"
    
    print("測試文本:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    print(f"\n查詢文本: {query_text}\n")
    
    # 1. OpenAI Embedding (不分詞)
    print("1. OpenAI Embedding (整篇文章)")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        client = OpenAI(api_key=api_key)
        
        try:
            # 獲取測試文本的embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=test_texts + [query_text]
            )
            
            embeddings = [item.embedding for item in response.data]
            test_embeddings = np.array(embeddings[:-1])
            query_embedding = np.array([embeddings[-1]])
            
            # 計算相似度
            similarities = cosine_similarity(query_embedding, test_embeddings)[0]
            
            print("相似度分數:")
            for i, sim in enumerate(similarities):
                print(f"  文本{i+1}: {sim:.4f}")
            
            print(f"最相似: 文本{np.argmax(similarities)+1}")
            
        except Exception as e:
            print(f"OpenAI Embedding 錯誤: {e}")
    else:
        print("未設定OpenAI API金鑰，跳過測試")
    
    print("\n" + "="*50)
    
    # 2. TF-IDF (需要分詞)
    print("\n2. TF-IDF + jieba分詞")
    
    # 分詞處理
    def segment_text(text):
        words = jieba.cut(text)
        return ' '.join([word for word in words if len(word) > 1 and not word.isdigit()])
    
    segmented_texts = [segment_text(text) for text in test_texts]
    segmented_query = segment_text(query_text)
    
    print("分詞結果:")
    for i, seg_text in enumerate(segmented_texts, 1):
        print(f"  文本{i}: {seg_text[:50]}...")
    print(f"  查詢: {segmented_query}")
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(segmented_texts)
    query_vector = vectorizer.transform([segmented_query])
    
    # 計算相似度
    tfidf_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    print("\n相似度分數:")
    for i, sim in enumerate(tfidf_similarities):
        print(f"  文本{i+1}: {sim:.4f}")
    
    print(f"最相似: 文本{np.argmax(tfidf_similarities)+1}")
    
    print("\n" + "="*50)
    
    # 3. TF-IDF (不分詞，字元級)
    print("\n3. TF-IDF (不分詞，字元級n-gram)")
    
    # 字元級向量化
    char_vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),  # 使用2-4字元的n-gram
        max_features=1000
    )
    
    char_matrix = char_vectorizer.fit_transform(test_texts)
    char_query_vector = char_vectorizer.transform([query_text])
    
    # 計算相似度
    char_similarities = cosine_similarity(char_query_vector, char_matrix)[0]
    
    print("相似度分數:")
    for i, sim in enumerate(char_similarities):
        print(f"  文本{i+1}: {sim:.4f}")
    
    print(f"最相似: 文本{np.argmax(char_similarities)+1}")
    
    print("\n" + "="*80)
    print("總結:")
    print("1. OpenAI Embedding: 深度語意理解，不需分詞")
    print("2. TF-IDF + 分詞: 需要分詞切割詞彙，統計詞頻")
    print("3. TF-IDF + 字元n-gram: 不需分詞，但可能抓不到語意")
    print("\n建議:")
    print("- 使用OpenAI Embedding時，直接輸入完整文章即可")
    print("- 使用TF-IDF等傳統方法時，中文需要分詞處理")

if __name__ == "__main__":
    test_vectorization_methods()
