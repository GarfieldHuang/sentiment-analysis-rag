#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最簡化RAG新聞比對系統
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

def load_excel_data(excel_path):
    """載入Excel檔案"""
    try:
        df = pd.read_excel(excel_path)
        print(f"成功載入 {len(df)} 筆重訊款次資料")
        
        # 建立文字描述
        documents = []
        vector_db = []
        
        for index in range(len(df)):
            row = df.iloc[index]
            doc_text = ""
            
            for col in df.columns:
                try:
                    cell_value = row[col]
                    if pd.notna(cell_value):
                        doc_text += str(cell_value) + " "
                except:
                    continue
            
            doc_text = doc_text.strip()
            if doc_text:
                documents.append(doc_text)
                
                point = {
                    'id': index,
                    'content': doc_text,
                    'raw_data': row.to_dict()
                }
                vector_db.append(point)
        
        return documents, vector_db
        
    except Exception as e:
        print(f"載入Excel檔案錯誤: {e}")
        return [], []

def get_news_content(url):
    """擷取新聞內容"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 移除不需要的標籤
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # 取得標題
        title = ""
        for selector in ['h1', 'h2', '.title', '.headline']:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                break
        
        # 取得內容
        content = soup.get_text()
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content)
        
        return title, content
        
    except Exception as e:
        print(f"擷取新聞失敗: {e}")
        return "", ""

def preprocess_text(text):
    """文字預處理"""
    if not text:
        return ""
    
    words = jieba.cut(text)
    filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
    return ' '.join(filtered_words)

def find_matches(news_content, documents, vector_db, vectorizer, document_vectors, top_k=10):
    """找到相似的重訊款次"""
    if not news_content or document_vectors is None:
        return []
    
    try:
        processed_content = preprocess_text(news_content)
        if not processed_content:
            return []
        
        news_vector = vectorizer.transform([processed_content])
        similarities = cosine_similarity(news_vector, document_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            if similarity_score > 0:
                result = {
                    'rank': len(results) + 1,
                    'regulation_id': int(vector_db[idx]['id']),
                    'similarity': similarity_score,
                    'content': vector_db[idx]['content'],
                    'raw_data': vector_db[idx]['raw_data']
                }
                results.append(result)
        
        return results
        
    except Exception as e:
        print(f"比對過程錯誤: {e}")
        return []

def main():
    """主程式"""
    print("=== 最簡化RAG新聞比對系統 ===")
    
    # 載入重訊款次資料
    excel_file = "../data/重訊款次整理_20250715彙整提供.xlsx"
    documents, vector_db = load_excel_data(excel_file)
    
    if not documents:
        print("無法載入重訊款次資料")
        return
    
    # 建立向量化器
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    document_vectors = vectorizer.fit_transform(documents)
    print(f"向量資料庫建立完成，共 {len(vector_db)} 個點")
    
    # 載入新聞資料
    try:
        news_df = pd.read_csv("../data/新聞.csv")
        print(f"載入 {len(news_df)} 筆新聞")
    except Exception as e:
        print(f"載入新聞錯誤: {e}")
        return
    
    all_results = []
    
    # 處理每則新聞
    for index in range(len(news_df)):
        row = news_df.iloc[index]
        url = str(row['新聞超連結']).strip()
        
        print(f"\n處理第 {index + 1} 則新聞: {url}")
        
        if not url or url == 'nan':
            print("無效的網址")
            continue
        
        # 擷取新聞內容
        title, content = get_news_content(url)
        
        if not content or len(content.strip()) < 10:
            print("無法擷取新聞內容或內容太短")
            continue
        
        # 合併標題和內容
        full_content = f"{title} {content}"
        
        # 找到相似的重訊款次
        matches = find_matches(full_content, documents, vector_db, vectorizer, document_vectors)
        
        if matches:
            result = {
                'news_index': index + 1,
                'news_url': url,
                'news_title': title,
                'news_content_preview': content[:200] + "..." if len(content) > 200 else content,
                'matches': matches
            }
            
            all_results.append(result)
            
            # 顯示結果
            print(f"標題: {title}")
            print(f"找到 {len(matches)} 個相似款次:")
            for match in matches[:3]:
                print(f"  排名 {match['rank']}: 相似度 {match['similarity']:.4f}")
                print(f"  內容: {match['content'][:80]}...")
        else:
            print("沒有找到相似的重訊款次")
    
    # 儲存結果
    if all_results:
        output_data = []
        
        for result in all_results:
            for match in result['matches']:
                row_data = {
                    '新聞編號': result['news_index'],
                    '新聞網址': result['news_url'],
                    '新聞標題': result['news_title'],
                    '新聞內容預覽': result['news_content_preview'],
                    '比對排名': match['rank'],
                    '相似度分數': round(match['similarity'], 4),
                    '重訊款次ID': match['regulation_id'],
                    '重訊款次內容': match['content'][:300] + "..." if len(match['content']) > 300 else match['content']
                }
                
                # 添加原始資料
                for key, value in match['raw_data'].items():
                    row_data[f'原始_{key}'] = value
                
                output_data.append(row_data)
        
        if output_data:
            df = pd.DataFrame(output_data)
            df.to_excel('../output/新聞比對結果.xlsx', index=False)
            print(f"\n結果已儲存到: 新聞比對結果.xlsx")
            print(f"=== 比對完成 ===")
            print(f"成功處理 {len(all_results)} 則新聞")
        else:
            print("沒有資料可儲存")
    else:
        print("沒有成功處理任何新聞")

if __name__ == "__main__":
    main()
