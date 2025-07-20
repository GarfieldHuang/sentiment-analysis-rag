#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG新聞比對系統
比對新聞內容與重訊款次的向量資料庫系統
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class NewsMatcherRAG:
    def __init__(self, excel_file_path, csv_file_path, top_k=10):
        """
        初始化RAG新聞比對系統
        
        Args:
            excel_file_path: 重訊款次Excel檔案路徑
            csv_file_path: 新聞CSV檔案路徑
            top_k: KNN搜索的K值，預設為10
        """
        self.excel_file_path = excel_file_path
        self.csv_file_path = csv_file_path
        self.top_k = top_k
        self.vector_db = []  # 向量資料庫
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.document_vectors = None
        self.regulations_data = None
        
        # 初始化jieba
        jieba.set_dictionary('dict.txt.big')  # 使用繁體中文字典
        
    def load_regulations_data(self):
        """
        載入重訊款次Excel檔案並建立向量資料庫
        """
        try:
            # 讀取Excel檔案
            df = pd.read_excel(self.excel_file_path)
            print(f"成功載入重訊款次檔案，共 {len(df)} 筆資料")
            
            # 假設Excel檔案的欄位包含款次資訊
            # 您可能需要根據實際Excel結構調整欄位名稱
            self.regulations_data = df
            
            # 建立文字描述用於向量化
            documents = []
            for index, row in df.iterrows():
                # 合併所有文字欄位作為文件內容
                doc_text = ""
                for col in df.columns:
                    if pd.notna(row[col]):
                        doc_text += str(row[col]) + " "
                
                documents.append(doc_text.strip())
                
                # 建立向量資料庫點
                point = {
                    'id': index,
                    'content': doc_text.strip(),
                    'raw_data': row.to_dict()
                }
                self.vector_db.append(point)
            
            # 向量化文件
            self.document_vectors = self.vectorizer.fit_transform(documents)
            print(f"向量資料庫建立完成，共 {len(self.vector_db)} 個點")
            
            return True
            
        except Exception as e:
            print(f"載入重訊款次檔案時發生錯誤: {e}")
            return False
    
    def extract_news_content(self, url):
        """
        從網址擷取新聞內容
        
        Args:
            url: 新聞網址
            
        Returns:
            新聞標題和內容
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除不需要的元素
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # 嘗試找到標題
            title = ""
            title_selectors = ['h1', 'h2', '.title', '.headline', '[class*="title"]']
            for selector in title_selectors:
                title_element = soup.select_one(selector)
                if title_element:
                    title = title_element.get_text().strip()
                    break
            
            # 嘗試找到內容
            content = ""
            content_selectors = [
                '.content', '.article-content', '.news-content', 
                '[class*="content"]', '[class*="article"]', 'article', 'main'
            ]
            
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    content = content_element.get_text().strip()
                    break
            
            # 如果找不到特定的內容區域，使用整個body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            # 清理文字
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content)
            
            return title, content
            
        except Exception as e:
            print(f"擷取新聞內容時發生錯誤 ({url}): {e}")
            return "", ""
    
    def preprocess_text(self, text):
        """
        文字預處理
        
        Args:
            text: 輸入文字
            
        Returns:
            處理後的文字
        """
        # 使用jieba進行中文分詞
        words = jieba.cut(text)
        
        # 過濾停用詞和短詞
        filtered_words = [word for word in words if len(word) > 1]
        
        return ' '.join(filtered_words)
    
    def find_similar_regulations(self, news_content):
        """
        使用KNN找到相似的重訊款次
        
        Args:
            news_content: 新聞內容
            
        Returns:
            相似的重訊款次列表
        """
        if not self.document_vectors:
            print("向量資料庫尚未建立")
            return []
        
        # 預處理新聞內容
        processed_content = self.preprocess_text(news_content)
        
        # 向量化新聞內容
        news_vector = self.vectorizer.transform([processed_content])
        
        # 計算相似度
        similarities = cosine_similarity(news_vector, self.document_vectors)[0]
        
        # 取得top_k個最相似的結果
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            result = {
                'regulation_id': self.vector_db[idx]['id'],
                'similarity_score': similarities[idx],
                'content': self.vector_db[idx]['content'],
                'raw_data': self.vector_db[idx]['raw_data']
            }
            results.append(result)
        
        return results
    
    def load_news_data(self):
        """
        載入新聞CSV檔案
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            print(f"成功載入新聞檔案，共 {len(df)} 筆新聞")
            return df
        except Exception as e:
            print(f"載入新聞檔案時發生錯誤: {e}")
            return None
    
    def process_all_news(self):
        """
        處理所有新聞並進行比對
        """
        # 載入重訊款次資料
        if not self.load_regulations_data():
            return
        
        # 載入新聞資料
        news_df = self.load_news_data()
        if news_df is None:
            return
        
        results = []
        
        for index, row in news_df.iterrows():
            url = row['新聞超連結']
            print(f"\n處理新聞 {index + 1}/{len(news_df)}: {url}")
            
            # 擷取新聞內容
            title, content = self.extract_news_content(url)
            
            if not content:
                print(f"無法擷取新聞內容: {url}")
                continue
            
            # 合併標題和內容
            full_content = f"{title} {content}"
            
            # 找到相似的重訊款次
            similar_regulations = self.find_similar_regulations(full_content)
            
            result = {
                'news_url': url,
                'news_title': title,
                'news_content': content[:500] + "..." if len(content) > 500 else content,
                'similar_regulations': similar_regulations
            }
            
            results.append(result)
            
            # 顯示結果
            print(f"新聞標題: {title}")
            print(f"找到 {len(similar_regulations)} 個相似的重訊款次:")
            for i, reg in enumerate(similar_regulations[:3]):  # 只顯示前3個
                print(f"  {i+1}. 相似度: {reg['similarity_score']:.4f}")
                print(f"     內容: {reg['content'][:100]}...")
        
        return results
    
    def save_results(self, results, output_file='../output/news_matching_results.xlsx'):
        """
        儲存比對結果
        
        Args:
            results: 比對結果
            output_file: 輸出檔案名稱
        """
        try:
            output_data = []
            
            for result in results:
                for i, reg in enumerate(result['similar_regulations']):
                    row = {
                        '新聞網址': result['news_url'],
                        '新聞標題': result['news_title'],
                        '排名': i + 1,
                        '相似度分數': reg['similarity_score'],
                        '重訊款次ID': reg['regulation_id'],
                        '重訊款次內容': reg['content'][:200] + "..." if len(reg['content']) > 200 else reg['content']
                    }
                    
                    # 添加重訊款次的原始資料
                    for key, value in reg['raw_data'].items():
                        row[f'重訊_{key}'] = value
                    
                    output_data.append(row)
            
            df = pd.DataFrame(output_data)
            df.to_excel(output_file, index=False)
            print(f"\n結果已儲存到: {output_file}")
            
        except Exception as e:
            print(f"儲存結果時發生錯誤: {e}")

def main():
    """
    主程式
    """
    # 設定檔案路徑
    excel_file = "../data/重訊款次整理_20250715彙整提供.xlsx"
    csv_file = "../data/新聞.csv"
    
    # 建立RAG系統
    matcher = NewsMatcherRAG(excel_file, csv_file, top_k=10)
    
    # 處理所有新聞
    results = matcher.process_all_news()
    
    if results:
        # 儲存結果
        matcher.save_results(results)
        print("\n新聞比對完成!")
    else:
        print("處理過程中發生錯誤")

if __name__ == "__main__":
    main()
