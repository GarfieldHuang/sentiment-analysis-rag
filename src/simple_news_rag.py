#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版RAG新聞比對系統
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

class SimpleNewsRAG:
    def __init__(self, top_k=10):
        """
        初始化簡化版RAG系統
        
        Args:
            top_k: KNN搜索的K值，預設為10
        """
        self.top_k = top_k
        self.vector_db = []
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.document_vectors = None
        self.regulations_data = None
        
    def load_excel_data(self, excel_path):
        """
        載入Excel檔案並建立向量資料庫
        """
        try:
            # 讀取Excel檔案的所有工作表
            excel_file = pd.ExcelFile(excel_path)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                df['工作表'] = sheet_name
                all_data.append(df)
            
            # 合併所有工作表
            combined_df = pd.concat(all_data, ignore_index=True)
            self.regulations_data = combined_df
            
            print(f"成功載入 {len(combined_df)} 筆重訊款次資料")
            
            # 建立文字描述
            documents = []
            for index, row in combined_df.iterrows():
                doc_text = ""
                for col in combined_df.columns:
                    cell_value = row[col]
                    if pd.notna(cell_value) and col != '工作表':
                        doc_text += str(cell_value) + " "
                
                documents.append(doc_text.strip())
                
                # 建立向量資料庫點
                point = {
                    'id': index,
                    'content': doc_text.strip(),
                    'raw_data': row.to_dict()
                }
                self.vector_db.append(point)
            
            # 向量化
            self.document_vectors = self.vectorizer.fit_transform(documents)
            print(f"向量資料庫建立完成，共 {len(self.vector_db)} 個點")
            
            return True
            
        except Exception as e:
            print(f"載入Excel檔案錯誤: {e}")
            return False
    
    def get_news_content(self, url):
        """
        簡化版新聞內容擷取
        """
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
            content = ""
            text = soup.get_text()
            
            # 清理文字
            content = re.sub(r'\s+', ' ', text)
            content = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content)
            
            return title, content
            
        except Exception as e:
            print(f"擷取新聞失敗 ({url}): {e}")
            return "", ""
    
    def preprocess_chinese_text(self, text):
        """
        中文文字預處理
        """
        # 使用jieba分詞
        words = jieba.cut(text)
        # 過濾短詞和數字
        filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
        return ' '.join(filtered_words)
    
    def find_top_k_matches(self, news_content):
        """
        找到最相似的K個重訊款次
        """
        if not self.document_vectors:
            return []
        
        # 預處理新聞內容
        processed_content = self.preprocess_chinese_text(news_content)
        
        # 向量化新聞
        news_vector = self.vectorizer.transform([processed_content])
        
        # 計算相似度
        similarities = cosine_similarity(news_vector, self.document_vectors)[0]
        
        # 取得top_k
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = float(similarities[idx])
            if similarity_score > 0:  # 只保留有相似度的結果
                result = {
                    'rank': len(results) + 1,
                    'regulation_id': int(self.vector_db[idx]['id']),
                    'similarity': similarity_score,
                    'content': self.vector_db[idx]['content'],
                    'raw_data': self.vector_db[idx]['raw_data']
                }
                results.append(result)
        
        return results
    
    def process_news_csv(self, csv_path):
        """
        處理新聞CSV檔案
        """
        try:
            news_df = pd.read_csv(csv_path)
            print(f"載入 {len(news_df)} 筆新聞")
            
            all_results = []
            
            for index, row in news_df.iterrows():
                url = row['新聞超連結']
                print(f"\n處理第 {index + 1} 則新聞: {url}")
                
                # 擷取新聞內容
                title, content = self.get_news_content(url)
                
                if not content or len(content.strip()) == 0:
                    print("無法擷取新聞內容")
                    continue
                
                # 合併標題和內容
                full_content = f"{title} {content}"
                
                # 找到相似的重訊款次
                matches = self.find_top_k_matches(full_content)
                
                result = {
                    'news_index': index + 1,
                    'news_url': url,
                    'news_title': title,
                    'news_content_preview': content[:200] + "...",
                    'matches': matches
                }
                
                all_results.append(result)
                
                # 顯示結果
                print(f"標題: {title}")
                print(f"找到 {len(matches)} 個相似款次:")
                for match in matches[:3]:  # 顯示前3個
                    print(f"  排名 {match['rank']}: 相似度 {match['similarity']:.4f}")
                    print(f"  內容: {match['content'][:80]}...")
            
            return all_results
            
        except Exception as e:
            print(f"處理新聞CSV錯誤: {e}")
            return []
    
    def save_results_to_excel(self, results, output_file='../output/新聞比對結果.xlsx'):
        """
        儲存結果到Excel
        """
        try:
            output_data = []
            
            for result in results:
                for match in result['matches']:
                    row = {
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
                        if key != '工作表':
                            row[f'原始_{key}'] = value
                    
                    output_data.append(row)
            
            df = pd.DataFrame(output_data)
            df.to_excel(output_file, index=False)
            print(f"\n結果已儲存到: {output_file}")
            
        except Exception as e:
            print(f"儲存結果錯誤: {e}")

def main():
    """
    主程式入口
    """
    print("=== RAG新聞比對系統 ===")
    
    # 設定檔案路徑
    excel_file = "../data/重訊款次整理_20250715彙整提供.xlsx"
    csv_file = "../data/新聞.csv"
    
    # 建立RAG系統 (可調整top_k參數)
    rag = SimpleNewsRAG(top_k=10)
    
    # 載入重訊款次資料
    if not rag.load_excel_data(excel_file):
        print("無法載入重訊款次資料，程式結束")
        return
    
    # 處理新聞並比對
    results = rag.process_news_csv(csv_file)
    
    if results:
        # 儲存結果
        rag.save_results_to_excel(results)
        print("\n=== 比對完成 ===")
        print(f"總共處理了 {len(results)} 則新聞")
    else:
        print("沒有處理任何新聞")

if __name__ == "__main__":
    main()
