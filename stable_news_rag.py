#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
穩定版RAG新聞比對系統
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

class StableNewsRAG:
    def __init__(self, top_k=10):
        self.top_k = top_k
        self.vector_db = []
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.document_vectors = None
        self.regulations_data = None
        
    def load_excel_data(self, excel_path):
        """載入Excel檔案"""
        try:
            excel_file = pd.ExcelFile(excel_path)
            all_data = []
            
            for sheet_name in excel_file.sheet_names:
                print(f"讀取工作表: {sheet_name}")
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                df['工作表'] = sheet_name
                all_data.append(df)
            
            combined_df = pd.concat(all_data, ignore_index=True)
            self.regulations_data = combined_df
            
            print(f"成功載入 {len(combined_df)} 筆重訊款次資料")
            
            # 建立文字描述
            documents = []
            for index, row in combined_df.iterrows():
                doc_text = ""
                for col in combined_df.columns:
                    try:
                        cell_value = row[col]
                        if pd.notna(cell_value) and str(cell_value).strip() != '' and col != '工作表':
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
                    self.vector_db.append(point)
            
            if documents:
                self.document_vectors = self.vectorizer.fit_transform(documents)
                print(f"向量資料庫建立完成，共 {len(self.vector_db)} 個點")
                return True
            else:
                print("沒有找到有效的文字資料")
                return False
            
        except Exception as e:
            print(f"載入Excel檔案錯誤: {e}")
            return False
    
    def get_news_content(self, url):
        """擷取新聞內容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除不需要的標籤
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                tag.decompose()
            
            # 取得標題
            title = ""
            title_selectors = ['h1', 'h2', '.title', '.headline', '[class*="title"]']
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text().strip()
                    break
            
            # 取得內容
            content = ""
            content_selectors = [
                '.content', '.article-content', '.news-content', 
                '[class*="content"]', '[class*="article"]', 'article', 'main'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text().strip()
                    break
            
            # 如果沒找到特定內容區域，使用body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            # 清理文字
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', content)
            
            return title, content
            
        except Exception as e:
            print(f"擷取新聞失敗: {e}")
            return "", ""
    
    def preprocess_text(self, text):
        """文字預處理"""
        if not text:
            return ""
            
        # 使用jieba分詞
        words = jieba.cut(text)
        # 過濾短詞和數字
        filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
        return ' '.join(filtered_words)
    
    def find_matches(self, news_content):
        """找到相似的重訊款次"""
        if not self.document_vectors or not news_content:
            return []
        
        try:
            # 預處理新聞內容
            processed_content = self.preprocess_text(news_content)
            
            if not processed_content:
                return []
            
            # 向量化新聞
            news_vector = self.vectorizer.transform([processed_content])
            
            # 計算相似度
            similarities = cosine_similarity(news_vector, self.document_vectors)[0]
            
            # 取得top_k
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            results = []
            for idx in top_indices:
                similarity_score = float(similarities[idx])
                if similarity_score > 0:
                    result = {
                        'rank': len(results) + 1,
                        'regulation_id': int(self.vector_db[idx]['id']),
                        'similarity': similarity_score,
                        'content': self.vector_db[idx]['content'],
                        'raw_data': self.vector_db[idx]['raw_data']
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"比對過程錯誤: {e}")
            return []
    
    def process_news(self, csv_path):
        """處理新聞CSV"""
        try:
            news_df = pd.read_csv(csv_path)
            print(f"載入 {len(news_df)} 筆新聞")
            
            all_results = []
            
            for index, row in news_df.iterrows():
                url = str(row['新聞超連結']).strip()
                print(f"\n處理第 {index + 1} 則新聞: {url}")
                
                if not url or url == 'nan':
                    print("無效的網址")
                    continue
                
                # 擷取新聞內容
                title, content = self.get_news_content(url)
                
                if not content or len(content.strip()) < 10:
                    print("無法擷取新聞內容或內容太短")
                    continue
                
                # 合併標題和內容
                full_content = f"{title} {content}"
                
                # 找到相似的重訊款次
                matches = self.find_matches(full_content)
                
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
            
            return all_results
            
        except Exception as e:
            print(f"處理新聞錯誤: {e}")
            return []
    
    def save_results(self, results, output_file='新聞比對結果.xlsx'):
        """儲存結果"""
        if not results:
            print("沒有結果可儲存")
            return
            
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
            
            if output_data:
                df = pd.DataFrame(output_data)
                df.to_excel(output_file, index=False)
                print(f"\n結果已儲存到: {output_file}")
            else:
                print("沒有資料可儲存")
                
        except Exception as e:
            print(f"儲存結果錯誤: {e}")

def main():
    """主程式"""
    print("=== 穩定版RAG新聞比對系統 ===")
    
    # 檔案路徑
    excel_file = "重訊款次整理_20250715彙整提供.xlsx"
    csv_file = "新聞.csv"
    
    # 建立RAG系統
    rag = StableNewsRAG(top_k=10)
    
    # 載入重訊款次資料
    if not rag.load_excel_data(excel_file):
        print("無法載入重訊款次資料")
        return
    
    # 處理新聞
    results = rag.process_news(csv_file)
    
    if results:
        rag.save_results(results)
        print(f"\n=== 比對完成 ===")
        print(f"成功處理 {len(results)} 則新聞")
    else:
        print("沒有成功處理任何新聞")

if __name__ == "__main__":
    main()
