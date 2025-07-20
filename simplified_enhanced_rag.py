#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版增強RAG系統 - 不需要OpenAI API
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Pydantic 模型定義
class ClauseMatch(BaseModel):
    """單一款次匹配結果"""
    clause_number: int = Field(..., description="符合的重訊款次編號")
    confidence_score: float = Field(..., ge=0, le=1, description="信心分數(0-1)")
    reason: str = Field(..., description="符合該款次的原因說明")
    key_elements: List[str] = Field(default=[], description="關鍵匹配要素")

class NewsAnalysisResult(BaseModel):
    """新聞分析結果"""
    news_title: str = Field(..., description="新聞標題")
    news_url: str = Field(..., description="新聞網址")
    primary_match: ClauseMatch = Field(..., description="主要匹配的款次")
    alternative_matches: List[ClauseMatch] = Field(default=[], description="其他可能匹配的款次")
    analysis_summary: str = Field(..., description="分析總結")

class SimplifiedEnhancedRAG:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.vector_db = []
        self.consolidated_clauses = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.document_vectors = None
    
    def consolidate_clauses(self, excel_path):
        """整併相同款次的資料"""
        try:
            df = pd.read_excel(excel_path)
            print(f"載入 {len(df)} 筆原始資料")
            
            consolidated = {}
            
            for _, row in df.iterrows():
                clause_num = row['重訊款次']
                if pd.isna(clause_num) or clause_num == 'X':
                    continue
                    
                clause_num = int(clause_num) if str(clause_num).isdigit() else str(clause_num)
                
                if clause_num not in consolidated:
                    consolidated[clause_num] = {
                        'clause_number': clause_num,
                        'categories': set(),
                        'content_types': set(),
                        'check_items': set(),
                        'subjects': set(),
                        'events': set(),
                        'definitions': set(),
                        'standards': set(),
                        'notes': set(),
                        'examples': set(),
                        'raw_rows': []
                    }
                
                clause_data = consolidated[clause_num]
                
                # 安全地添加各欄位資料
                safe_add = lambda field, target_set: target_set.add(str(field)) if pd.notna(field) else None
                
                safe_add(row['類別'], clause_data['categories'])
                safe_add(row['報導內容'], clause_data['content_types'])
                safe_add(row['檢查項目'], clause_data['check_items'])
                safe_add(row['主體(如未敘明，即上市公司及重要子公司)'], clause_data['subjects'])
                safe_add(row['事件(邏輯)\n涵蓋什麼樣的關鍵字'], clause_data['events'])
                safe_add(row['條款語意定義\n'], clause_data['definitions'])
                safe_add(row['重大性判斷標準\n(若未敘明，都剪)'], clause_data['standards'])
                safe_add(row['不列剪報之備註'], clause_data['notes'])
                safe_add(row['釋例新聞連結'], clause_data['examples'])
                
                clause_data['raw_rows'].append(row.to_dict())
            
            # 轉換集合為列表並建立文字描述
            for clause_num, data in consolidated.items():
                for key in ['categories', 'content_types', 'check_items', 'subjects', 
                           'events', 'definitions', 'standards', 'notes', 'examples']:
                    data[key] = list(data[key])
                
                # 建立完整的文字描述
                description_parts = []
                description_parts.append(f"重訊款次 {clause_num}")
                
                if data['categories']:
                    description_parts.append(f"類別: {', '.join(data['categories'])}")
                if data['content_types']:
                    description_parts.append(f"報導內容: {', '.join(data['content_types'])}")
                if data['check_items']:
                    description_parts.append(f"檢查項目: {', '.join(data['check_items'])}")
                if data['subjects']:
                    description_parts.append(f"主體: {', '.join(data['subjects'])}")
                if data['events']:
                    description_parts.append(f"事件邏輯: {', '.join(data['events'])}")
                if data['definitions']:
                    description_parts.append(f"條款定義: {', '.join(data['definitions'])}")
                if data['standards']:
                    description_parts.append(f"判斷標準: {', '.join(data['standards'])}")
                
                data['full_description'] = ' '.join(description_parts)
            
            self.consolidated_clauses = consolidated
            print(f"整併完成，共 {len(consolidated)} 個唯一款次")
            
            return consolidated
            
        except Exception as e:
            print(f"整併款次時發生錯誤: {e}")
            return {}
    
    def build_vector_database(self):
        """建立向量資料庫"""
        if not self.consolidated_clauses:
            return False
        
        try:
            documents = []
            self.vector_db = []
            
            for clause_num, data in self.consolidated_clauses.items():
                documents.append(data['full_description'])
                
                point = {
                    'clause_number': clause_num,
                    'content': data['full_description'],
                    'consolidated_data': data
                }
                self.vector_db.append(point)
            
            if documents:
                self.document_vectors = self.vectorizer.fit_transform(documents)
                print(f"向量資料庫建立完成，共 {len(self.vector_db)} 個款次")
                return True
            
            return False
                
        except Exception as e:
            print(f"建立向量資料庫錯誤: {e}")
            return False
    
    def get_news_content(self, url):
        """擷取新聞內容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            title = ""
            for selector in ['h1', 'h2', '.title', '.headline']:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text().strip()
                    break
            
            content = soup.get_text()
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
        
        words = jieba.cut(text)
        filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
        return ' '.join(filtered_words)
    
    def vector_search(self, news_content):
        """向量搜索候選款次"""
        if self.document_vectors is None or not news_content:
            return []
        
        try:
            processed_content = self.preprocess_text(news_content)
            if not processed_content:
                return []
            
            news_vector = self.vectorizer.transform([processed_content])
            similarities = cosine_similarity(news_vector, self.document_vectors)[0]
            
            top_indices = np.argsort(similarities)[::-1][:self.top_k]
            
            candidates = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    candidate = {
                        'clause_number': self.vector_db[idx]['clause_number'],
                        'similarity': float(similarities[idx]),
                        'content': self.vector_db[idx]['content'],
                        'consolidated_data': self.vector_db[idx]['consolidated_data']
                    }
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"向量搜索錯誤: {e}")
            return []
    
    def analyze_with_rules(self, news_title, news_content, news_url, candidates):
        """使用規則分析新聞並判斷符合的款次"""
        if not candidates:
            return None
        
        try:
            # 取得最相似的候選款次
            best_candidate = candidates[0]
            clause_data = best_candidate['consolidated_data']
            
            # 分析關鍵要素
            key_elements = []
            reason_parts = []
            
            # 檢查類別匹配
            if clause_data['categories']:
                for category in clause_data['categories']:
                    if category in news_content or category in news_title:
                        key_elements.append(f"類別匹配: {category}")
                        reason_parts.append(f"新聞內容與「{category}」類別相關")
            
            # 檢查報導內容匹配
            if clause_data['content_types']:
                for content_type in clause_data['content_types']:
                    if content_type in news_content or content_type in news_title:
                        key_elements.append(f"內容類型匹配: {content_type}")
                        reason_parts.append(f"涉及「{content_type}」相關事項")
            
            # 檢查關鍵字匹配
            if clause_data['events']:
                for event in clause_data['events']:
                    # 簡單的關鍵字匹配
                    event_keywords = event.split()
                    for keyword in event_keywords:
                        if len(keyword) > 2 and (keyword in news_content or keyword in news_title):
                            key_elements.append(f"關鍵字匹配: {keyword}")
                            reason_parts.append(f"包含關鍵字「{keyword}」")
                            break
            
            # 如果沒有找到具體匹配，使用向量相似度
            if not key_elements:
                key_elements.append("向量相似度匹配")
                reason_parts.append(f"與該款次的向量相似度最高（{best_candidate['similarity']:.4f}）")
            
            # 建立主要匹配結果
            primary_match = ClauseMatch(
                clause_number=best_candidate['clause_number'],
                confidence_score=min(best_candidate['similarity'] * 2, 1.0),  # 調整信心分數
                reason='; '.join(reason_parts),
                key_elements=key_elements
            )
            
            # 建立其他可能匹配
            alternative_matches = []
            for candidate in candidates[1:3]:  # 取前2個替代選項
                if candidate['similarity'] > 0.05:  # 只保留有一定相似度的
                    alt_match = ClauseMatch(
                        clause_number=candidate['clause_number'],
                        confidence_score=candidate['similarity'],
                        reason=f"向量相似度為 {candidate['similarity']:.4f}",
                        key_elements=["向量相似度匹配"]
                    )
                    alternative_matches.append(alt_match)
            
            # 建立分析總結
            analysis_summary = f"基於向量相似度分析，新聞「{news_title}」最符合第{best_candidate['clause_number']}款重訊項目"
            if len(candidates) > 1:
                analysis_summary += f"，另有{len(alternative_matches)}個可能的替代選項"
            
            result = NewsAnalysisResult(
                news_title=news_title,
                news_url=news_url,
                primary_match=primary_match,
                alternative_matches=alternative_matches,
                analysis_summary=analysis_summary
            )
            
            return result
            
        except Exception as e:
            print(f"規則分析錯誤: {e}")
            return None
    
    def process_news(self, csv_path):
        """處理新聞並進行分析"""
        try:
            news_df = pd.read_csv(csv_path)
            print(f"載入 {len(news_df)} 筆新聞")
            
            results = []
            
            for index in range(len(news_df)):
                row = news_df.iloc[index]
                url = str(row['新聞超連結']).strip()
                
                print(f"\n處理第 {index + 1} 則新聞: {url}")
                
                if not url or url == 'nan':
                    continue
                
                title, content = self.get_news_content(url)
                
                if not content or len(content.strip()) < 10:
                    print("無法擷取新聞內容或內容太短")
                    continue
                
                candidates = self.vector_search(f"{title} {content}")
                
                if not candidates:
                    print("沒有找到候選款次")
                    continue
                
                analysis_result = self.analyze_with_rules(title, content, url, candidates)
                
                if analysis_result:
                    results.append(analysis_result)
                    print(f"主要符合款次: {analysis_result.primary_match.clause_number}")
                    print(f"信心分數: {analysis_result.primary_match.confidence_score:.4f}")
                    print(f"符合原因: {analysis_result.primary_match.reason}")
                
            return results
            
        except Exception as e:
            print(f"處理新聞錯誤: {e}")
            return []
    
    def save_results(self, results, output_file='簡化版AI新聞比對結果.xlsx'):
        """儲存分析結果"""
        if not results:
            print("沒有結果可儲存")
            return
        
        try:
            output_data = []
            
            for result in results:
                row = {
                    '新聞標題': result.news_title,
                    '新聞網址': result.news_url,
                    '符合第幾款': result.primary_match.clause_number,
                    '信心分數': round(result.primary_match.confidence_score, 4),
                    '符合原因說明': result.primary_match.reason,
                    '關鍵匹配要素': ', '.join(result.primary_match.key_elements),
                    '匹配類型': '主要匹配',
                    '分析總結': result.analysis_summary
                }
                output_data.append(row)
                
                for alt_match in result.alternative_matches:
                    row = {
                        '新聞標題': result.news_title,
                        '新聞網址': result.news_url,
                        '符合第幾款': alt_match.clause_number,
                        '信心分數': round(alt_match.confidence_score, 4),
                        '符合原因說明': alt_match.reason,
                        '關鍵匹配要素': ', '.join(alt_match.key_elements),
                        '匹配類型': '其他可能',
                        '分析總結': ''
                    }
                    output_data.append(row)
            
            df = pd.DataFrame(output_data)
            df.to_excel(output_file, index=False)
            print(f"\n結果已儲存到: {output_file}")
            
        except Exception as e:
            print(f"儲存結果錯誤: {e}")

def main():
    """主程式"""
    print("=== 簡化版AI增強RAG新聞比對系統 ===")
    
    rag = SimplifiedEnhancedRAG(top_k=5)
    
    if not rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx"):
        print("無法整併款次資料")
        return
    
    if not rag.build_vector_database():
        print("無法建立向量資料庫")
        return
    
    results = rag.process_news("新聞.csv")
    
    if results:
        rag.save_results(results)
        print(f"\n=== 分析完成 ===")
        print(f"成功分析 {len(results)} 則新聞")
        
        # 顯示總結
        print("\n=== 分析結果總結 ===")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.news_title}")
            print(f"   符合第 {result.primary_match.clause_number} 款")
            print(f"   信心分數: {result.primary_match.confidence_score:.4f}")
            print(f"   原因: {result.primary_match.reason}")
            print()
    else:
        print("沒有成功分析任何新聞")

if __name__ == "__main__":
    main()
