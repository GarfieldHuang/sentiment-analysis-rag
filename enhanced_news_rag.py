#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增強版RAG新聞比對系統 - 使用OpenAI Embedding與FAISS向量資料庫
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import jieba
import faiss
import openai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
import pickle
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

class EnhancedNewsRAG:
    def __init__(self, top_k=5, openai_api_key=None):
        """
        初始化增強版RAG系統
        
        Args:
            top_k: 向量搜索回傳的候選數量
            openai_api_key: OpenAI API金鑰
        """
        self.top_k = top_k
        self.vector_db = []
        self.consolidated_clauses = {}
        self.faiss_index = None
        self.embedding_dimension = 1536  # text-embedding-3-small 維度
        self.document_embeddings = None
        
        # 設定OpenAI
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # 嘗試從環境變數讀取
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = None
                print("警告：未設定OpenAI API金鑰，將使用基礎分析模式")
        if openai_api_key:
            openai.api_key = openai_api_key
        elif os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            print("警告: 未設定OpenAI API金鑰，請設定環境變數 OPENAI_API_KEY")
        
        self.client = openai.OpenAI()
    
    def clear_vector_database(self):
        """清空向量資料庫"""
        self.vector_db = []
        self.consolidated_clauses = {}
        self.faiss_index = None
        self.document_embeddings = None
        print("向量資料庫已清空")
    
    def consolidate_clauses(self, excel_path):
        """整併相同款次的資料"""
        try:
            # 清空現有的整併資料
            self.consolidated_clauses = {}
            
            df = pd.read_excel(excel_path)
            print(f"載入 {len(df)} 筆原始資料")
            
            # 檢查重訊款次欄位
            if '重訊款次' not in df.columns:
                print("錯誤：找不到'重訊款次'欄位")
                print(f"可用欄位: {list(df.columns)}")
                return {}
            
            # 按重訊款次分組整併
            consolidated = {}
            valid_count = 0
            
            for _, row in df.iterrows():
                clause_num = row['重訊款次']
                if pd.isna(clause_num) or clause_num == 'X':
                    continue
                
                valid_count += 1
                    
                # 處理款次編號，確保一致性
                if pd.notna(clause_num):
                    # 處理浮點數（如 1.0 -> 1）
                    if isinstance(clause_num, float) and clause_num.is_integer():
                        clause_num = int(clause_num)
                    elif str(clause_num).replace('.0', '').isdigit():
                        clause_num = int(float(clause_num))
                    else:
                        clause_num = str(clause_num).strip()
                
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
                
                # 整併各欄位資料
                clause_data = consolidated[clause_num]
                
                if pd.notna(row['類別']):
                    clause_data['categories'].add(str(row['類別']))
                if pd.notna(row['報導內容']):
                    clause_data['content_types'].add(str(row['報導內容']))
                if pd.notna(row['檢查項目']):
                    clause_data['check_items'].add(str(row['檢查項目']))
                if pd.notna(row['主體(如未敘明，即上市公司及重要子公司)']):
                    clause_data['subjects'].add(str(row['主體(如未敘明，即上市公司及重要子公司)']))
                if pd.notna(row['事件(邏輯)\n涵蓋什麼樣的關鍵字']):
                    clause_data['events'].add(str(row['事件(邏輯)\n涵蓋什麼樣的關鍵字']))
                if pd.notna(row['條款語意定義\n']):
                    clause_data['definitions'].add(str(row['條款語意定義\n']))
                if pd.notna(row['重大性判斷標準\n(若未敘明，都剪)']):
                    clause_data['standards'].add(str(row['重大性判斷標準\n(若未敘明，都剪)']))
                if pd.notna(row['不列剪報之備註']):
                    clause_data['notes'].add(str(row['不列剪報之備註']))
                if pd.notna(row['釋例新聞連結']):
                    clause_data['examples'].add(str(row['釋例新聞連結']))
                
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
            print(f"處理了 {valid_count} 筆有效資料，整併完成，共 {len(consolidated)} 個唯一款次")
            
            # 顯示整併統計
            merged_count = valid_count - len(consolidated)
            if merged_count > 0:
                print(f"合併了 {merged_count} 筆重複款次的資料")
                
            # 顯示款次清單
            clause_list = sorted(consolidated.keys())
            print(f"款次清單: {clause_list}")
            
            return consolidated
            
        except Exception as e:
            print(f"整併款次時發生錯誤: {e}")
            return {}
    
    def get_embeddings(self, texts):
        """使用OpenAI text-embedding-3-small獲取向量"""
        if not self.client:
            print("錯誤：未設定OpenAI API金鑰")
            return None
        
        try:
            # 預處理文字，確保不超過token限制
            processed_texts = []
            for text in texts:
                processed_text = self.preprocess_text(text)
                # 再次檢查長度，確保安全
                processed_text = self.truncate_text_for_embedding(processed_text, max_tokens=6000)
                processed_texts.append(processed_text)
            
            # 批次處理文字以提高效率，減小批次大小避免token超限
            embeddings = []
            batch_size = 20  # 減小批次大小
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                # 檢查批次中是否有空文字
                valid_batch = [text for text in batch if text.strip()]
                if not valid_batch:
                    continue
                
                try:
                    response = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=valid_batch
                    )
                    
                    for embedding_obj in response.data:
                        embeddings.append(embedding_obj.embedding)
                        
                    print(f"已處理 {i + len(valid_batch)}/{len(texts)} 個文字...")
                    
                except Exception as batch_error:
                    print(f"批次 {i//batch_size + 1} 處理錯誤: {batch_error}")
                    # 嘗試逐一處理這個批次
                    for single_text in valid_batch:
                        try:
                            single_response = self.client.embeddings.create(
                                model="text-embedding-3-small",
                                input=[single_text]
                            )
                            embeddings.append(single_response.data[0].embedding)
                        except Exception as single_error:
                            print(f"單一文字處理失敗: {single_error}")
                            # 添加零向量作為占位符
                            embeddings.append([0.0] * self.embedding_dimension)
                    
            return np.array(embeddings, dtype=np.float32) if embeddings else None
            
        except Exception as e:
            print(f"取得Embedding錯誤: {e}")
            return None

    def build_vector_database(self):
        """建立FAISS向量資料庫"""
        if not self.consolidated_clauses:
            print("請先執行款次整併")
            return False
        
        try:
            # 清空現有的向量資料庫
            documents = []
            self.vector_db = []
            self.faiss_index = None
            self.document_embeddings = None
            
            print("開始建立向量資料庫...")
            
            for clause_num, data in self.consolidated_clauses.items():
                documents.append(data['full_description'])
                
                point = {
                    'clause_number': clause_num,
                    'content': data['full_description'],
                    'consolidated_data': data
                }
                self.vector_db.append(point)

            if documents:
                print("正在生成Embedding向量...")
                embeddings = self.get_embeddings(documents)
                
                if embeddings is not None:
                    self.document_embeddings = embeddings
                    
                    # 建立FAISS索引
                    self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # 內積相似度
                    
                    # 正規化向量以便使用內積計算餘弦相似度
                    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # 添加向量到索引
                    self.faiss_index.add(normalized_embeddings)
                    
                    print(f"FAISS向量資料庫建立完成，共 {len(self.vector_db)} 個款次")
                    return True
                else:
                    print("無法取得Embedding向量")
                    return False
            else:
                print("沒有找到有效的款次資料")
                return False
                
        except Exception as e:
            print(f"建立向量資料庫錯誤: {e}")
            return False
    
    def save_vector_database(self, save_path="vector_db"):
        """儲存向量資料庫"""
        try:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, f"{save_path}_faiss.index")
                
            # 儲存其他資料
            with open(f"{save_path}_metadata.pkl", 'wb') as f:
                pickle.dump({
                    'vector_db': self.vector_db,
                    'consolidated_clauses': self.consolidated_clauses,
                    'document_embeddings': self.document_embeddings
                }, f)
                
            print(f"向量資料庫已儲存至 {save_path}")
            return True
            
        except Exception as e:
            print(f"儲存向量資料庫錯誤: {e}")
            return False
    
    def load_vector_database(self, load_path="vector_db"):
        """載入向量資料庫"""
        try:
            # 載入FAISS索引
            if os.path.exists(f"{load_path}_faiss.index"):
                self.faiss_index = faiss.read_index(f"{load_path}_faiss.index")
                
            # 載入其他資料
            if os.path.exists(f"{load_path}_metadata.pkl"):
                with open(f"{load_path}_metadata.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.vector_db = data['vector_db']
                    self.consolidated_clauses = data['consolidated_clauses']
                    self.document_embeddings = data['document_embeddings']
                    
                print(f"向量資料庫已從 {load_path} 載入")
                return True
            else:
                print(f"找不到向量資料庫檔案: {load_path}")
                return False
                
        except Exception as e:
            print(f"載入向量資料庫錯誤: {e}")
            return False
    
    def get_news_content(self, url):
        """擷取並清理新聞內容"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 移除不需要的標籤和元素
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'iframe', 'noscript', 'meta', 'link',
                           'ads', 'advertisement', '.ad', '.ads']):
                tag.decompose()
            
            # 移除廣告相關的class和id
            for element in soup.find_all(attrs={'class': re.compile(r'(ad|advertisement|banner|popup)', re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': re.compile(r'(ad|advertisement|banner|popup)', re.I)}):
                element.decompose()
            
            # 取得標題
            title = ""
            for selector in ['h1', 'h2', '.title', '.headline', 'title']:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    title = element.get_text().strip()
                    break
            
            # 嘗試找到主要內容區域
            main_content = ""
            content_selectors = [
                'article', '.article', '.content', '.main-content', 
                '.post-content', '.entry-content', '.story-content',
                'main', '.main', '.news-content', '.article-content'
            ]
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element.get_text()
                    break
            
            # 如果找不到主要內容區域，使用整個body
            if not main_content:
                body = soup.find('body')
                if body:
                    main_content = body.get_text()
                else:
                    main_content = soup.get_text()
            
            # 清理內容
            content = self.clean_html_content(main_content)
            
            # 合併標題和內容
            full_content = f"{title} {content}" if title else content
            
            return title, full_content
            
        except Exception as e:
            print(f"擷取新聞失敗: {e}")
            return "", ""
    
    def clean_html_content(self, text):
        """清理HTML內容"""
        if not text:
            return ""
        
        try:
            # 移除HTML標籤
            soup = BeautifulSoup(text, 'html.parser')
            
            # 移除不需要的元素
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'iframe', 'noscript', 'meta', 'link']):
                element.decompose()
            
            # 取得純文字
            clean_text = soup.get_text()
            
            # 清理多餘空白和換行
            clean_text = re.sub(r'\s+', ' ', clean_text)
            clean_text = re.sub(r'\n+', '\n', clean_text)
            
            return clean_text.strip()
            
        except Exception as e:
            print(f"HTML清理錯誤: {e}")
            return text

    def truncate_text_for_embedding(self, text, max_tokens=6000):
        """截斷文字以符合token限制"""
        if not text:
            return ""
        
        # 估算token數量（中文約1字=1.5-2 tokens，英文約1字=0.75 tokens）
        # 保守估計：1字=2 tokens
        estimated_tokens = len(text) * 2
        
        if estimated_tokens <= max_tokens:
            return text
        
        # 計算需要保留的字元數
        max_chars = max_tokens // 2
        
        # 優先保留開頭部分（標題和摘要通常在前面）
        if len(text) > max_chars:
            # 嘗試在句號或換行處截斷
            truncated = text[:max_chars]
            
            # 尋找最後一個句號或換行
            last_period = max(truncated.rfind('。'), truncated.rfind('\n'))
            if last_period > max_chars * 0.8:  # 如果截斷點在80%之後
                truncated = truncated[:last_period + 1]
            
            print(f"文字過長，已截斷至 {len(truncated)} 字元 (原長度: {len(text)})")
            return truncated
        
        return text

    def preprocess_text(self, text):
        """增強的文字預處理"""
        if not text:
            return ""
        
        # 1. 清理HTML
        clean_text = self.clean_html_content(text)
        
        # 2. 截斷過長文字
        truncated_text = self.truncate_text_for_embedding(clean_text)
        
        # 3. 對於OpenAI Embedding，不需要分詞，直接返回清理後的文字
        # OpenAI模型有自己的tokenization機制，能更好理解語意
        return truncated_text
    
    def preprocess_text_with_jieba(self, text):
        """使用jieba分詞的文字預處理（適用於TF-IDF等傳統方法）"""
        if not text:
            return ""
        
        # 1. 清理HTML
        clean_text = self.clean_html_content(text)
        
        # 2. 截斷過長文字
        truncated_text = self.truncate_text_for_embedding(clean_text)
        
        # 3. 中文分詞（適用於TF-IDF等需要分詞的傳統方法）
        words = jieba.cut(truncated_text)
        filtered_words = [word for word in words if len(word) > 1 and not word.isdigit()]
        
        return ' '.join(filtered_words)
    
    def vector_search(self, news_content):
        """使用FAISS進行向量搜索候選款次"""
        if self.faiss_index is None or not news_content:
            return []
        
        try:
            processed_content = self.preprocess_text(news_content)
            if not processed_content:
                return []
            
            # 取得新聞內容的Embedding
            news_embedding = self.get_embeddings([processed_content])
            if news_embedding is None:
                print("無法取得新聞內容的Embedding")
                return []
            
            # 正規化向量
            news_embedding = news_embedding / np.linalg.norm(news_embedding, axis=1, keepdims=True)
            
            # 使用FAISS搜索最相似的向量
            similarities, indices = self.faiss_index.search(news_embedding, self.top_k)
            
            candidates = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity > 0 and idx < len(self.vector_db):
                    candidate = {
                        'clause_number': self.vector_db[idx]['clause_number'],
                        'similarity': float(similarity),
                        'content': self.vector_db[idx]['content'],
                        'consolidated_data': self.vector_db[idx]['consolidated_data']
                    }
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            print(f"向量搜索錯誤: {e}")
            return []
            print(f"向量搜索錯誤: {e}")
            return []
    
    def analyze_with_gpt4o(self, news_title, news_content, news_url, candidates):
        """使用GPT-4o分析新聞並判斷符合的款次"""
        try:
            # 準備候選款次資訊
            candidates_info = []
            for candidate in candidates:
                data = candidate['consolidated_data']
                clause_info = {
                    'clause_number': candidate['clause_number'],
                    'categories': data['categories'],
                    'content_types': data['content_types'],
                    'check_items': data['check_items'],
                    'events': data['events'],
                    'definitions': data['definitions'],
                    'standards': data['standards'],
                    'similarity_score': candidate['similarity']
                }
                candidates_info.append(clause_info)
            
            # 建立提示詞
            prompt = f"""
你是一個專業的重大訊息分析專家，請分析以下新聞內容並判斷其符合哪個重訊款次。

新聞標題: {news_title}
新聞內容: {news_content[:2000]}...

候選的重訊款次（按向量相似度排序）:
{json.dumps(candidates_info, ensure_ascii=False, indent=2)}

請根據新聞內容分析：
1. 主要符合的款次編號及原因
2. 關鍵匹配要素
3. 信心分數（0-1）
4. 其他可能的款次（如果有）
5. 分析總結

請以JSON格式回應，符合以下結構：
{{
    "primary_match": {{
        "clause_number": 數字,
        "confidence_score": 0.0-1.0,
        "reason": "符合原因的詳細說明",
        "key_elements": ["關鍵要素1", "關鍵要素2"]
    }},
    "alternative_matches": [
        {{
            "clause_number": 數字,
            "confidence_score": 0.0-1.0,
            "reason": "符合原因說明",
            "key_elements": ["關鍵要素"]
        }}
    ],
    "analysis_summary": "整體分析總結"
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是專業的重大訊息分析師，請仔細分析新聞內容並判斷符合的重訊款次。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # 解析回應
            response_text = response.choices[0].message.content.strip()
            
            # 移除可能的markdown格式
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            gpt_result = json.loads(response_text)
            
            # 使用Pydantic驗證結果
            result = NewsAnalysisResult(
                news_title=news_title,
                news_url=news_url,
                primary_match=ClauseMatch(**gpt_result['primary_match']),
                alternative_matches=[ClauseMatch(**alt) for alt in gpt_result.get('alternative_matches', [])],
                analysis_summary=gpt_result.get('analysis_summary', '')
            )
            
            return result
            
        except Exception as e:
            print(f"GPT-4o分析錯誤: {e}")
            # 返回備用結果
            if candidates:
                return NewsAnalysisResult(
                    news_title=news_title,
                    news_url=news_url,
                    primary_match=ClauseMatch(
                        clause_number=candidates[0]['clause_number'],
                        confidence_score=candidates[0]['similarity'],
                        reason=f"向量相似度最高的款次（分數: {candidates[0]['similarity']:.4f}）",
                        key_elements=["向量相似度"]
                    ),
                    alternative_matches=[],
                    analysis_summary="GPT-4o分析失敗，使用向量相似度結果"
                )
            else:
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
                
                # 擷取新聞內容
                title, content = self.get_news_content(url)
                
                if not content or len(content.strip()) < 10:
                    print("無法擷取新聞內容或內容太短")
                    continue
                
                # 向量搜索候選款次
                candidates = self.vector_search(f"{title} {content}")
                
                if not candidates:
                    print("沒有找到候選款次")
                    continue
                
                # 使用GPT-4o分析
                print(f"使用GPT-4o分析...")
                analysis_result = self.analyze_with_gpt4o(title, content, url, candidates)
                
                if analysis_result:
                    results.append(analysis_result)
                    print(f"主要符合款次: {analysis_result.primary_match.clause_number}")
                    print(f"信心分數: {analysis_result.primary_match.confidence_score:.4f}")
                    print(f"符合原因: {analysis_result.primary_match.reason}")
                
            return results
            
        except Exception as e:
            print(f"處理新聞錯誤: {e}")
            return []
    
    def save_results(self, results, output_file='AI增強新聞比對結果.xlsx'):
        """儲存分析結果"""
        if not results:
            print("沒有結果可儲存")
            return
        
        try:
            output_data = []
            
            for result in results:
                # 主要匹配
                row = {
                    '新聞標題': result.news_title,
                    '新聞網址': result.news_url,
                    '符合款次': result.primary_match.clause_number,
                    '信心分數': round(result.primary_match.confidence_score, 4),
                    '符合原因': result.primary_match.reason,
                    '關鍵要素': ', '.join(result.primary_match.key_elements),
                    '匹配類型': '主要匹配',
                    '分析總結': result.analysis_summary
                }
                output_data.append(row)
                
                # 其他可能匹配
                for alt_match in result.alternative_matches:
                    row = {
                        '新聞標題': result.news_title,
                        '新聞網址': result.news_url,
                        '符合款次': alt_match.clause_number,
                        '信心分數': round(alt_match.confidence_score, 4),
                        '符合原因': alt_match.reason,
                        '關鍵要素': ', '.join(alt_match.key_elements),
                        '匹配類型': '其他可能',
                        '分析總結': ''
                    }
                    output_data.append(row)
            
            df = pd.DataFrame(output_data)
            df.to_excel(output_file, index=False)
            print(f"\n結果已儲存到: {output_file}")
            
        except Exception as e:
            print(f"儲存結果錯誤: {e}")

def main(force_rebuild=False):
    """主程式"""
    print("=== AI增強版RAG新聞比對系統 ===")
    
    # 檢查OpenAI API金鑰
    if not os.getenv("OPENAI_API_KEY"):
        print("請設定環境變數 OPENAI_API_KEY")
        print("例如: set OPENAI_API_KEY=your_api_key_here")
        return
    
    # 建立增強版RAG系統（需要OpenAI API金鑰）
    rag = EnhancedNewsRAG(top_k=5)
    
    # 檢查是否強制重建或載入失敗
    if force_rebuild:
        print("強制重建向量資料庫...")
        rag.clear_vector_database()
        build_new_db = True
    else:
        # 嘗試載入現有的向量資料庫
        build_new_db = not rag.load_vector_database()
    
    if build_new_db:
        if not force_rebuild:
            print("未找到現有向量資料庫，開始建立新的...")
        
        # 整併款次資料
        if not rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx"):
            print("無法整併款次資料")
            return
        
        # 建立向量資料庫
        if not rag.build_vector_database():
            print("無法建立向量資料庫")
            return
            
        # 儲存向量資料庫供下次使用
        rag.save_vector_database()
    else:
        print("成功載入現有向量資料庫")
    
    # 處理新聞
    results = rag.process_news("新聞.csv")
    
    if results:
        rag.save_results(results)
        print(f"\n=== 分析完成 ===")
        print(f"成功分析 {len(results)} 則新聞")
        
        # 顯示總結
        for result in results:
            print(f"\n新聞: {result.news_title}")
            print(f"符合款次: {result.primary_match.clause_number}")
            print(f"信心分數: {result.primary_match.confidence_score:.4f}")
            print(f"原因: {result.primary_match.reason}")
    else:
        print("沒有成功分析任何新聞")

if __name__ == "__main__":
    import sys
    
    # 檢查命令行參數
    force_rebuild = "--rebuild" in sys.argv or "-r" in sys.argv
    
    if force_rebuild:
        print("檢測到重建參數，將強制重建向量資料庫")
    
    main(force_rebuild=force_rebuild)
