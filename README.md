# RAG新聞比對系統使用說明

## 系統概述
這是一個基於RAG（Retrieval-Augmented Generation）技術的新聞比對系統，用於比對新聞內容與重訊款次。系統提供兩種向量化技術：**TF-IDF本地運算**和**OpenAI Embedding + FAISS**，後者提供更高品質的語意理解能力。

## 檔案說明
- `simplified_enhanced_rag.py`: TF-IDF版本（免費、快速）
- `enhanced_news_rag.py`: **OpenAI Embedding + FAISS版本**（高品質、需API）
- `minimal_news_rag.py`: 基礎版程式
- `test_openai_faiss_rag.py`: 測試OpenAI+FAISS系統
- `compare_rag_performance.py`: 效能比較程式
- `view_enhanced_results.py`: 查看結果
- `setup_openai.py`: 設定OpenAI API金鑰
- `requirements.txt`: 所需套件清單（包含faiss-cpu）
- `README.md`: 使用說明

## 向量技術比較

| 特性 | TF-IDF版本 | OpenAI Embedding版本 |
|------|------------|---------------------|
| **語意理解** | 關鍵字匹配 | 深度語意理解 |
| **成本** | 免費 | 需要API費用 |
| **速度** | 極快 | 較慢（需API呼叫） |
| **準確度** | 中等 | 高 |
| **離線使用** | ✅ | ❌ |
| **向量維度** | 1000維（可調） | 1536維 |
| **推薦用途** | 快速原型、本地測試 | 正式系統、高準確度需求 |

## 安裝需求
```bash
pip install -r requirements.txt
```

所需套件包含：
- pandas: 資料處理
- numpy: 數值計算
- requests: 網路請求
- beautifulsoup4: HTML解析
- jieba: 中文分詞
- scikit-learn: 機器學習（TF-IDF版本）
- **faiss-cpu: 高效向量搜索（OpenAI版本）**
- openpyxl: Excel檔案處理
- pydantic: 型別檢查
- **openai: OpenAI API（需要）**
- openai: OpenAI API（可選）

## 使用方法

### 1. 準備資料檔案
確保工作目錄中有以下檔案：
- `重訊款次整理_20250715彙整提供.xlsx`: 重訊款次資料（93筆原始資料）
- `新聞.csv`: 新聞超連結清單（12筆新聞）

### 2. 選擇執行版本

#### A. OpenAI Embedding版本（推薦，高品質）
```bash
# 1. 設定OpenAI API金鑰
export OPENAI_API_KEY="your-api-key-here"
# 或執行設定程式
python setup_openai.py

# 2. 執行OpenAI版本
python enhanced_news_rag.py

# 3. 測試系統功能
python test_openai_faiss_rag.py
```

#### B. TF-IDF版本（免費，快速）
```bash
# 執行TF-IDF版本（無需API金鑰）
python simplified_enhanced_rag.py

# 測試系統
python test_enhanced_rag.py
```

#### C. 向量資料庫管理
```bash
# 強制重建向量資料庫
python enhanced_news_rag.py --rebuild

# 專用重建工具
python rebuild_vector_db.py

# 檢查款次整併結果
python test_clause_consolidation.py
```

#### D. 效能與測試
```bash
# 比較兩種版本的效能差異
python compare_rag_performance.py

# 測試向量化方法差異
python test_vectorization_comparison.py

# 測試內容清理和token限制
python test_content_cleaning.py
```

# 查看結果
python view_enhanced_results.py
```

### 3. 測試系統功能
```bash
# 測試OpenAI+FAISS系統
python test_openai_faiss_rag.py

# 比較向量化方法（分詞 vs 不分詞）
python test_vectorization_comparison.py

# 測試內容清理和token限制
python test_content_cleaning.py

# 測試TF-IDF版本
python test_enhanced_rag.py
```

### 4. 調整參數
可以在程式中調整以下參數：
- `top_k`: KNN搜索的K值（預設為5）
- `max_tokens`: Embedding最大token數（預設為6000）
- `batch_size`: API批次大小（預設為20）
- `max_features`: TF-IDF最大特徵數（預設為1000）
- `ngram_range`: N-gram範圍（預設為(1, 2)）

## 系統功能

### 1. 款次整併
- 將93筆原始資料整併為43個唯一重訊款次
- 合併相同款次的多筆記錄
- 整合類別、報導內容、檢查項目等資訊
- 建立完整的文字描述用於向量化

### 2. 向量資料庫建立
- 讀取Excel檔案中的重訊款次資料
- 將每個整併後的款次轉換為文字描述
- 清理和預處理文字，移除HTML標籤
- 使用OpenAI Embedding或TF-IDF向量化建立向量資料庫
- 每個款次對應一個高維向量點

### 3. 新聞內容擷取與清理
- 從CSV檔案讀取新聞網址
- 自動擷取新聞標題和內容
- **HTML內容清理**：移除script、style、廣告等無關元素
- **智能內容截取**：自動識別主要內容區域
- **Token限制處理**：自動截斷過長文字以符合API限制
- 使用jieba進行中文分詞

### 4. 智能比對分析
- 使用餘弦相似度計算新聞與款次的相似度
- FAISS高效向量搜索或KNN演算法找到最相似的候選款次
- 規則分析：結合關鍵字匹配、類別匹配
- 信心分數計算：綜合評估匹配可信度

### 5. Pydantic型別檢查
- 使用Pydantic模型確保輸出資料型別正確
- 驗證信心分數範圍（0-1）
- 檢查必要欄位完整性

### 6. 結果輸出
- 儲存比對結果到Excel檔案
- 包含符合第幾款、信心分數、原因說明
- 提供主要匹配和替代選項
- 結果檔案：`簡化版AI新聞比對結果.xlsx`

## 輸出格式

### 增強版結果檔案（推薦）
Excel檔案包含以下欄位：
- **新聞標題**: 新聞的標題
- **新聞網址**: 新聞的網址連結
- **符合第幾款**: 重訊款次編號
- **信心分數**: 匹配的信心度（0-1）
- **符合原因說明**: 詳細的符合原因
- **關鍵匹配要素**: 匹配的關鍵元素
- **匹配類型**: 主要匹配或其他可能
- **分析總結**: 整體分析結果

### 基礎版結果檔案
Excel檔案包含以下欄位：
- 新聞編號
- 新聞網址
- 新聞標題
- 新聞內容預覽
- 比對排名
- 相似度分數
- 重訊款次ID
- 重訊款次內容
- 原始資料欄位

## 技術說明

### 內容清理與預處理
- **HTML清理**：移除script、style、nav、header、footer等標籤
- **廣告過濾**：自動識別並移除廣告相關元素
- **主要內容提取**：優先提取article、.content等主要內容區域
- **Token限制**：自動截斷過長文字，避免超過OpenAI API限制
- **智能截斷**：在句號或換行處截斷，保持語意完整

### 款次整併技術
- 按重訊款次編號分組
- 合併相同款次的多筆記錄
- 整合類別、內容、檢查項目等資訊
- 建立完整的文字描述

### 向量化方法
**OpenAI Embedding版本**：
- 模型：text-embedding-3-small
- 維度：1536維高品質向量
- 批次處理：20個文字/批次
- 錯誤恢復：自動重試機制

**TF-IDF版本**：
### 向量化技術深度解析

#### 🤖 OpenAI Embedding（推薦）
- **輸入方式**：整篇文章直接輸入，無需分詞
- **技術原理**：Transformer模型，理解語意和上下文
- **處理示例**：
  ```
  輸入："台積電宣布投資新廠房建設，預計投入資金100億美元"
  輸出：1536維高品質語意向量
  ```
- **優勢**：保留完整語意、理解句子關聯、多語言支援

#### 📊 TF-IDF（傳統方法）
- **輸入方式**：需要分詞處理，以詞彙為單位
- **技術原理**：統計詞頻和逆文件頻率
- **處理示例**：
  ```
  原文："台積電宣布投資新廠房建設"
  分詞："台積電 宣布 投資 新廠房 建設"
  輸出：1000維稀疏向量
  ```
- **為什麼需要分詞**：
  - 中文沒有空格分隔詞彙
  - TF-IDF需要明確的詞邊界來統計詞頻
  - 分詞品質直接影響向量品質

#### 🔍 技術比較
| 特性 | OpenAI Embedding | TF-IDF + 分詞 |
|------|-----------------|---------------|
| **是否需要分詞** | ❌ 不需要 | ✅ 必須 |
| **語意理解** | 深度理解 | 關鍵字匹配 |
| **上下文** | 保留完整 | 可能丟失 |
| **處理速度** | 較慢（API） | 極快 |
| **準確度** | 高 | 中等 |

- 使用TF-IDF（Term Frequency-Inverse Document Frequency）
- 支援中文分詞處理（jieba）
- N-gram特徵提取（1-gram和2-gram）
- 最大特徵數：1000

### 相似度計算
- 餘弦相似度（Cosine Similarity）
- 範圍：0-1（1表示完全相似）
- 結合向量相似度和規則匹配

### 智能分析
- 關鍵字匹配：檢查新聞內容中的關鍵詞
- 類別匹配：比對新聞與款次類別
- 信心分數：綜合評估匹配可信度
- 替代選項：提供其他可能的匹配款次

### KNN搜索
- 根據相似度排序
- 回傳top_k個最相似結果
- 可調整K值參數（預設為5）

### Pydantic型別檢查
- 確保輸出資料型別正確
- 驗證信心分數範圍
- 檢查必要欄位完整性

## 向量資料庫技術細節

### OpenAI Embedding + FAISS版本
- **模型**: text-embedding-3-small（1536維向量）
- **向量資料庫**: FAISS（Facebook AI Similarity Search）
- **相似度計算**: 餘弦相似度（內積正規化）
- **索引類型**: IndexFlatIP（精確搜索）
- **儲存機制**: 自動儲存/載入向量資料庫檔案
- **API費用**: 約$0.00002/1K tokens

### TF-IDF版本
- **向量化**: TF-IDF（詞頻-逆文件頻率）
- **特徵數**: 1000維（可調整）
- **N-gram**: 1-gram + 2-gram
- **中文分詞**: jieba
- **相似度**: 餘弦相似度
- **優勢**: 完全免費、快速、本地運算

### 效能對比實測
```
建立向量資料庫:
- TF-IDF: ~1秒
- OpenAI: ~10-30秒（取決於網路）

搜索速度:
- TF-IDF: ~0.001秒
- OpenAI: ~1-3秒（取決於API回應）

準確度:
- TF-IDF: 關鍵字匹配準確
- OpenAI: 語意理解更精確
```

## 實際執行結果

### 系統處理統計
- **原始資料**: 93筆重訊款次記錄
- **整併後**: 43個唯一重訊款次
- **新聞處理**: 12筆新聞中11筆成功分析
- **平均信心分數**: 0.3365
- **最高信心分數**: 0.5280
- **最低信心分數**: 0.2187

### 比對效果範例
```
新聞: "台產董事會通過30％現金減資 每股退還新台幣3元"
符合第11款（信心分數: 0.2187）
原因: 與該款次的向量相似度最高

新聞: "隆銘綠能審委會及董事會通過變更112年度私募普通股案之資金用途計畫"
符合第23款（信心分數: 0.2991）
原因: 新聞內容與「財務」類別相關

新聞: "華新大增資 強化國際布局…同步加碼子公司義大CAS"
符合第6款（信心分數: 0.5280）
原因: 與該款次的向量相似度最高
```

### 最常匹配的款次
- 第6款: 3次匹配
- 第20款: 3次匹配
- 第11款、第22款、第23款等: 各1次匹配

## 進階功能

### OpenAI GPT-4o版本
如需更精確的分析，可使用完整版程式：
```bash
# 設定OpenAI API金鑰
python setup_openai.py

# 執行完整版分析
python enhanced_news_rag.py
```

### 自訂規則分析
可在 `analyze_with_rules` 函式中：
- 調整信心分數計算方式
- 添加新的匹配規則
- 修改關鍵字匹配邏輯

## 注意事項
1. 確保網路連接正常，以便擷取新聞內容
2. 部分新聞網站可能有反爬蟲機制
3. 處理大量新聞時請耐心等待
4. 信心分數>0.3的結果較為可信
5. 關注「類別匹配」的結果準確度較高

## 錯誤處理
- 網路連線失敗：跳過該新聞
- 內容擷取失敗：顯示警告訊息
- 檔案讀取錯誤：顯示詳細錯誤資訊
- Excel欄位名稱問題：自動處理換行符號
- 向量化失敗：提供備用分析方法

## 效能優化建議
1. 調整`top_k`參數來改變候選款次數量（預設為5）
2. 調整`max_features`參數以平衡效能與準確度（預設為1000）
3. 使用更適合的中文停用詞表
4. 考慮使用更進階的向量化方法（如Word2Vec、BERT）
5. 批次處理新聞以提高效率
6. 針對特定領域調整關鍵字匹配規則

## 系統架構
```
輸入資料 → 款次整併 → 向量資料庫 → 新聞擷取 → 向量搜索 → 規則分析 → Pydantic驗證 → 結果輸出
```

## 未來改進方向
1. 整合更多新聞來源
2. 支援即時新聞監控
3. 增加人工標註功能
4. 建立歷史比對記錄
5. 開發Web介面
6. 支援多語言新聞分析
