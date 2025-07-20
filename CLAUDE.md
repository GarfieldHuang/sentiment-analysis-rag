# CLAUDE.md

本檔案為 Claude Code (claude.ai/code) 在此程式庫中工作時的指引。

## 專案概述

這是一個基於 RAG（檢索增強生成）技術的新聞分析系統，用於比對新聞內容與重訊款次。系統提供兩種向量化方法：

1. **OpenAI Embedding + FAISS** (`enhanced_news_rag.py`) - 高品質語意理解
2. **TF-IDF** (`simplified_enhanced_rag.py`) - 快速本地運算

## 核心指令

### 環境設定
```bash
# 安裝依賴套件
pip install -r requirements.txt

# 設定 OpenAI API 金鑰（增強版需要）
export OPENAI_API_KEY="your-api-key-here"
# 或執行設定腳本
python setup_openai.py
```

### 執行系統

#### OpenAI 版本（推薦）
```bash
# 主要執行
cd src && python enhanced_news_rag.py

# 強制重建向量資料庫
cd src && python enhanced_news_rag.py --rebuild

# 測試系統
python tests/test_openai_faiss_rag.py
```

#### TF-IDF 版本（快速、免費）
```bash
# 主要執行
cd src && python simplified_enhanced_rag.py

# 測試系統
python tests/test_enhanced_rag.py
```

### 效能測試與分析
```bash
# 比較兩個版本的效能
python utils/compare_rag_performance.py

# 測試向量化方法
python tests/test_vectorization_comparison.py

# 測試內容清理和 token 限制
python tests/test_content_cleaning.py

# 測試款次整併
python tests/test_clause_consolidation.py
```

### 資料管理
```bash
# 重建向量資料庫
python scripts/rebuild_vector_db.py

# 查看結果
python utils/view_enhanced_results.py
python utils/view_results.py

# 除錯 Excel 結構
python utils/debug_excel.py
python utils/check_excel_structure.py
```

## 系統架構

### 核心處理流程
```
輸入資料 → 款次整併 → 向量資料庫 → 新聞擷取 → 向量搜尋 → 規則分析 → Pydantic 驗證 → 結果輸出
```

### 關鍵元件

#### 1. 資料整併
- 將 93 筆原始記錄整併為 43 個唯一重訊款次
- 按款次編號合併重複記錄
- 整合類別、內容和檢查項目

#### 2. 向量資料庫
- **OpenAI 版本**：使用 `text-embedding-3-small` 的 1536 維嵌入向量
- **TF-IDF 版本**：使用 jieba 分詞的 1000 維稀疏向量
- 使用 FAISS 進行高效相似度搜尋（OpenAI）或 scikit-learn KNN（TF-IDF）

#### 3. 新聞處理
- 從 `新聞.csv` 中的網址擷取新聞內容
- 智慧內容清理（移除腳本、廣告、導航）
- OpenAI API 的 token 限制處理
- 使用 jieba 進行中文分詞

#### 4. 比對引擎
- 餘弦相似度計算
- 基於規則的分析與關鍵字匹配
- 信心分數計算
- Pydantic 模型驗證

### 檔案結構
- **主要系統**：`src/enhanced_news_rag.py`、`src/simplified_enhanced_rag.py`
- **測試**：`tests/` 目錄下的各種元件測試檔案
- **工具程式**：`scripts/` 目錄下的設定腳本、`utils/` 目錄下的工具程式
- **資料**：`data/重訊款次整理_20250715彙整提供.xlsx`（93 款次）、`data/新聞.csv`（12 則新聞網址）
- **輸出**：`output/AI增強新聞比對結果.xlsx`

## 開發指引

### 測試
- 修改程式碼後務必執行相關測試檔案
- TF-IDF 版本測試使用 `python tests/test_enhanced_rag.py`
- OpenAI 版本測試使用 `python tests/test_openai_faiss_rag.py`

### 設定參數
- `top_k`：KNN 搜尋結果數量（預設：5）
- `max_tokens`：OpenAI 嵌入向量 token 限制（預設：6000）
- `batch_size`：API 批次大小（預設：20）
- `max_features`：TF-IDF 特徵數量（預設：1000）
- `ngram_range`：TF-IDF 的 N-gram 範圍（預設：(1, 2)）

### 向量資料庫管理
- OpenAI 向量儲存於：`vector_db_faiss.index`、`vector_db_metadata.pkl`
- 使用 `--rebuild` 標記強制重新生成
- 已實作自動儲存/載入機制

### 錯誤處理
- 網路故障：跳過有問題的新聞項目
- 內容擷取失敗：記錄警告訊息
- Excel 欄位問題：自動處理換行字元
- API 故障：實作重試機制

### 程式碼模式
- 使用 Pydantic 模型進行型別驗證（`ClauseMatch`、`NewsAnalysisResult`）
- 使用 jieba 正確處理中文文字
- 遵循新聞擷取的 HTML 清理模式
- 使用批次處理 API 呼叫以管理成本

## Claude AI 助手指引

### 工作原則
- 始終以繁體中文（zh-tw）回應使用者
- 提供精確、簡潔的技術支援
- 確保程式碼品質和專案一致性
- 積極協助解決開發中的技術難題
