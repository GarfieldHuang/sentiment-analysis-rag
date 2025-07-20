# 基於RAG技術的新聞輿情分析系統

這是一個基於RAG（檢索增強生成）技術的新聞分析系統，用於比對新聞內容與重訊款次。系統提供兩種向量化方法：OpenAI Embedding + FAISS（高品質語意理解）和 TF-IDF（快速本地運算）。

## 專案結構

```
sentiment-analysis-rag/
├── src/                    # 主要源碼
│   ├── enhanced_news_rag.py       # OpenAI版本（推薦）
│   ├── simplified_enhanced_rag.py # TF-IDF版本（快速、免費）
│   ├── news_matcher.py           # 新聞比對器
│   ├── stable_news_rag.py        # 穩定版本
│   ├── simple_news_rag.py        # 簡單版本
│   └── minimal_news_rag.py       # 最小版本
├── scripts/                # 工具腳本
│   ├── setup_openai.py           # OpenAI API設定
│   ├── setup_openai_faiss.py     # FAISS設定
│   ├── rebuild_vector_db.py      # 重建向量資料庫
│   └── guide.py                  # 使用指南
├── tests/                  # 測試檔案
│   ├── test_enhanced_rag.py      # TF-IDF版本測試
│   ├── test_openai_faiss_rag.py  # OpenAI版本測試
│   ├── test_clause_consolidation.py # 款次整併測試
│   ├── test_content_cleaning.py  # 內容清理測試
│   └── test_vectorization_comparison.py # 向量化比較測試
├── utils/                  # 工具程式
│   ├── view_enhanced_results.py  # 查看增強結果
│   ├── view_results.py           # 查看結果
│   ├── debug_excel.py            # 除錯Excel
│   ├── check_excel_structure.py  # 檢查Excel結構
│   ├── check_structure.py        # 檢查結構
│   └── compare_rag_performance.py # 效能比較
├── data/                   # 資料檔案
│   ├── 新聞.csv                   # 新聞網址資料
│   └── 重訊款次整理_20250715彙整提供.xlsx # 重訊款次資料
├── output/                 # 輸出檔案
│   └── AI增強新聞比對結果.xlsx      # 分析結果（自動生成）
├── docs/                   # 文件
│   ├── CLAUDE.md                 # Claude AI 助手指引
│   └── README.md                 # 專案說明
├── requirements.txt        # Python依賴套件
└── .gitignore             # Git忽略檔案
```

## 快速開始

### 1. 環境設定

```bash
# 安裝依賴套件
pip install -r requirements.txt

# 設定 OpenAI API 金鑰（增強版需要）
export OPENAI_API_KEY="your-api-key-here"
# 或執行設定腳本
python scripts/setup_openai.py
```

### 2. 執行系統

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

### 3. 效能測試與分析
```bash
# 比較兩個版本的效能
python utils/compare_rag_performance.py

# 測試向量化方法
python tests/test_vectorization_comparison.py

# 測試內容清理和 token 限制
python tests/test_content_cleaning.py
```

### 4. 查看結果
```bash
# 查看增強版結果
python utils/view_enhanced_results.py

# 查看一般結果
python utils/view_results.py
```

## 系統特色

- **雙重向量化方法**：OpenAI Embedding（高精度）+ TF-IDF（高速度）
- **智慧內容清理**：自動移除HTML、腳本、廣告等雜訊
- **款次自動整併**：將93筆原始記錄整併為43個唯一重訊款次
- **多層次比對**：向量相似度 + 規則分析 + 關鍵字匹配
- **完整測試覆蓋**：包含單元測試和效能測試
- **結果可視化**：Excel格式輸出，便於分析

## 技術架構

### 核心處理流程
```
輸入資料 → 款次整併 → 向量資料庫 → 新聞擷取 → 向量搜尋 → 規則分析 → Pydantic 驗證 → 結果輸出
```

### 關鍵技術
- **向量資料庫**：FAISS（OpenAI版）/ scikit-learn KNN（TF-IDF版）
- **中文處理**：jieba 分詞
- **API管理**：OpenAI API with token限制和批次處理
- **資料驗證**：Pydantic 模型
- **網頁擷取**：requests + BeautifulSoup

## 開發指引

詳細的開發指引和系統說明請參考 [docs/CLAUDE.md](docs/CLAUDE.md)。

## 授權

本專案僅供學術研究和內部使用。