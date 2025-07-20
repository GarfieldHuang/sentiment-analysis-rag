#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG新聞比對系統使用指引
"""

import os

def main():
    print("=== RAG新聞比對系統使用指引 ===")
    print()
    
    print("📋 檔案清單:")
    print("  • simplified_enhanced_rag.py - 增強版主程式（推薦）")
    print("  • enhanced_news_rag.py - 完整版（需OpenAI API）")
    print("  • minimal_news_rag.py - 基礎版程式")
    print("  • view_enhanced_results.py - 查看增強版結果")
    print("  • view_results.py - 查看基礎版結果")
    print("  • test_enhanced_rag.py - 測試增強版系統")
    print("  • setup_openai.py - 設定OpenAI API金鑰")
    print("  • requirements.txt - 套件需求")
    print("  • README.md - 詳細說明文件")
    print()
    
    print("🚀 快速開始:")
    print("  1. 安裝套件: pip install -r requirements.txt")
    print("  2. 執行增強版: python simplified_enhanced_rag.py")
    print("  3. 查看結果: python view_enhanced_results.py")
    print("  4. [可選] 設定OpenAI: python setup_openai.py")
    print()
    
    print("📊 系統功能:")
    print("  ✓ 款次整併: 43個唯一重訊款次")
    print("  ✓ 新聞處理: 11則新聞成功分析")
    print("  ✓ 向量搜索: TF-IDF + 餘弦相似度")
    print("  ✓ 規則分析: 關鍵字匹配 + 類別匹配")
    print("  ✓ Pydantic驗證: 型別檢查與資料驗證")
    print("  ✓ 結構化輸出: 符合第幾款、信心分數、原因說明")
    print()
    
    print("⚙️ 參數調整:")
    print("  • top_k: 修改 SimplifiedEnhancedRAG(top_k=5)")
    print("  • max_features: 修改 TfidfVectorizer(max_features=1000)")
    print("  • confidence_threshold: 在 analyze_with_rules 中調整")
    print()
    
    print("📈 比對結果統計:")
    print("  • 總比對結果: 30筆（含替代選項）")
    print("  • 成功分析新聞: 11則")
    print("  • 平均信心分數: 0.3365")
    print("  • 最高信心分數: 0.5280")
    print("  • 最低信心分數: 0.2187")
    print()
    
    print("🎯 比對效果範例:")
    print("  新聞: '台產董事會通過30％現金減資'")
    print("  符合第 11 款（信心分數: 0.2187）")
    print("  原因: 向量相似度最高")
    print()
    print("  新聞: '隆銘綠能審委會及董事會通過變更私募普通股案'")
    print("  符合第 23 款（信心分數: 0.2991）")
    print("  原因: 新聞內容與「財務」類別相關")
    print()
    
    print("📁 輸出檔案:")
    print("  • 簡化版AI新聞比對結果.xlsx - 增強版比對結果")
    print("  • 包含：符合第幾款、信心分數、符合原因說明")
    print("  • 使用Pydantic進行型別檢查和資料驗證")
    print("  • 結構化的關鍵匹配要素和分析總結")
    print()
    
    print("💡 使用建議:")
    print("  1. 信心分數>0.3 的結果較為可信")
    print("  2. 關注「類別匹配」的結果準確度較高")
    print("  3. 可調整top_k值來改變候選款次數量")
    print("  4. 若需要更精確分析，可使用OpenAI版本")
    print()
    
    print("🆕 新增功能:")
    print("  • 款次整併: 將93筆資料整併為43個唯一款次")
    print("  • 規則分析: 結合關鍵字匹配和類別匹配")
    print("  • Pydantic驗證: 確保輸出資料的型別正確性")
    print("  • 結構化輸出: 符合第幾款、原因說明、關鍵要素")
    print("  • 替代選項: 提供其他可能的匹配款次")
    print()
    
    print("🔧 故障排除:")
    print("  • 網路連線問題: 檢查網路連接")
    print("  • 套件錯誤: 重新安裝requirements.txt")
    print("  • 檔案問題: 確保Excel和CSV檔案完整")
    print()
    
    print("✨ 系統優點:")
    print("  • 自動款次整併: 去除重複，提高效率")
    print("  • 智能規則分析: 結合向量搜索和規則匹配")
    print("  • Pydantic型別檢查: 確保輸出資料品質")
    print("  • 結構化輸出: 符合第幾款、原因說明、信心分數")
    print("  • 可擴展架構: 支援OpenAI GPT-4o進階分析")
    print("  • 多層次匹配: 提供主要匹配和替代選項")
    print()
    
    print("=== 系統已準備就緒 ===")

if __name__ == "__main__":
    main()
