#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI + FAISS RAG系統快速設定指南
"""

import os
import subprocess
import sys

def install_requirements():
    """安裝所需套件"""
    print("正在安裝所需套件...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ 套件安裝完成")
        return True
    except subprocess.CalledProcessError:
        print("× 套件安裝失敗")
        return False

def setup_openai_key():
    """設定OpenAI API金鑰"""
    print("\n設定OpenAI API金鑰...")
    
    # 檢查是否已設定
    if os.getenv('OPENAI_API_KEY'):
        print("✓ 已設定OpenAI API金鑰")
        return True
    
    print("請到 https://platform.openai.com/api-keys 取得API金鑰")
    api_key = input("請輸入您的OpenAI API金鑰: ").strip()
    
    if api_key:
        # 設定環境變數（臨時）
        os.environ['OPENAI_API_KEY'] = api_key
        
        # 寫入.env檔案
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(f'OPENAI_API_KEY={api_key}\n')
        
        print("✓ API金鑰已設定並儲存至.env檔案")
        print("注意：重新啟動終端機後請執行 'set OPENAI_API_KEY=your-key' 或使用python-dotenv")
        return True
    else:
        print("× 未輸入API金鑰")
        return False

def test_system():
    """測試系統功能"""
    print("\n正在測試系統...")
    
    try:
        from enhanced_news_rag import EnhancedNewsRAG
        
        # 簡單測試
        rag = EnhancedNewsRAG(top_k=3)
        
        if os.getenv('OPENAI_API_KEY'):
            print("✓ OpenAI API金鑰已設定")
            
            # 測試款次整併
            if rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx"):
                print("✓ 款次整併功能正常")
            else:
                print("× 款次整併失敗")
                return False
            
            print("系統準備就緒！")
            print("\n可以執行的指令:")
            print("  python enhanced_news_rag.py          # 執行OpenAI版本")
            print("  python test_openai_faiss_rag.py      # 測試系統功能")
            print("  python compare_rag_performance.py    # 效能比較")
        else:
            print("× 未設定OpenAI API金鑰")
            print("可以執行TF-IDF版本: python simplified_enhanced_rag.py")
        
        return True
        
    except ImportError as e:
        print(f"× 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"× 測試失敗: {e}")
        return False

def main():
    """主設定程序"""
    print("=== OpenAI + FAISS RAG系統設定 ===\n")
    
    # 1. 安裝套件
    if not install_requirements():
        return
    
    # 2. 設定API金鑰
    setup_openai_key()
    
    # 3. 測試系統
    test_system()
    
    print("\n=== 設定完成 ===")
    print("系統已準備就緒，可以開始使用！")

if __name__ == "__main__":
    main()
