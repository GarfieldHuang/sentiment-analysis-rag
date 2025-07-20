#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
設定OpenAI API金鑰
"""

import os

def setup_openai_key():
    """設定OpenAI API金鑰"""
    print("=== OpenAI API金鑰設定 ===")
    print()
    
    # 檢查是否已設定
    if os.getenv("OPENAI_API_KEY"):
        print("✓ OpenAI API金鑰已設定")
        print(f"  金鑰: {os.getenv('OPENAI_API_KEY')[:10]}...")
        return True
    
    print("請選擇設定方式:")
    print("1. 設定環境變數（推薦）")
    print("2. 在程式中臨時設定")
    print("3. 跳過（測試系統功能）")
    
    choice = input("\n請選擇 (1-3): ").strip()
    
    if choice == "1":
        print("\n環境變數設定方式:")
        print("Windows Command Prompt:")
        print("  set OPENAI_API_KEY=your_api_key_here")
        print("\nWindows PowerShell:")
        print("  $env:OPENAI_API_KEY='your_api_key_here'")
        print("\n設定後請重新啟動終端機")
        
    elif choice == "2":
        api_key = input("請輸入您的OpenAI API金鑰: ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            print("✓ API金鑰已臨時設定")
            return True
        else:
            print("✗ 未輸入API金鑰")
            
    elif choice == "3":
        print("將跳過OpenAI功能，僅測試向量搜索")
        return False
        
    else:
        print("無效選擇")
        
    return False

def test_api_key():
    """測試API金鑰是否有效"""
    try:
        import openai
        client = openai.OpenAI()
        
        # 測試API呼叫
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 使用較便宜的模型測試
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✓ API金鑰有效")
        return True
        
    except Exception as e:
        print(f"✗ API金鑰測試失敗: {e}")
        return False

if __name__ == "__main__":
    if setup_openai_key():
        test_api_key()
    
    print("\n設定完成後可以執行:")
    print("  python test_enhanced_rag.py  # 測試系統")
    print("  python enhanced_news_rag.py  # 執行完整分析")
