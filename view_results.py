#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看比對結果
"""

import pandas as pd

def view_results(file_path='新聞比對結果.xlsx'):
    """查看比對結果"""
    try:
        df = pd.read_excel(file_path)
        print(f"=== 比對結果總覽 ===")
        print(f"總共有 {len(df)} 筆比對結果")
        print(f"涉及 {df['新聞編號'].nunique()} 則新聞")
        print()
        
        # 按新聞編號分組顯示
        for news_id in sorted(df['新聞編號'].unique()):
            news_data = df[df['新聞編號'] == news_id]
            print(f"新聞 {news_id}:")
            print(f"  標題: {news_data.iloc[0]['新聞標題']}")
            print(f"  網址: {news_data.iloc[0]['新聞網址']}")
            print(f"  前3個最相似的重訊款次:")
            
            top_3 = news_data.head(3)
            for _, row in top_3.iterrows():
                print(f"    排名 {row['比對排名']}: 相似度 {row['相似度分數']:.4f}")
                print(f"      款次ID: {row['重訊款次ID']}")
                print(f"      內容: {row['重訊款次內容'][:100]}...")
                print()
            print("-" * 80)
        
        print(f"\n=== 統計資訊 ===")
        print(f"平均相似度: {df['相似度分數'].mean():.4f}")
        print(f"最高相似度: {df['相似度分數'].max():.4f}")
        print(f"最低相似度: {df['相似度分數'].min():.4f}")
        
        # 顯示各欄位
        print(f"\n=== 檔案欄位 ===")
        for col in df.columns:
            print(f"  {col}")
        
    except Exception as e:
        print(f"讀取結果檔案錯誤: {e}")

if __name__ == "__main__":
    view_results()
