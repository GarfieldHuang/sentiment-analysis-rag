#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看簡化版AI新聞比對結果
"""

import pandas as pd

def view_enhanced_results():
    """查看增強版分析結果"""
    try:
        df = pd.read_excel('簡化版AI新聞比對結果.xlsx')
        print("=== 簡化版AI新聞比對結果 ===")
        print(f"總共 {len(df)} 筆比對結果")
        print(f"涉及 {df['新聞標題'].nunique()} 則新聞")
        print()
        
        # 按新聞分組顯示
        main_results = df[df['匹配類型'] == '主要匹配']
        
        print("=== 主要匹配結果 ===")
        for _, row in main_results.iterrows():
            print(f"新聞: {row['新聞標題']}")
            print(f"  ✓ 符合第 {row['符合第幾款']} 款")
            print(f"  ✓ 信心分數: {row['信心分數']:.4f}")
            print(f"  ✓ 符合原因: {row['符合原因說明']}")
            print(f"  ✓ 關鍵要素: {row['關鍵匹配要素']}")
            print(f"  ✓ 分析總結: {row['分析總結']}")
            print()
        
        # 統計分析
        print("=== 統計分析 ===")
        clause_counts = main_results['符合第幾款'].value_counts()
        print("最常匹配的款次:")
        for clause, count in clause_counts.head(5).items():
            print(f"  第 {clause} 款: {count} 次")
        
        print(f"\n平均信心分數: {main_results['信心分數'].mean():.4f}")
        print(f"最高信心分數: {main_results['信心分數'].max():.4f}")
        print(f"最低信心分數: {main_results['信心分數'].min():.4f}")
        
        # 顯示欄位
        print(f"\n=== 輸出欄位 ===")
        for col in df.columns:
            print(f"  {col}")
        
    except Exception as e:
        print(f"讀取結果錯誤: {e}")

if __name__ == "__main__":
    view_enhanced_results()
