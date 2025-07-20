#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查重訊款次資料結構
"""

import pandas as pd

def check_excel_structure():
    """檢查Excel檔案結構"""
    try:
        df = pd.read_excel("重訊款次整理_20250715彙整提供.xlsx")
        print("Excel檔案結構:")
        print(f"總行數: {len(df)}")
        print(f"總列數: {len(df.columns)}")
        print("\n欄位名稱:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        print("\n前5行資料:")
        print(df.head())
        
        print("\n重訊款次欄位的唯一值:")
        if '重訊款次' in df.columns:
            unique_clauses = df['重訊款次'].dropna().unique()
            print(f"唯一款次數: {len(unique_clauses)}")
            for clause in unique_clauses:
                count = len(df[df['重訊款次'] == clause])
                print(f"  {clause}: {count} 筆")
        
        return df
        
    except Exception as e:
        print(f"錯誤: {e}")
        return None

if __name__ == "__main__":
    check_excel_structure()
