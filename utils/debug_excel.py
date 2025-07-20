#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
調試Excel欄位問題
"""

import pandas as pd

def debug_excel_columns():
    """調試Excel欄位問題"""
    try:
        df = pd.read_excel("../data/重訊款次整理_20250715彙整提供.xlsx")
        print("所有欄位名稱:")
        for i, col in enumerate(df.columns):
            print(f"{i}: '{col}'")
        
        print("\n測試存取各欄位:")
        row = df.iloc[0]
        for col in df.columns:
            try:
                value = row[col]
                print(f"✓ {col}: {value}")
            except Exception as e:
                print(f"✗ {col}: 錯誤 - {e}")
                
    except Exception as e:
        print(f"錯誤: {e}")

if __name__ == "__main__":
    debug_excel_columns()
