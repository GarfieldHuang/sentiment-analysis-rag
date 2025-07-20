#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
檢查重訊款次整併是否正確
"""

import pandas as pd
from enhanced_news_rag import EnhancedNewsRAG
import os

def check_clause_consolidation():
    """檢查款次整併結果"""
    print("=== 檢查重訊款次整併 ===\n")
    
    # 1. 直接檢查Excel檔案
    try:
        df = pd.read_excel("重訊款次整理_20250715彙整提供.xlsx")
        print(f"Excel檔案載入成功，共 {len(df)} 行")
        
        # 檢查重訊款次欄位
        if '重訊款次' in df.columns:
            clause_col = df['重訊款次']
            print(f"找到'重訊款次'欄位")
        else:
            print("錯誤：找不到'重訊款次'欄位")
            print(f"可用欄位: {list(df.columns)}")
            return
        
        # 統計原始資料
        valid_clauses = clause_col.dropna()
        valid_clauses = valid_clauses[valid_clauses != 'X']
        
        print(f"有效款次記錄: {len(valid_clauses)} 筆")
        print(f"空值記錄: {clause_col.isna().sum()} 筆")
        print(f"'X'標記記錄: {(clause_col == 'X').sum()} 筆")
        
        # 統計唯一款次
        unique_clauses = valid_clauses.unique()
        print(f"唯一款次數量: {len(unique_clauses)}")
        
        # 顯示重複款次
        value_counts = valid_clauses.value_counts()
        duplicates = value_counts[value_counts > 1]
        if len(duplicates) > 0:
            print(f"\n需要合併的重複款次:")
            for clause, count in duplicates.items():
                print(f"  款次 {clause}: {count} 筆")
        
    except Exception as e:
        print(f"檢查Excel檔案錯誤: {e}")
        return
    
    print("\n" + "="*50)
    
    # 2. 使用RAG系統檢查整併結果
    print("\n使用RAG系統檢查整併結果...")
    
    # 檢查API金鑰
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        rag = EnhancedNewsRAG(top_k=3, openai_api_key=api_key)
    else:
        print("未設定OpenAI API金鑰，使用基礎模式")
        rag = EnhancedNewsRAG(top_k=3)
    
    # 先清空
    rag.clear_vector_database()
    
    # 執行整併
    consolidated = rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
    
    if consolidated:
        print(f"\n整併成功，共 {len(consolidated)} 個唯一款次")
        
        # 檢查整併前後的數量
        reduction = len(valid_clauses) - len(consolidated)
        print(f"資料壓縮: {len(valid_clauses)} -> {len(consolidated)} (減少 {reduction} 筆)")
        
        # 顯示款次詳情
        print(f"\n整併後的款次列表:")
        for clause_num, data in sorted(consolidated.items()):
            raw_count = len(data['raw_rows'])
            print(f"  款次 {clause_num}: 合併了 {raw_count} 筆原始資料")
            
            # 顯示類別資訊
            if data['categories']:
                print(f"    類別: {', '.join(data['categories'])}")
            if data['content_types']:
                print(f"    報導內容: {', '.join(data['content_types'])}")
        
        # 測試向量資料庫建立
        print(f"\n測試向量資料庫建立...")
        if api_key:
            if rag.build_vector_database():
                print(f"✓ 向量資料庫建立成功")
                print(f"向量資料庫包含 {len(rag.vector_db)} 個款次")
                
                # 檢查款次編號一致性
                db_clauses = set(item['clause_number'] for item in rag.vector_db)
                consolidated_clauses = set(consolidated.keys())
                
                if db_clauses == consolidated_clauses:
                    print("✓ 向量資料庫款次編號與整併結果一致")
                else:
                    print("× 向量資料庫款次編號不一致")
                    print(f"整併結果: {sorted(consolidated_clauses)}")
                    print(f"向量資料庫: {sorted(db_clauses)}")
            else:
                print("× 向量資料庫建立失敗")
        else:
            print("跳過向量資料庫測試（需要OpenAI API金鑰）")
            
    else:
        print("× 整併失敗")

def test_rebuild_functionality():
    """測試重建功能"""
    print(f"\n" + "="*50)
    print("測試重建功能...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("需要OpenAI API金鑰來測試重建功能")
        return
    
    rag = EnhancedNewsRAG(top_k=3, openai_api_key=api_key)
    
    # 第一次建立
    print("第一次建立...")
    rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
    first_count = len(rag.consolidated_clauses)
    
    # 第二次建立（應該清空重建）
    print("第二次建立（測試清空）...")
    rag.consolidate_clauses("重訊款次整理_20250715彙整提供.xlsx")
    second_count = len(rag.consolidated_clauses)
    
    if first_count == second_count:
        print(f"✓ 重建功能正常，兩次都是 {second_count} 個款次")
    else:
        print(f"× 重建功能異常，第一次 {first_count}，第二次 {second_count}")

if __name__ == "__main__":
    check_clause_consolidation()
    test_rebuild_functionality()
