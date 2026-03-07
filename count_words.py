#!/usr/bin/env python3
import os

def count_chars_in_file(filepath):
    """返回文件的字符数， 若读取失败则返回 0"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return len(f.read())
    except Exception as e:
        print(f"警告：无法读取文件 {filepath}，已跳过。错误信息：{e}")
        return 0

def main():
    total = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义要排除的目录名（仅目录名，不是完整路径）
    exclude_dir_names = {'data', 'myPic', '.idea', '.git'}
    # 定义要统计的文件后缀（元组格式）
    target_extensions = ('.py', '.md')  # 用元组存储多个后缀
    
    for root, dirs, files in os.walk(script_dir):
        # 核心：原地修改dirs列表，移除所有排除的目录名
        # os.walk 会直接跳过这些目录，不会遍历其下的任何内容
        dirs[:] = [d for d in dirs if d not in exclude_dir_names]
        
        for file in files:
            # 正确用法：传入元组，匹配多个后缀
            if file.endswith(target_extensions):
                filepath = os.path.join(root, file)
                total += count_chars_in_file(filepath)
    
    print(f"排除指定目录及其所有子目录后的总字符数：{total}")

if __name__ == "__main__":
    main()