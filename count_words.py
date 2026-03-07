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
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(script_dir):
        for file in files:
            if file.endswith('.md'): # or file.endswith('.py'):
                filepath = os.path.join(root, file)
                total += count_chars_in_file(filepath)
    print(total)

if __name__ == "__main__":
    main()