#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ALMA Project Level 4: Concat MS files for each target
功能：合并每个源目录下所有 DataSet 开头的 MS 文件
"""

import os
import sys
import glob

# ================= Configuration =================
# 当前工作目录
base_dir = os.getcwd()

# 目标源列表文件
TARGET_LIST_FILE = 'target_list.txt'

# 输入目录（包含各个源的目录）
INPUT_ROOT_DIR = 'Each_target_img'

# 输出目录（可选，如果为空则输出到源目录下）
OUTPUT_DIR = ''  # 留空表示输出到每个源的目录下
# =================================================

def get_target_names(filename):
    """读取目标源列表文件"""
    if not os.path.exists(filename):
        print("Error: Target list file '{}' not found.".format(filename))
        return []
    
    targets = []
    with open(filename, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('#'):
                targets.append(name)
    
    print("Found {} targets in target list.".format(len(targets)))
    return targets

def get_targets_from_directory():
    """从 Each_target_img 目录获取所有源的名称"""
    input_dir = os.path.join(base_dir, INPUT_ROOT_DIR)
    
    if not os.path.exists(input_dir):
        print("Error: Input directory '{}' not found.".format(input_dir))
        return []
    
    # 获取所有子目录作为源名称
    targets = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            targets.append(item)
    
    targets.sort()
    print("Found {} target directories in '{}'.".format(len(targets), INPUT_ROOT_DIR))
    return targets

def find_ms_files(target_dir):
    """在目标源目录下查找所有 DataSet 开头的 MS 文件"""
    ms_files = []
    
    # 查找所有 DataSet_*.ms 文件
    pattern = os.path.join(target_dir, 'DataSet_*.ms')
    ms_files = glob.glob(pattern)
    
    # 排序以保证顺序一致
    ms_files.sort()
    
    return ms_files

def concat_ms_files(target_name, ms_file_list, output_ms):
    """使用 CASA concat 合并 MS 文件"""
    print("\n" + "="*60)
    print("Processing target: {}".format(target_name))
    print("="*60)
    print("Number of MS files to concat: {}".format(len(ms_file_list)))
    
    if len(ms_file_list) == 0:
        print("Warning: No MS files found for target '{}'.".format(target_name))
        return False
    
    # 显示所有输入文件
    print("\nInput MS files:")
    for i, ms in enumerate(ms_file_list, 1):
        print("  [{}] {}".format(i, os.path.basename(ms)))
    
    print("\nOutput MS file: {}".format(output_ms))
    
    # 检查输出文件是否已存在
    if os.path.exists(output_ms):
        print("Warning: Output MS file already exists. Removing it...")
        import shutil
        shutil.rmtree(output_ms)
    
    # 如果只有一个 MS 文件，复制而不是 concat
    if len(ms_file_list) == 1:
        print("\nOnly one MS file found. Copying instead of concat...")
        import shutil
        shutil.copytree(ms_file_list[0], output_ms)
        print("Copy completed successfully!")
        return True
    
    # 执行 concat
    try:
        print("\nRunning CASA concat...")
        concat(vis=ms_file_list, concatvis=output_ms)
        print("Concat completed successfully!")
        return True
    except Exception as e:
        print("Error during concat: {}".format(str(e)))
        return False

def main():
    """主函数"""
    print("="*60)
    print("ALMA Project Level 4: Concat MS files")
    print("="*60)
    print("Working directory: {}".format(base_dir))
    print("")
    
    # 1. 获取目标源列表
    # 优先使用 target_list.txt，如果不存在则从目录读取
    if os.path.exists(TARGET_LIST_FILE):
        print("Reading target list from '{}'...".format(TARGET_LIST_FILE))
        targets = get_target_names(TARGET_LIST_FILE)
    else:
        print("Target list file not found. Reading from directory...")
        targets = get_targets_from_directory()
    
    if not targets:
        print("Error: No targets found!")
        sys.exit(1)
    
    print("\nTargets to process:")
    for i, target in enumerate(targets, 1):
        print("  [{}] {}".format(i, target))
    print("")
    
    # 2. 处理每个目标源
    input_root = os.path.join(base_dir, INPUT_ROOT_DIR)
    success_count = 0
    failed_count = 0
    
    for target in targets:
        target_dir = os.path.join(input_root, target)
        
        # 检查目标目录是否存在
        if not os.path.isdir(target_dir):
            print("\nWarning: Target directory '{}' not found. Skipping.".format(target))
            failed_count += 1
            continue
        
        # 查找该目录下所有 DataSet MS 文件
        ms_files = find_ms_files(target_dir)
        
        if not ms_files:
            print("\nWarning: No DataSet MS files found in '{}'. Skipping.".format(target_dir))
            failed_count += 1
            continue
        
        # 确定输出文件名和路径
        if OUTPUT_DIR:
            output_dir = os.path.join(base_dir, OUTPUT_DIR)
            os.makedirs(output_dir, exist_ok=True)
            output_ms = os.path.join(output_dir, '{}.ms'.format(target))
        else:
            # 输出到源目录下
            output_ms = os.path.join(target_dir, '{}.ms'.format(target))
        
        # 执行 concat
        if concat_ms_files(target, ms_files, output_ms):
            success_count += 1
        else:
            failed_count += 1
    
    # 3. 输出总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Total targets: {}".format(len(targets)))
    print("Successfully concatenated: {}".format(success_count))
    print("Failed: {}".format(failed_count))
    print("="*60)
    
    if failed_count > 0:
        print("\nSome targets failed. Please check the log above.")
        sys.exit(1)
    else:
        print("\nAll targets processed successfully!")

if __name__ == '__main__':
    main()
