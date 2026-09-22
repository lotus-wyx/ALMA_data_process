#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ALMA Project Level 4: Concat MS files for each target
功能：合并每个源目录下所有 DataSet 开头的 MS 文件
"""

import os
import sys
import glob
import json
import numpy as np

# ================= Configuration =================
# 当前工作目录
base_dir = os.getcwd()

# 目标源列表文件
TARGET_LIST_FILE = 'target_list.txt'

# 输入目录（包含各个源的目录）
INPUT_ROOT_DIR = 'Each_target_img'

# 输出目录（可选，如果为空则输出到源目录下）
OUTPUT_DIR = ''  # 留空表示输出到每个源的目录下

# 不同频段不要放进同一个 MS。按相邻 MS 的实际频率覆盖范围分组：
# gap 大于该阈值才拆分；同一组内仍然保留原来的 concat 逻辑。
BAND_SPLIT = True
# 可通过环境变量 ALMA_FREQUENCY_GAP_GHZ 覆盖；例如 80 表示 gap > 80 GHz 才拆组。
try:
    FREQUENCY_GAP_GHZ = float(os.environ.get('ALMA_FREQUENCY_GAP_GHZ', '100.0'))
except ValueError:
    FREQUENCY_GAP_GHZ = 100.0
    print("Warning: invalid ALMA_FREQUENCY_GAP_GHZ; using 100.0 GHz")
BAND_PREFIX = 'band'
# =================================================

def get_target_names(filename):
    """读取目标源列表文件"""
    if not os.path.exists(filename):
        print("Error: Target list file '{}' not found.".format(filename))
        return []
    
    targets = []
    seen = set()
    with open(filename, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('#') and name not in seen:
                targets.append(name)
                seen.add(name)
    
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

def get_ms_frequency_span_ghz(ms_file):
    """读取 MS 的实际频率覆盖范围（GHz）。"""
    spw_table = ms_file + os.sep + 'SPECTRAL_WINDOW'
    try:
        tb.open(spw_table)
        channel_frequency = np.asarray(tb.getcol('CHAN_FREQ'), dtype=float)
        if channel_frequency.size == 0:
            raise ValueError('CHAN_FREQ is empty')
        return (float(channel_frequency.min()) / 1e9,
                float(channel_frequency.max()) / 1e9)
    finally:
        try:
            tb.close()
        except Exception:
            pass

def write_group_manifest(target_dir, target, group_records):
    """记录每个动态频率组的范围，供后续抽谱/成 line map 使用。"""
    manifest = os.path.join(target_dir, '{}_ms_groups.json'.format(target))
    with open(manifest, 'w') as handle:
        json.dump(group_records, handle, indent=2, sort_keys=True)
    print("Wrote frequency-group manifest: {}".format(manifest))

def group_ms_files_by_band(ms_files):
    """按相邻 MS 之间的频率 gap 动态分组。"""
    spans = []
    for ms_file in ms_files:
        freq_min, freq_max = get_ms_frequency_span_ghz(ms_file)
        spans.append((freq_min, freq_max, ms_file))

    spans.sort(key=lambda item: item[0])
    groups = []
    for freq_min, freq_max, ms_file in spans:
        if not groups or freq_min - groups[-1]['max_ghz'] > FREQUENCY_GAP_GHZ:
            groups.append({'min_ghz': freq_min, 'max_ghz': freq_max, 'files': []})
        group = groups[-1]
        group['files'].append(ms_file)
        group['max_ghz'] = max(group['max_ghz'], freq_max)

    result = {}
    for index, group in enumerate(groups, start=1):
        band = '{}_{:02d}'.format(BAND_PREFIX, index)
        result[band] = group['files']
        print("  {}: {:.3f}-{:.3f} GHz -> {}".format(
            ', '.join(os.path.basename(path) for path in group['files']),
            group['min_ghz'], group['max_ghz'], band))
    return result

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
    
    def remove_ms_path(path):
        """删除旧 MS；软链接必须 unlink，不能对其调用 rmtree。"""
        import shutil
        if os.path.islink(path) or os.path.isfile(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    # 检查输出文件是否已存在（包括上一次生成的软链接）
    if os.path.lexists(output_ms):
        print("Warning: Output MS file already exists. Removing it...")
        remove_ms_path(output_ms)
    
    # 如果只有一个 MS 文件，建立软链接而不是复制或 concat。
    # CASA/casacore 会正常跟随该链接读取 MS；这样不会产生一份重复数据。
    if len(ms_file_list) == 1:
        source_ms = os.path.abspath(ms_file_list[0])
        print("\nOnly one MS file found. Creating symbolic link instead of concat...")
        try:
            os.symlink(source_ms, output_ms)
            print("Symbolic link created: {} -> {}".format(output_ms, source_ms))
            return True
        except OSError as e:
            print("Error creating symbolic link: {}".format(str(e)))
            return False
    
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
        
        # 查找该目录下所有 DataSet MS 文件；不要把已经生成的 band MS 再次纳入输入
        ms_files = find_ms_files(target_dir)
        
        if not ms_files:
            print("\nWarning: No DataSet MS files found in '{}'. Skipping.".format(target_dir))
            failed_count += 1
            continue
        
        # 确定输出目录
        if OUTPUT_DIR:
            output_dir = os.path.join(base_dir, OUTPUT_DIR)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = target_dir

        if BAND_SPLIT:
            print("\nClassifying input MS files by frequency gap (threshold: {:.1f} GHz)...".format(
                FREQUENCY_GAP_GHZ))
            try:
                grouped_ms_files = group_ms_files_by_band(ms_files)
            except Exception as e:
                print("Error reading MS frequency metadata: {}".format(str(e)))
                failed_count += 1
                continue

            target_ok = True
            group_records = []
            for band, band_ms_files in grouped_ms_files.items():
                output_ms = os.path.join(
                    output_dir, '{}_{}.ms'.format(target, band))
                if not concat_ms_files(
                        '{} [{}]'.format(target, band), band_ms_files, output_ms):
                    target_ok = False
                    continue

                spans = [get_ms_frequency_span_ghz(ms) for ms in band_ms_files]
                group_records.append({
                    'group': band,
                    'min_freq_GHz': min(span[0] for span in spans),
                    'max_freq_GHz': max(span[1] for span in spans),
                    'center_freq_GHz': 0.5 * (
                        min(span[0] for span in spans) + max(span[1] for span in spans)),
                    'ms_file': os.path.basename(output_ms),
                })

            if target_ok:
                write_group_manifest(target_dir, target, group_records)
                success_count += 1
            else:
                failed_count += 1
        else:
            # 兼容旧行为：所有 DataSet MS 合并成一个 <target>.ms
            output_ms = os.path.join(output_dir, '{}.ms'.format(target))
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
