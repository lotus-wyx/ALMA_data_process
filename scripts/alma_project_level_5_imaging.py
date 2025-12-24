#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ALMA Project Level 5: Imaging with tclean
功能：对每个源目录下合并后的 MS 文件进行 tclean 成像
"""

import os
import sys
import glob
import numpy as np

# ================= Configuration =================
# 当前工作目录
base_dir = os.getcwd()

# 目标源列表文件
TARGET_LIST_FILE = 'target_list.txt'

# 输入目录（包含各个源的目录）
INPUT_ROOT_DIR = 'Each_target_img'

# 输出图像子目录名（在每个源目录下创建）
OUTPUT_SUBDIR = 'cubes'

# 成像参数
imsize_set = 800  # 默认图像大小
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

def calculate_cell(vis):
    """
    根据 MS 文件计算合适的 cell 大小
    基于最大基线和参考频率计算合成波束
    """
    print("  Calculating cell size for {}...".format(os.path.basename(vis)))
    
    # 读取参考频率
    tb.open(vis + os.sep + 'SPECTRAL_WINDOW')
    spw_ref_freq_col = tb.getcol('REF_FREQUENCY')
    tb.close()
    ref_freq_Hz = spw_ref_freq_col[0]
    print("    Reference frequency: {:.4f} GHz".format(ref_freq_Hz / 1e9))
    
    # 读取 UVW 数据计算基线
    tb.open(vis)
    uvw = tb.getcol('UVW')
    tb.close()
    
    # 计算 UV 距离
    uvdist = np.sqrt(np.sum(np.square(uvw[0:2, :]), axis=0))
    maxuvdist = np.max(uvdist)
    print("    Maximum UV distance: {:.2f} m".format(maxuvdist))
    
    # 使用 90% 百分位数（更稳健）
    L90uvdist = np.percentile(uvdist, 90)
    print("    90th percentile UV distance: {:.2f} m".format(L90uvdist))
    
    # 计算合成波束大小（使用 90% 百分位数）
    synbeam = 0.574 * 2.99792458e8 / ref_freq_Hz / L90uvdist / np.pi * 180.0 * 3600.0
    
    # 保留 2 位有效数字
    synbeam_nprec = 2
    synbeam_ndigits = (synbeam_nprec - 1) - int(np.floor(np.log10(synbeam)))
    synbeam = (np.round(synbeam * 10**(synbeam_ndigits))) / 10**(synbeam_ndigits)
    print("    Synthesized beam: {:.3f} arcsec".format(synbeam))
    
    # 使用 5 倍过采样
    oversampling = 5.0
    imcell_arcsec = synbeam / oversampling
    imcell = '{}arcsec'.format(imcell_arcsec)
    print("    Cell size: {}".format(imcell))
    
    return imcell

def tclean_for_msfile(msfile, image_path, field_='', imsize_=imsize_set):
    """
    对 MS 文件执行 tclean 成像
    """
    print("\n" + "-"*60)
    print("Imaging: {}".format(os.path.basename(msfile)))
    print("-"*60)
    
    if not os.path.exists(msfile):
        print("Error: MS file '{}' not found!".format(msfile))
        return False
    
    try:
        imcell = calculate_cell(msfile)
    except Exception as e:
        print("Error calculating cell size: {}".format(str(e)))
        return False
    
    # 步骤 1: 创建 dirty 图像
    dirty_image = image_path + '_dirty'
    print("\n  Step 1: Creating dirty image...")
    
    try:
        tclean(
            vis=msfile,
            imagename=dirty_image,
            field=field_,
            datacolumn='corrected',
            specmode='cube',
            width=1,
            start=1,
            nchan=-1,
            veltype='radio',
            imsize=imsize_,
            cell=imcell,
            deconvolver='hogbom',
            weighting='natural',
            #restoringbeam='common',
            niter=0
        )
    except Exception as e:
        print("Error during dirty imaging: {}".format(str(e)))
        return False
    
    # 步骤 2: 计算 RMS
    dirty_image_file = dirty_image + '.image'
    if not os.path.exists(dirty_image_file):
        print("Error: Dirty image was not created!")
        return False
    
    print("\n  Step 2: Calculating RMS...")
    try:
        result_imstat_dict = imstat(dirty_image_file)
        if len(result_imstat_dict['rms']) == 0:
            print("Error: Failed to determine RMS!")
            return False
        
        threshold_clean = result_imstat_dict['rms'][0]
        print("    RMS: {:.6e}".format(threshold_clean))
        print("    Threshold (2*RMS): {:.6e}".format(2 * threshold_clean))
    except Exception as e:
        print("Error calculating RMS: {}".format(str(e)))
        return False
    
    # 步骤 3: 执行 clean
    clean_image = image_path
    print("\n  Step 3: Running tclean...")
    
    try:
        tclean(
            vis=msfile,
            imagename=clean_image,
            field=field_,
            datacolumn='corrected',
            specmode='cube',
            width=1,
            start=1,
            nchan=-1,
            veltype='radio',
            imsize=imsize_,
            cell=imcell,
            deconvolver='hogbom',
            weighting='natural',
            #restoringbeam='common',
            threshold=2 * threshold_clean,
            niter=300000
        )
        print("\n  Imaging completed!")
        return True
    except Exception as e:
        print("Error during clean: {}".format(str(e)))
        return False

def main():
    """主函数"""
    print("="*60)
    print("ALMA Project Level 5: Imaging with tclean")
    print("="*60)
    print("Working directory: {}".format(base_dir))
    print("")
    
    # 获取目标源列表
    if os.path.exists(TARGET_LIST_FILE):
        print("Reading targets from '{}'...".format(TARGET_LIST_FILE))
        targets = get_target_names(TARGET_LIST_FILE)
    else:
        print("Reading targets from directory...")
        targets = get_targets_from_directory()
    
    if not targets:
        print("Error: No targets found!")
        sys.exit(1)
    
    print("\nTargets to process:")
    for i, target in enumerate(targets, 1):
        print("  [{}] {}".format(i, target))
    print("")
    
    # 处理每个目标源
    input_root = os.path.join(base_dir, INPUT_ROOT_DIR)
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for target in targets:
        print("\n" + "="*60)
        print("Processing: {}".format(target))
        print("="*60)
        
        target_dir = os.path.join(input_root, target)
        
        if not os.path.isdir(target_dir):
            print("Warning: Directory not found. Skipping.")
            skipped_count += 1
            continue
        
        # 查找 MS 文件
        ms_file = os.path.join(target_dir, '{}.ms'.format(target))
        
        if not os.path.exists(ms_file):
            print("Warning: MS file not found. Skipping.")
            skipped_count += 1
            continue
        
        # 创建输出目录
        output_dir = os.path.join(target_dir, OUTPUT_SUBDIR)
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置输出路径
        image_path = os.path.join(output_dir, target)
        
        # 检查输出图像是否已存在
        final_image = image_path + '.image'
        if os.path.exists(final_image):
            print("Output image '{}' already exists. Skipping.".format(os.path.basename(final_image)))
            skipped_count += 1
            continue
        
        # 执行成像
        if tclean_for_msfile(ms_file, image_path, field_='', imsize_=imsize_set):
            success_count += 1
        else:
            failed_count += 1
    
    # 输出总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Total: {}".format(len(targets)))
    print("Success: {}".format(success_count))
    print("Failed: {}".format(failed_count))
    print("Skipped: {}".format(skipped_count))
    print("="*60)
    
    if failed_count > 0:
        sys.exit(1)
    else:
        print("\nAll done!")

if __name__ == '__main__':
    main()
