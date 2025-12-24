#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 7 - Generate UVFITS from Level 6 Results

Phase 1: 从 gaussian_fit_summary.txt 提取线心和FWHM
Phase 2: CASA处理 - uvcontsub, split, concat, exportuvfits
"""

import os
import sys
import csv

# ============================================================================
# Phase 1: Extract Line Information (Pure Python)
# ============================================================================

def extract_line_info_from_summaries(project_dir):
    """从所有源的 summary 文件提取线心和 FWHM"""
    
    csv_file = os.path.join(project_dir, 'target_line_list.csv')
    if not os.path.exists(csv_file):
        print("ERROR: target_line_list.csv not found in {}".format(project_dir))
        return {}
    
    # 读取目标列表
    with open(csv_file) as f:
        targets = [row['name'] for row in csv.DictReader(f)]
    
    print("=" * 70)
    print("Phase 1: Extracting line information from {} sources".format(len(targets)))
    print("=" * 70)
    
    line_info = {}
    
    for name in targets:
        summary_file = os.path.join(
            project_dir, 'Each_target_img', name, 'cubes',
            '{}_gaussian_fit_summary.txt'.format(name)
        )
        
        if not os.path.exists(summary_file):
            print("  {:<20s} - SKIP: Summary file not found".format(name))
            continue
        
        # 读取所有 aperture 的拟合结果，选择 SNR 最高的
        best_snr = 0
        best_result = None
        
        with open(summary_file) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                try:
                    snr = float(parts[6])
                    if snr > best_snr:
                        best_snr = snr
                        center = float(parts[2].split('±')[0])  # GHz
                        sigma = float(parts[3].split('±')[0])   # GHz
                        fwhm = 2.355 * sigma                    # GHz
                        aperture = parts[0]
                        best_result = (center, fwhm, aperture)
                except (ValueError, IndexError):
                    continue
        
        if best_result:
            line_info[name] = {
                'line_cen': best_result[0],  # GHz
                'FWHM': best_result[1],      # GHz
                'aperture': best_result[2],
                'snr': best_snr
            }
            print("  {:<20s} - OK: cen={:.3f} GHz, FWHM={:.3f} GHz, SNR={:.1f}".format(
                name, best_result[0], best_result[1], best_snr))
        else:
            print("  {:<20s} - FAILED: No valid fitting results".format(name))
    
    print("=" * 70)
    print("Phase 1 completed: {}/{} sources extracted".format(len(line_info), len(targets)))
    print("=" * 70)
    
    return line_info


def save_line_info_table(line_info, project_dir):
    """保存线信息为 CSV 文件"""
    output_file = os.path.join(project_dir, 'line_info.csv')
    
    with open(output_file, 'w') as f:
        f.write("name,line_cen_GHz,FWHM_GHz,aperture,snr\n")
        for name, info in line_info.items():
            f.write("{},{:.6f},{:.6f},{},{:.1f}\n".format(
                name, info['line_cen'], info['FWHM'], info['aperture'], info['snr']))
    
    print("\nLine information saved to: {}".format(output_file))
    return output_file


# ============================================================================
# Phase 2: CASA Processing (uvcontsub + exportuvfits)
# ============================================================================

def get_line_free_channels(spw, line_channels, cont_channels, nchan):
    """
    生成连续谱通道的范围字符串
    格式：'spw:chan_l~chan_r;chan_l~chan_r'
    """
    if len(cont_channels) == 0:
        return None
    
    line_start = line_channels[0]
    line_end = line_channels[-1]
    
    # 边缘保护：避开前后各2个通道
    if line_start <= 2 and line_end < nchan - 3:
        # 发射线在左边
        fitspec = '{}:{}~{}'.format(spw, line_end + 1, nchan - 3)
    elif line_end >= nchan - 3 and line_start > 2:
        # 发射线在右边
        fitspec = '{}:2~{}'.format(spw, line_start - 1)
    elif line_start > 2 and line_end < nchan - 3:
        # 发射线在中间，两段连续谱
        fitspec = '{}:2~{};{}~{}'.format(spw, line_start - 1, line_end + 1, nchan - 3)
    else:
        # 发射线占据了几乎整个频段
        fitspec = None
    
    return fitspec


def process_to_uvfits_casa(project_dir, line_factor=0.7, cont_factor=1.5):
    """
    CASA 处理流程（需要在 CASA 环境中运行）
    
    Parameters:
    -----------
    project_dir : str
        项目目录
    line_factor : float
        发射线区域：line_cen ± (factor * FWHM)
    cont_factor : float
        连续谱拟合区域：line_cen ± (factor * FWHM) 外
    """
    import shutil
    import numpy as np
    
    # 读取 line_info.csv
    csv_file = os.path.join(project_dir, 'line_info.csv')
    if not os.path.exists(csv_file):
        print("ERROR: line_info.csv not found. Please run Phase 1 first.")
        return
    
    line_info = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            line_info[row['name']] = (float(row['line_cen_GHz']), float(row['FWHM_GHz']))
    
    print("=" * 70)
    print("Phase 2: CASA processing for {} sources".format(len(line_info)))
    print("Line region: line_cen ± {:.1f} * FWHM".format(line_factor))
    print("Cont region: outside line_cen ± {:.1f} * FWHM".format(cont_factor))
    print("=" * 70)
    
    success_count = 0
    
    for name, (line_cen, fwhm) in line_info.items():
        print("\n" + "-" * 70)
        print("Processing: {}".format(name))
        print("-" * 70)
        
        # 查找主 MS 文件
        ms_dir = os.path.join(project_dir, 'Each_target_img', name)
        if not os.path.exists(ms_dir):
            print("  ERROR: Directory not found")
            continue
        
        ms_file = os.path.join(ms_dir, '{}.ms'.format(name))
        if not os.path.exists(ms_file):
            print("  ERROR: Main MS file not found: {}".format(ms_file))
            continue
        
        print("  Processing main MS: {}.ms".format(name))
        
        # 定义频率范围
        line_min = line_cen - line_factor * fwhm
        line_max = line_cen + line_factor * fwhm
        cont_min = line_cen - cont_factor * fwhm
        cont_max = line_cen + cont_factor * fwhm
        
        print("  Line region: {:.3f} - {:.3f} GHz".format(line_min, line_max))
        
        contsub_files = []

        listfile = os.path.join(ms_dir, '{}.listobs'.format(name))
        if os.path.exists(listfile)==False:
            listobs(ms_file,listfile=listfile)
        with open(listfile,'r') as listobs_file:
            all_lines = listobs_file.readlines()
        for index,item in enumerate(all_lines):
            if item.startswith('Spectral Windows:'):
                spw_index = index    
            if item.startswith('Sources:'):
                src_index = index       
                break
        spw_lines = all_lines[spw_index+2:src_index]
        spwid = []

        for line in spw_lines:
            parts = line.split()
            spwid.append(eval(parts[0]))
        ms.open(ms_file)
        for spw in spwid:
            # 获取频率数组（LSRK frame）
            freq = ms.cvelfreqs(spwids=spw, mode='channel', width=0, outframe='LSRK') / 1e9  # GHz
            nchan = len(freq)
            
            # 判断是否包含发射线
            if freq.min() > line_max or freq.max() < line_min:
                print("  SPW{} does not contain line, skipped".format(spw))
                continue
            
            # 找到发射线和连续谱通道
            line_channels = np.where((freq >= line_min) & (freq <= line_max))[0]
            cont_channels = np.where((freq < cont_min) | (freq > cont_max))[0]
            
            if len(line_channels) == 0:
                print("  SPW{} has no line channels, skipped".format(spw))
                continue
            
            if len(cont_channels) < 5:
                print("  SPW{} has too few continuum channels, skipped".format(spw))
                continue
            
            # 生成 fitspec（使用范围格式）
            fitspec = get_line_free_channels(spw, line_channels, cont_channels, nchan)
            
            if fitspec is None:
                print("  SPW{} cannot generate valid fitspec, skipped".format(spw))
                continue
            
            print("  SPW{}: {} line channels, fitspec='{}'".format(spw, len(line_channels), fitspec))
            
            # uvcontsub 输出目录
            contsub_dir = os.path.join(ms_dir, 'contsub')
            if not os.path.exists(contsub_dir):
                os.makedirs(contsub_dir)
            
            output_ms = os.path.join(contsub_dir, '{}_spw{}_contsub'.format(name, spw))
            if os.path.exists(output_ms):
                shutil.rmtree(output_ms)
            
            print("  Running uvcontsub...")
            uvcontsub(vis=ms_file, outputvis=output_ms, spw=str(spw), 
                        fitspec=fitspec, fitorder=0)
            
            # split 并平均发射线通道
            output_avg = output_ms + '_avg'
            if os.path.exists(output_avg):
                shutil.rmtree(output_avg)
            
            line_start = line_channels[0]
            line_end = line_channels[-1]
            width = len(line_channels)
            
            print("  Averaging {} channels...".format(width))
            split(vis=output_ms, outputvis=output_avg,
                    spw='{}:{}~{}'.format(spw, line_start, line_end),
                    width=width, datacolumn='data')
            
            contsub_files.append(output_avg)
        ms.close()

        
        if not contsub_files:
            print("  ERROR: No contsub files generated")
            continue
        
        # concat 合并多个 SPW
        concat_ms = os.path.join(ms_dir, 'contsub', '{}_line.ms'.format(name))
        if os.path.exists(concat_ms):
            shutil.rmtree(concat_ms)
        
        if len(contsub_files) == 1:
            # 只有一个 SPW，直接使用
            concat_ms = contsub_files[0]
            final_ms = concat_ms
        else:
            # 多个 SPW，需要 concat）
            print("  Concatenating {} SPW files...".format(len(contsub_files)))
            concat(vis=contsub_files, concatvis=concat_ms, freqtol='1GHz')
            
            # 修正 CHAN_WIDTH（处理多个SPW通道宽度不一致的问题）
            print("  Checking and fixing CHAN_WIDTH...")
            tb.open(concat_ms + '/SPECTRAL_WINDOW', nomodify=False)
            chan_width = tb.getcol('CHAN_WIDTH')[0]
            
            if len(chan_width) > 1:
                print("  Found {} SPWs with different channel widths, fixing...".format(len(chan_width)))
                # 找到最小通道宽度作为目标宽度
                tar_chan_width_ind = np.argmin(np.abs(chan_width))
                tar_chan_width = chan_width[tar_chan_width_ind]
                
                # 修正所有 SPW 的通道宽度
                for i in range(len(chan_width)):
                    if i != tar_chan_width_ind:
                        tb.putcell('CHAN_WIDTH', i, [tar_chan_width])
                        tb.putcell('EFFECTIVE_BW', i, [abs(tar_chan_width)])
                        tb.putcell('RESOLUTION', i, [abs(tar_chan_width)])
                        tb.putcell('TOTAL_BANDWIDTH', i, abs(tar_chan_width))
                    
                    # 对每个 SPW 单独 split
                    spw_ms = os.path.join(ms_dir, 'contsub', '{}_line_spw{}.ms'.format(name, i))
                    if os.path.exists(spw_ms):
                        shutil.rmtree(spw_ms)
                    split(concat_ms, outputvis=spw_ms, spw=str(i), datacolumn='data')
                
                tb.close()
                
                # 再次 concat 所有 SPW
                final_ms = os.path.join(ms_dir, 'contsub', '{}_line_1chan.ms'.format(name))
                if os.path.exists(final_ms):
                    shutil.rmtree(final_ms)
                
                spw_ms_list = [os.path.join(ms_dir, 'contsub', '{}_line_spw{}.ms'.format(name, i)) 
                              for i in range(len(chan_width))]
                concat(vis=spw_ms_list, concatvis=final_ms, freqtol='1GHz')
                
                print("  CHAN_WIDTH fixed and re-concatenated")
            else:
                tb.close()
                final_ms = concat_ms
        
        # 修正 RECEPTOR_ANGLE（GILDAS 兼容性）
        try:
            tb.open(final_ms + '/FEED', nomodify=False)
            receptor_angles = tb.getcol('RECEPTOR_ANGLE')
            need_fix = False
            
            for i in range(len(receptor_angles)):
                if len(set(receptor_angles[i][:])) != 1:
                    receptor_angles[i][:] = 0.0
                    need_fix = True
            
            if need_fix:
                print("  Fixing RECEPTOR_ANGLE for GILDAS compatibility...")
                tb.putcol('RECEPTOR_ANGLE', receptor_angles)
            
            tb.close()
        except Exception as e:
            print("  WARNING: Could not fix RECEPTOR_ANGLE: {}".format(str(e)))
        
        # exportuvfits
        output_dir = os.path.join(project_dir, 'size_gildas',name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        uvfits_file = os.path.join(output_dir, 'line.uvfits')
        if os.path.exists(uvfits_file):
            os.remove(uvfits_file)
        
        print("  Exporting to UVFITS...")
        exportuvfits(vis=final_ms, fitsfile=uvfits_file, datacolumn='data', overwrite=True,multisource=False)
        
        print("  SUCCESS: {}".format(uvfits_file))
        success_count += 1
    
    print("\n" + "=" * 70)
    print("Phase 2 completed: {}/{} sources processed".format(success_count, len(line_info)))
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python alma_project_level_7_generate_uvfits.py <project_dir> [--phase 1|2]")
        print("")
        print("Example:")
        print("  Phase 1 only: python alma_project_level_7_generate_uvfits.py /alma/xxx --phase 1")
        print("  Phase 2 only: casa -c alma_project_level_7_generate_uvfits.py /alma/xxx --phase 2")
        print("  Both phases: Run Phase 1 first, then Phase 2 in CASA")
        sys.exit(1)
    
    # 解析参数
    project_dir = sys.argv[1]
    phase = None
    
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--phase' and i+1 < len(sys.argv):
            phase = sys.argv[i+1]
    
    # Phase 1: Extract line information (Pure Python)
    if phase is None or phase == '1':
        line_info = extract_line_info_from_summaries(project_dir)
        if line_info:
            save_line_info_table(line_info, project_dir)
        else:
            print("\nERROR: No line information extracted")
    
    # Phase 2: CASA processing, export uvfits
    if phase == '2':
        try:
            # 检查是否在 CASA 环境中
            ms.open
            process_to_uvfits_casa(project_dir)
        except NameError:
            print("\nERROR: Phase 2 must be run in CASA environment!")
            print("Usage: casa --nologger --nogui -c alma_project_level_7_generate_uvfits.py <project_dir> --phase 2")
