#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Phase 3 & 4: Generate Line Map and GILDAS Scripts

Phase 3: 使用 CASA 的 imcontsub 和 immoments 生成 line map
Phase 4: 生成 GILDAS uv_fit 脚本（可选）

需要在 CASA 环境中运行
"""

import os
import sys
import csv
import numpy as np


def load_target_list(csv_file):
    """从CSV文件读取目标源列表"""
    targets = []
    if not os.path.exists(csv_file):
        print("Error: CSV file '{}' not found.".format(csv_file))
        return targets
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        targets = list(reader)
    
    print("Loaded {} targets from {}".format(len(targets), csv_file))
    return targets


def read_gaussian_fit_result(summary_file):
    """从 gaussian_fit_summary.txt 读取最佳拟合结果（SNR最高的）"""
    if not os.path.exists(summary_file):
        print("  Warning: Summary file not found: {}".format(summary_file))
        return None, None
    
    best_snr = 0
    best_center = None
    best_sigma = None
    
    with open(summary_file, 'r') as f:
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
                    center_str = parts[2].split('±')[0]
                    sigma_str = parts[3].split('±')[0]
                    best_center = float(center_str)
                    best_sigma = float(sigma_str)
            except (ValueError, IndexError) as e:
                continue
    
    if best_center is not None:
        print("  Best fit: center={:.4f} GHz, sigma={:.4f} GHz, SNR={:.1f}".format(
            best_center, best_sigma, best_snr))
    
    return best_center, best_sigma


def freq_to_channel(image_file, *freq_ghz_list):
    """将多个频率（GHz）一次性转换为通道号"""
    ia.open(image_file)
    cs = ia.coordsys()
    
    freq_axis = cs.findaxisbyname('frequency')
    shape = ia.shape()
    nchan = shape[freq_axis]
    
    ref_pix = cs.referencepixel()['numeric'][freq_axis]
    ref_val = cs.referencevalue()['numeric'][freq_axis]
    increment = cs.increment()['numeric'][freq_axis]
    
    ia.close()
    
    channels = []
    for freq_ghz in freq_ghz_list:
        freq_hz = freq_ghz * 1e9
        channel = ref_pix + (freq_hz - ref_val) / increment
        channels.append(int(round(channel)))
    
    return channels, nchan


def record_imfit_result(line_map_im, cube_dir):
    """运行 imfit 并返回高斯参数"""
    Gaussian_par = [0, 0, 0.0001, 0.5, 0.4, 0, 0.5]  # 默认值

    ia.open(line_map_im)
    cs = ia.coordsys()
    shape = ia.shape()
    nx, ny = shape[0], shape[1]
    box_str = '{},{},{},{}'.format(int(nx * 0.35), int(nx * 0.35), int(ny * 0.65), int(ny * 0.65))
    # 获取图像中心像素
    center_pix = [nx/2, ny/2]
    
    # 转换为世界坐标
    world = cs.toworld(center_pix, 'n')['numeric']
    
    # 获取 RA/Dec 轴索引
    ra_axis = cs.findaxisbyname('right ascension')
    dec_axis = cs.findaxisbyname('declination')
    
    # 提取 RA/Dec（单位：弧度 -> 角秒）
    ra = world[ra_axis] * 180 / np.pi * 3600   # arcsec
    dec = world[dec_axis] * 180 / np.pi * 3600  # arcsec
    
    ia.close()
    print("  Image center: RA={:.4f} arcsec, Dec={:.4f} arcsec".format(ra, dec))

    try:
        imfit_results = imfit(
            imagename=line_map_im,
            box=box_str,
            residual=cube_dir + '/imfit.residual',
            logfile=cube_dir + '/imfit.log',
            overwrite=True
        )
        result = None
        if imfit_results['converged'][0]:
            if imfit_results['results']['component0']['ispoint'] == False:
                print('  imfit for line map succeeded!')
                result = imfit_results['deconvolved']['component0']
        
        if result is None:
            print('  WARNING: imfit cannot converge or resolve, using default parameters')
        else:
            flux = result.get('flux', {}).get('value', [0.0001])[0]
            shape = result.get('shape', {})
            maj_ax = shape.get('majoraxis', {}).get('value', 0.4)
            min_ax = shape.get('minoraxis', {}).get('value', 0.32)
            pa = shape.get('positionangle', {}).get('value', 0)
            
            if 'direction' in shape:
                ra_fit = shape['direction']['m0']['value'] * 180 / np.pi
                if ra_fit < 0 and ra > 0:
                    ra_fit = ra_fit + 360
                xoff = ra_fit * 3600 - ra
                yoff = shape['direction']['m1']['value'] * 180 / np.pi * 3600 - dec
            else:
                xoff, yoff = 0, 0

            Gaussian_par = [xoff, yoff, flux, maj_ax / 2, min_ax / 2, pa, 0.5]
        
    except Exception as e:
        print("  ERROR in imfit: {}".format(str(e)))
        print("  Using default Gaussian parameters")
    
    return Gaussian_par


def generate_gildas_script(name, work_dir, Gaussian_par):
    """Phase 4: 生成 GILDAS uv_fit 脚本"""
    print("\n  [Phase 4] Generating uv_fit script for GILDAS...")
    
    gildas_dir = os.path.join(work_dir, 'size_gildas', name)
    if not os.path.exists(gildas_dir):
        os.makedirs(gildas_dir)
    
    log_dir = os.path.join(work_dir, 'size_gildas', 'test_log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    try:
        filename_line = os.path.join(gildas_dir, 'uv_fit_line_min.map')
        with open(filename_line, 'w') as file:
            file.write("@ fits_to_uvt {}/line\n".format(name))
            # gildas_uvt_path = os.path.join(work_dir, 'size_gildas', name).replace('/', '\\')
            file.write("sic copy line.uvt {}/line.uvt\n".format(name))
            file.write("define real xoff yoff flux maj min pa nu /global\n")
            file.write("define character uvtname*128 /global\n")
            file.write("define character fittab*128 /global\n")
            file.write("define character resname*128 /global\n")
            file.write("let uvtname {}/line\n".format(name))
            file.write("let fittab {}/line_result\n".format(name))
            file.write("let resname {}/line_residual\n".format(name))
            file.write("let xoff {}\n".format(Gaussian_par[0]))
            file.write("let yoff {}\n".format(Gaussian_par[1]))
            file.write("let flux {}\n".format(Gaussian_par[2]))
            file.write("let maj {}\n".format(Gaussian_par[3]))
            file.write("let min {}\n".format(Gaussian_par[4])) 
            file.write("let pa {}\n".format(Gaussian_par[5]))
            file.write("let nu {}\n".format(Gaussian_par[6]))
            file.write("run uv_fit uv_fit.init /nowindow\n")
            # gildas_log_path = os.path.join(work_dir, 'size_gildas', 'test_log').replace('/', '\\')
            file.write("sic copy \\home\\wyx\\.gag\\logs\\uv_fit.gildas test_log/line_result_{}.gildas\n".format(name))
        
        filename_line = os.path.join(gildas_dir, 'uv_fit_line_q.map')
        with open(filename_line, 'w') as file:
            file.write("@ fits_to_uvt {}/line\n".format(name))
            # gildas_uvt_path = os.path.join(work_dir, 'size_gildas', name).replace('/', '\\')
            file.write("sic copy line.uvt {}/line.uvt\n".format(name))
            file.write("define real xoff yoff flux maj q pa nu /global\n")
            file.write("define character uvtname*128 /global\n")
            file.write("define character fittab*128 /global\n")
            file.write("define character resname*128 /global\n")
            file.write("let uvtname {}/line\n".format(name))
            file.write("let fittab {}/line_result\n".format(name))
            file.write("let resname {}/line_residual\n".format(name))
            file.write("let xoff {}\n".format(Gaussian_par[0]))
            file.write("let yoff {}\n".format(Gaussian_par[1]))
            file.write("let flux {}\n".format(Gaussian_par[2]))
            file.write("let maj {}\n".format(Gaussian_par[3]))
            file.write("let q {}\n".format(Gaussian_par[4]/Gaussian_par[3])) ## min is changed to ratio in new version gildas
            file.write("let pa {}\n".format(Gaussian_par[5]))
            file.write("let nu {}\n".format(Gaussian_par[6]))
            file.write("run uv_fit uv_fit.init /nowindow\n")
            # gildas_log_path = os.path.join(work_dir, 'size_gildas', 'test_log').replace('/', '\\')
            file.write("sic copy \\home\\wyx\\.gag\\logs\\uv_fit.gildas test_log/line_result_{}.gildas\n".format(name))
        
        print("  Created: {}".format(filename_line))
        return True
    
    except Exception as e:
        print("  ERROR generating GILDAS script: {}".format(str(e)))
        return False


def generate_linemap(name, work_dir, line_factor=0.7, cont_factor=1.5):
    """Phase 3: 生成 line map"""
    print("\n" + "=" * 60)
    print("[Phase 3] Processing: {}".format(name))
    print("=" * 60)
    
    cube_dir = os.path.join(work_dir, 'Each_target_img', name, 'cubes')
    clean_cube = os.path.join(cube_dir, '{}.image'.format(name))
    summary_file = os.path.join(cube_dir, '{}_gaussian_fit_summary.txt'.format(name))
    
    if not os.path.exists(clean_cube):
        print("  ERROR: Image cube not found: {}".format(clean_cube))
        return False
    
    line_cen, sigma = read_gaussian_fit_result(summary_file)
    
    if line_cen is None or sigma is None:
        print("  ERROR: Could not read fit results from summary file")
        return False
    
    fwhm = 2.355 * sigma
    print("  FWHM = {:.4f} GHz".format(fwhm))
    
    line_freq_min = line_cen - line_factor * fwhm
    line_freq_max = line_cen + line_factor * fwhm
    cont_freq_min = line_cen - cont_factor * fwhm
    cont_freq_max = line_cen + cont_factor * fwhm
    
    print("  Line region: {:.4f} - {:.4f} GHz (±{:.1f} FWHM)".format(
        line_freq_min, line_freq_max, line_factor))
    print("  Cont exclusion: {:.4f} - {:.4f} GHz (±{:.1f} FWHM)".format(
        cont_freq_min, cont_freq_max, cont_factor))
    
    channels, nchan = freq_to_channel(clean_cube, 
                                       line_freq_min, line_freq_max,
                                       cont_freq_min, cont_freq_max)
    line_chan_min, line_chan_max, cont_chan_min, cont_chan_max = channels
    
    line_chan_min = max(0, min(line_chan_min, line_chan_max))
    line_chan_max = min(nchan - 1, max(line_chan_min, line_chan_max))
    cont_chan_min = max(0, min(cont_chan_min, cont_chan_max))
    cont_chan_max = min(nchan - 1, max(cont_chan_min, cont_chan_max))
    
    print("  Total channels: {}".format(nchan))
    print("  Line channels: {} - {}".format(line_chan_min, line_chan_max))
    print("  Cont exclusion channels: {} - {}".format(cont_chan_min, cont_chan_max))
    
    cont_chans = []
    if cont_chan_min > 2:
        cont_chans.append("2~{}".format(cont_chan_min - 1))
    if cont_chan_max < nchan - 3:
        cont_chans.append("{}~{}".format(cont_chan_max + 1, nchan - 3))
    
    if not cont_chans:
        print("  ERROR: No valid continuum channels available")
        return False
    
    cont_sub_ch = ";".join(cont_chans)
    line_map_chan = "{}~{}".format(line_chan_min, line_chan_max)
    
    print("  Continuum fitting channels: {}".format(cont_sub_ch))
    print("  Line map channels: {}".format(line_map_chan))
    
    clean_line_im = os.path.join(cube_dir, '{}_line.image'.format(name))
    line_map_im = os.path.join(cube_dir, '{}.line_map'.format(name))
    
    import shutil
    for f in [clean_line_im, line_map_im]:
        if os.path.exists(f):
            shutil.rmtree(f)
            print("  Removed existing: {}".format(f))
    
    # imcontsub
    print("\n  Running imcontsub...")
    try:
        imcontsub(
            imagename=clean_cube,
            linefile=clean_line_im,
            fitorder=0,
            chans=cont_sub_ch
        )
        print("  Created: {}".format(clean_line_im))
    except Exception as e:
        print("  ERROR in imcontsub: {}".format(str(e)))
        return False
    
    # immoments
    print("\n  Running immoments...")
    try:
        immoments(
            axis='spec',
            imagename=clean_line_im,
            moments=[0],
            chans=line_map_chan,
            outfile=line_map_im
        )
        print("  Created: {}".format(line_map_im))
    except Exception as e:
        print("  ERROR in immoments: {}".format(str(e)))
        return False
    
    # imfit
    print("\n  Running imfit...")
    ia.open(line_map_im)
    lm_shape = ia.shape()
    ia.close()
    nx, ny = lm_shape[0], lm_shape[1]
    box_str = '{},{},{},{}'.format(int(nx * 0.35), int(nx * 0.35), int(ny * 0.65), int(ny * 0.65))
    imfit_results = imfit(
            imagename=line_map_im,
            box=box_str,
            residual=cube_dir + '/imfit.residual',
            logfile=cube_dir + '/imfit.log',
            overwrite=True
        )
    # Gaussian_par = run_imfit(line_map_im, cube_dir)
    
    # # 保存参数到文件供 Phase 4 使用
    # param_file = os.path.join(cube_dir, '{}_imfit_params.txt'.format(name))
    # with open(param_file, 'w') as f:
    #     f.write(' '.join([str(p) for p in Gaussian_par]))
    # print("  Saved imfit params: {}".format(param_file))
    
    print("\n  [Phase 3] SUCCESS: {} line map generated".format(name))
    return True


def generate_gildas_only(name, work_dir):
    """Phase 4: 只生成 GILDAS 脚本（不重复运行 Phase 3）"""
    print("\n" + "=" * 60)
    print("[Phase 4] Processing: {}".format(name))
    print("=" * 60)
    
    cube_dir = os.path.join(work_dir, 'Each_target_img', name, 'cubes')
    line_map_im = os.path.join(cube_dir, '{}.line_map'.format(name))
    
    if os.path.exists(line_map_im):
        # 如果参数文件不存在但 line_map 存在，重新运行 imfit
        print("  Running imfit to get parameters...")
        Gaussian_par = record_imfit_result(line_map_im, cube_dir)
    else:
        print("  ERROR: line_map not found: {}".format(line_map_im))
        print("  Please run Phase 3 first (--linemap-only)")
        return False
    
    return generate_gildas_script(name, work_dir, Gaussian_par)


def main():
    """主函数"""
    work_dir = os.getcwd()
    csv_file = os.path.join(work_dir, 'target_line_list.csv')
    
    target_name = None
    phase4_only = False
    
    for arg in sys.argv[1:]:
        if arg == '--phase4':
            phase4_only = True
        elif not arg.startswith('-') and not arg.endswith('.py'):
            target_name = arg
    
    print("=" * 70)
    if phase4_only:
        print("Phase 4: Generate GILDAS Scripts")
    else:
        print("Phase 3: Generate Line Maps")
    print("=" * 70)
    print("Working directory: {}".format(work_dir))
    
    if target_name:
        print("Target: {}".format(target_name))
    else:
        print("Mode: Process all targets")
    print("=" * 70)
    
    targets = load_target_list(csv_file)
    
    if len(targets) == 0:
        print("No targets found!")
        return
    
    if target_name:
        targets = [t for t in targets if t['name'] == target_name]
        if len(targets) == 0:
            print("Target '{}' not found in CSV file.".format(target_name))
            return
    
    success_count = 0
    for tgt in targets:
        name = tgt['name']
        if phase4_only:
            if generate_gildas_only(name, work_dir):
                success_count += 1
        else:
            if generate_linemap(name, work_dir):
                success_count += 1
    
    print("\n" + "=" * 70)
    print("Completed: {}/{} targets processed successfully".format(
        success_count, len(targets)))
    print("=" * 70)


if __name__ == '__main__' or 'casa' in dir():
    main()
