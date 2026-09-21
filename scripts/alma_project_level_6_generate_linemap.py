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
import json
import numpy as np
import shutil

def infer_band(target, target_name=None, work_dir='.'):
    """从 CSV 或 MS 分组 manifest 推断动态频率组。"""
    band = (target.get('band') or '').strip().lower()
    if band:
        return band

    if target_name:
        manifest = os.path.join(
            work_dir, 'Each_target_img', target_name,
            '{}_ms_groups.json'.format(target_name))
        if os.path.exists(manifest):
            try:
                with open(manifest) as handle:
                    groups = json.load(handle)
                frequency_ghz = float(target.get('line_freq_GHz', ''))
                containing = [group for group in groups
                              if group['min_freq_GHz'] <= frequency_ghz <= group['max_freq_GHz']]
                if containing:
                    return containing[0]['group']
                if groups:
                    return min(groups, key=lambda group: abs(
                        group['center_freq_GHz'] - frequency_ghz))['group']
            except (TypeError, ValueError, IOError, KeyError):
                pass

    # 没有 manifest 且 CSV 未指定 band 时，不按绝对频率强行拆分。
    return None

def group_file_suffix(group):
    return group if group.startswith('band_') else 'band_{}'.format(group)

def resolve_clean_cube(work_dir, base_name, target):
    """优先使用按频段成像的 cube，找不到时回退到旧 cube。"""
    band = infer_band(target, base_name, work_dir)
    cube_dir = os.path.join(work_dir, 'Each_target_img', base_name, 'cubes')
    candidates = []
    if band:
        candidates.append(os.path.join(
            cube_dir, '{}_{}.image'.format(base_name, group_file_suffix(band))))
    candidates.append(os.path.join(cube_dir, '{}.image'.format(base_name)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else candidates[-1]

def build_target_entries(rows):
    """为每个目标行补充 position_id、band 和唯一 uid。"""
    targets = []
    seen_pairs = set()

    for row in rows:
        name = row.get('name', '').strip()
        if not name:
            continue

        position_id = (row.get('position_id') or 'default').strip()
        if not position_id:
            position_id = 'default'

        clean_position_id = ''.join([
            ch if (ch.isalnum() or ch in ['-', '_', '.']) else '_'
            for ch in position_id
        ]).strip('_')
        if not clean_position_id:
            clean_position_id = 'default'

        band = infer_band(row, name) or ''
        pair_key = (name, clean_position_id, band)
        if pair_key in seen_pairs:
            raise ValueError(
                "Duplicate entry found in CSV for name='{}', position_id='{}', band='{}'. "
                "Please keep position_id and band unique within each name.".format(
                    name, clean_position_id, band or 'unspecified')
            )
        seen_pairs.add(pair_key)

        entry = dict(row)
        entry['position_id'] = clean_position_id
        entry['band'] = band
        entry['uid'] = '{}_{}'.format(name, clean_position_id)
        if band:
            entry['uid'] += '_{}'.format(group_file_suffix(band))
        targets.append(entry)

    return targets


def load_target_list(csv_file):
    """从CSV文件读取目标源列表"""
    targets = []
    if not os.path.exists(csv_file):
        print("Error: CSV file '{}' not found.".format(csv_file))
        return targets

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    try:
        targets = build_target_entries(rows)
    except ValueError as e:
        print("Error: {}".format(str(e)))
        return []

    print("Loaded {} targets from {}".format(len(targets), csv_file))
    return targets

def build_imfit_box_from_target_pixel(line_map_im, target, box_size_arcsec=5.0):
    ia.open(line_map_im)
    cs = ia.coordsys()
    shape = ia.shape()
    nx, ny = int(shape[0]), int(shape[1])

    # 目标中心像素（CSV里是0-based）
    cx = int(round(float(target['pixel_x'])))
    cy = int(round(float(target['pixel_y'])))

    # 从header取RA/Dec轴每像素角尺度（通常是弧度）
    inc = cs.increment()['numeric']
    ra_axis = cs.findaxisbyname('right ascension')
    dec_axis = cs.findaxisbyname('declination')

    # 转成 arcsec/pixel
    ra_cell_arcsec = abs(float(inc[ra_axis])) * 206264.806
    dec_cell_arcsec = abs(float(inc[dec_axis])) * 206264.806
    cell_arcsec = max(ra_cell_arcsec, dec_cell_arcsec)  # 保守一点，取较大值

    ia.close()

    half_box_pix = int(np.ceil((box_size_arcsec / 2.0) / cell_arcsec))
    half_box_pix = max(1, half_box_pix)

    x1 = max(0, cx - half_box_pix)
    y1 = max(0, cy - half_box_pix)
    x2 = min(nx - 1, cx + half_box_pix)
    y2 = min(ny - 1, cy + half_box_pix)

    return '{},{},{},{}'.format(x1, y1, x2, y2)

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


def get_valid_cube_channels(image_file):
    """找出具有真实图像数据的 spectral channels。

    tclean 在没有观测覆盖的频率平面中可能写入普通数值 0，而不是将
    CASA image mask 设为 false。这里同时要求该 channel 有有效像素且
    空间 RMS 为有限正数，避免 imcontsub 把空频率平面当成零连续谱。
    """
    ia.open(image_file)
    try:
        cs = ia.coordsys()
        freq_axis = cs.findaxisbyname('frequency')
        shape = ia.shape()
        try:
            cs.done()
        except Exception:
            pass
    finally:
        ia.close()

    nchan = int(shape[freq_axis])

    # tclean 的 sumwt 是判断 visibility 覆盖最直接的产品：没有观测的
    # channel 权重为零。新旧项目的 sumwt 维度可能不同，因此只有在
    # spectral axis 长度与 cube 一致时才采用，否则回退到 image RMS。
    suffix = '.image'
    sumwt_image = (
        image_file[:-len(suffix)] + '.sumwt'
        if image_file.endswith(suffix) else image_file + '.sumwt')
    if os.path.exists(sumwt_image):
        try:
            ia.open(sumwt_image)
            try:
                sumwt_cs = ia.coordsys()
                sumwt_freq_axis = sumwt_cs.findaxisbyname('frequency')
                sumwt_data = np.asarray(ia.getchunk(), dtype=float)
                try:
                    sumwt_cs.done()
                except Exception:
                    pass
            finally:
                ia.close()

            if sumwt_data.shape[sumwt_freq_axis] == nchan:
                weights_by_channel = np.moveaxis(
                    sumwt_data, sumwt_freq_axis, 0).reshape(nchan, -1)
                valid = np.any(
                    np.isfinite(weights_by_channel)
                    & (weights_by_channel > 0), axis=1)
                print("  Observed channels from sumwt: {} / {} "
                      "(excluded {} zero-weight)".format(
                          int(np.sum(valid)), nchan, int(np.sum(~valid))))
                return valid
            print("  Warning: sumwt spectral size does not match cube; "
                  "falling back to image RMS")
        except Exception as e:
            print("  Warning: could not read sumwt ({}); "
                  "falling back to image RMS".format(str(e)))

    collapse_axes = [axis for axis in range(len(shape)) if axis != freq_axis]
    stats = imstat(imagename=image_file, axes=collapse_axes)
    rms = np.asarray(stats.get('rms', []), dtype=float).ravel()
    npts = np.asarray(stats.get('npts', []), dtype=float).ravel()

    if rms.size != nchan or npts.size != nchan:
        raise ValueError(
            "Per-channel imstat returned rms={} and npts={} for {} channels".format(
                rms.size, npts.size, nchan))

    valid = np.isfinite(rms) & (rms > 0) & np.isfinite(npts) & (npts > 0)
    invalid_count = int(np.sum(~valid))
    print("  Observed channels: {} / {} (excluded {} empty/invalid)".format(
        int(np.sum(valid)), nchan, invalid_count))
    return valid


def channels_to_selection(channel_mask):
    """将布尔 channel mask 转换成 CASA 的分段选择字符串。"""
    channel_mask = np.asarray(channel_mask, dtype=bool).ravel()
    selected = np.flatnonzero(channel_mask)
    if selected.size == 0:
        return ''

    ranges = []
    run_start = int(selected[0])
    run_end = run_start
    for channel in selected[1:]:
        channel = int(channel)
        if channel == run_end + 1:
            run_end = channel
        else:
            ranges.append(
                str(run_start) if run_start == run_end
                else '{}~{}'.format(run_start, run_end))
            run_start = channel
            run_end = channel
    ranges.append(
        str(run_start) if run_start == run_end
        else '{}~{}'.format(run_start, run_end))
    return ';'.join(ranges)


def sorted_clipped_channel_pair(channel_a, channel_b, nchan):
    """排序两个 channel 边界并限制到 cube 范围。"""
    channel_min = max(0, min(int(channel_a), int(channel_b)))
    channel_max = min(nchan - 1, max(int(channel_a), int(channel_b)))
    return channel_min, channel_max


def record_imfit_result(line_map_im, cube_dir,target):
    """运行 imfit 并返回高斯参数"""
    Gaussian_par = [0, 0, 0.0001, 0.5, 0.4, 0, 0.5]  # 默认值

    ia.open(line_map_im)
    cs = ia.coordsys()
    shape = ia.shape()
    nx, ny = shape[0], shape[1]
    box_str = build_imfit_box_from_target_pixel(line_map_im, target, box_size_arcsec=5.0)
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

    ## 检查imfit结果文件是否已经存在，如果存在则删除
      # 去掉 .image 后缀
    imfit_log = os.path.join(cube_dir, line_map_im.split('.line_map')[0]+'_imfit.log')
    residual_file = os.path.join(cube_dir, line_map_im.split('.line_map')[0]+'_imfit.residual')

    if os.path.exists(imfit_log):
        os.remove(imfit_log)
        shutil.rmtree(residual_file)
        print("  Removed existing imfit log: {}".format(imfit_log))
        print("  Removed existing imfit residual: {}".format(residual_file))

    try:
        imfit_results = imfit(
            imagename=line_map_im,
            box=box_str,
            residual=residual_file,
            logfile=imfit_log,
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


def generate_gildas_script(base_name, uid, work_dir, Gaussian_par):
    """Phase 4: 生成 GILDAS uv_fit 脚本"""
    print("\n  [Phase 4] Generating uv_fit script for GILDAS...")
    
    gildas_dir = os.path.join(work_dir, 'size_gildas', uid)
    if not os.path.exists(gildas_dir):
        os.makedirs(gildas_dir)
    
    log_dir = os.path.join(work_dir, 'size_gildas', 'test_log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    try:
        filename_line = os.path.join(gildas_dir, 'uv_fit_line_min.map')
        with open(filename_line, 'w') as file:
            file.write("@ fits_to_uvt {}/line\n".format(uid))
            file.write("sic copy line.uvt {}/line.uvt\n".format(uid))
            file.write("define real xoff yoff flux maj min pa nu /global\n")
            file.write("define character uvtname*128 /global\n")
            file.write("define character fittab*128 /global\n")
            file.write("define character resname*128 /global\n")
            file.write("let uvtname {}/line\n".format(uid))
            file.write("let fittab {}/line_result\n".format(uid))
            file.write("let resname {}/line_residual\n".format(uid))
            file.write("let xoff {}\n".format(Gaussian_par[0]))
            file.write("let yoff {}\n".format(Gaussian_par[1]))
            file.write("let flux {}\n".format(Gaussian_par[2]))
            file.write("let maj {}\n".format(Gaussian_par[3]))
            file.write("let min {}\n".format(Gaussian_par[4])) 
            file.write("let pa {}\n".format(Gaussian_par[5]))
            file.write("let nu {}\n".format(Gaussian_par[6]))
            file.write("run uv_fit uv_fit.init /nowindow\n")
            file.write("sic copy \\home\\wyx\\.gag\\logs\\uv_fit.gildas test_log/line_result_{}.gildas\n".format(uid))
        
        filename_line = os.path.join(gildas_dir, 'uv_fit_line_q.map')
        with open(filename_line, 'w') as file:
            file.write("@ fits_to_uvt {}/line\n".format(uid))
            file.write("sic copy line.uvt {}/line.uvt\n".format(uid))
            file.write("define real xoff yoff flux maj q pa nu /global\n")
            file.write("define character uvtname*128 /global\n")
            file.write("define character fittab*128 /global\n")
            file.write("define character resname*128 /global\n")
            file.write("let uvtname {}/line\n".format(uid))
            file.write("let fittab {}/line_result\n".format(uid))
            file.write("let resname {}/line_residual\n".format(uid))
            file.write("let xoff {}\n".format(Gaussian_par[0]))
            file.write("let yoff {}\n".format(Gaussian_par[1]))
            file.write("let flux {}\n".format(Gaussian_par[2]))
            file.write("let maj {}\n".format(Gaussian_par[3]))
            file.write("let q {}\n".format(Gaussian_par[4]/Gaussian_par[3])) ## min is changed to ratio in new version gildas
            file.write("let pa {}\n".format(Gaussian_par[5]))
            file.write("let nu {}\n".format(Gaussian_par[6]))
            file.write("run uv_fit uv_fit.init /nowindow\n")
            file.write("sic copy \\home\\wyx\\.gag\\logs\\uv_fit.gildas test_log/line_result_{}.gildas\n".format(uid))
        
        print("  Created: {}".format(filename_line))
        return True
    
    except Exception as e:
        print("  ERROR generating GILDAS script: {}".format(str(e)))
        return False



def generate_linemap(target, work_dir, line_factor=0.7, cont_factor=1.5):
    """Phase 3: 生成 line map"""
    base_name = target['name']
    uid = target['uid']
    row_position_id = target.get('position_id', 'default')

    print("\n" + "=" * 60)
    print("[Phase 3] Processing: {} (uid={}, position_id={})".format(base_name, uid, row_position_id))
    print("=" * 60)
    
    cube_dir = os.path.join(work_dir, 'Each_target_img', base_name, 'cubes')
    clean_cube = resolve_clean_cube(work_dir, base_name, target)
    summary_file = os.path.join(cube_dir, '{}_gaussian_fit_summary.txt'.format(uid))
    
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

    line_chan_min, line_chan_max = sorted_clipped_channel_pair(
        line_chan_min, line_chan_max, nchan)
    cont_chan_min, cont_chan_max = sorted_clipped_channel_pair(
        cont_chan_min, cont_chan_max, nchan)
    
    print("  Total channels: {}".format(nchan))
    print("  Line channels: {} - {}".format(line_chan_min, line_chan_max))
    print("  Cont exclusion channels: {} - {}".format(cont_chan_min, cont_chan_max))
    
    try:
        observed_channels = get_valid_cube_channels(clean_cube)
    except Exception as e:
        print("  ERROR: Could not identify observed cube channels: {}".format(
            str(e)))
        return False

    # 保留首尾各两个 channel 的保护区，再排除谱线周围的 continuum
    # exclusion window。只有实际观测到的 channel 才传给 imcontsub。
    continuum_mask = observed_channels.copy()
    continuum_mask[:min(2, nchan)] = False
    continuum_mask[max(0, nchan - 2):] = False
    continuum_mask[cont_chan_min:cont_chan_max + 1] = False
    cont_sub_ch = channels_to_selection(continuum_mask)

    if not cont_sub_ch:
        print("  ERROR: No valid continuum channels available")
        return False

    line_mask = np.zeros(nchan, dtype=bool)
    line_mask[line_chan_min:line_chan_max + 1] = True
    line_mask &= observed_channels
    line_map_chan = channels_to_selection(line_mask)
    if not line_map_chan:
        print("  ERROR: No observed channels fall inside the line window")
        return False
    
    print("  Continuum fitting channels (observed only, N={}): {}".format(
        int(np.sum(continuum_mask)), cont_sub_ch))
    print("  Line map channels (observed only, N={}): {}".format(
        int(np.sum(line_mask)), line_map_chan))
    
    clean_line_im = os.path.join(cube_dir, '{}_line.image'.format(uid))
    line_map_im = os.path.join(cube_dir, '{}.line_map'.format(uid))
    
    
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
    
    imhead(imagename=clean_line_im, mode='put', hdkey='restfreq', hdvalue=str(line_cen)+'GHz')
    
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

    box_str = build_imfit_box_from_target_pixel(line_map_im, target, box_size_arcsec=5.0)
    imfit(
            imagename=line_map_im,
            box=box_str,
            residual=cube_dir + '/'+str(uid)+'_imfit.residual',
            logfile=cube_dir + '/'+str(uid)+'_imfit.log',
            overwrite=True
        )

    print("\n  [Phase 3] SUCCESS: {} line map generated".format(uid))
    return True



def generate_gildas_only(target, work_dir):
    """Phase 4: 只生成 GILDAS 脚本（不重复运行 Phase 3）"""
    base_name = target['name']
    uid = target['uid']
    row_position_id = target.get('position_id', 'default')

    print("\n" + "=" * 60)
    print("[Phase 4] Processing: {} (uid={}, position_id={})".format(base_name, uid, row_position_id))
    print("=" * 60)
    
    cube_dir = os.path.join(work_dir, 'Each_target_img', base_name, 'cubes')
    line_map_im = os.path.join(cube_dir, '{}.line_map'.format(uid))
    
    if os.path.exists(line_map_im):
        print("  Running imfit to get parameters...")
        Gaussian_par = record_imfit_result(line_map_im, cube_dir, target)
    else:
        print("  ERROR: line_map not found: {}".format(line_map_im))
        print("  Please run Phase 3 first (--linemap-only)")
        return False
    
    return generate_gildas_script(base_name, uid, work_dir, Gaussian_par)



def main():
    """主函数"""
    work_dir = os.getcwd()
    csv_file = os.path.join(work_dir, 'target_line_list.csv')
    
    target_name = None
    position_id = None
    phase4_only = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        if arg == '--phase4':
            phase4_only = True
            i += 1
            continue

        if arg == '--position-id':
            if i + 1 >= len(args):
                print("Error: --position-id requires a value")
                return
            position_id = args[i + 1]
            i += 2
            continue

        if arg.startswith('--position-id='):
            position_id = arg.split('=', 1)[1]
            i += 1
            continue

        if not arg.startswith('-') and not arg.endswith('.py'):
            target_name = arg

        i += 1
    
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

    if position_id:
        print("Position ID: {}".format(position_id))

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

    if position_id:
        targets = [t for t in targets if t.get('position_id') == position_id]
        if len(targets) == 0:
            if target_name:
                print("Target '{}' with position_id '{}' not found in CSV file.".format(target_name, position_id))
            else:
                print("No targets found for position_id '{}' in CSV file.".format(position_id))
            return
    
    success_count = 0
    for tgt in targets:
        if phase4_only:
            if generate_gildas_only(tgt, work_dir):
                success_count += 1
        else:
            if generate_linemap(tgt, work_dir):
                success_count += 1
    
    print("\n" + "=" * 70)
    print("Completed: {}/{} targets processed successfully".format(
        success_count, len(targets)))
    print("=" * 70)



if __name__ == '__main__' or 'casa' in dir():
    main()
