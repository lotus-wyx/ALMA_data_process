#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Phase 2: Spectral Extraction and Gaussian Fitting

从FITS cube中提取目标源光谱并进行高斯拟合（纯Python，不需要CASA）
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

C_KMS = 299792.458
DEFAULT_CATALOG_FILE = 'emission_line_fit_catalog.csv'

def infer_band(target_row, target_name=None, work_dir='.'):
    """返回动态频率组名；优先使用 CSV band，再根据 MS manifest 匹配频率。"""
    band = (target_row.get('band') or '').strip().lower()
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
                frequency_ghz = float(target_row.get('line_freq_GHz', ''))
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
    # 这样会回退到旧的 <target>.image；新项目应先运行 Level 4 生成 manifest。
    return None

def group_file_suffix(group):
    """将动态组名转换为文件后缀（band_01 -> band_01，旧 low -> band_low）。"""
    return group if group.startswith('band_') else 'band_{}'.format(group)

def resolve_image_file(base_name, target_row, work_dir='.'):
    """优先按 band 查找图像，不存在时回退到旧的 <target>.image。"""
    band = infer_band(target_row, base_name, work_dir)
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


def resolve_mfs_dirty_fits(base_name, target_row, work_dir='.'):
    """返回与目标 cube 同组的二维 MFS dirty FITS 路径。"""
    image_file = resolve_image_file(base_name, target_row, work_dir)
    suffix = '.image'
    if image_file.endswith(suffix):
        image_stem = image_file[:-len(suffix)]
    else:
        image_stem = image_file
    return image_stem + '_mfs_dirty.image.fits', image_file + '.fits'


def find_local_dirty_peak(mfs_fits, initial_x, initial_y,
                          radius_beams=1, min_peak_snr=5.0,
                          beam_fallback_fits=None):
    """在初始位置附近搜索显著的正峰值，返回像素位置和诊断量。"""
    with fits.open(mfs_fits, memmap=True) as hdul:
        image = np.squeeze(np.asarray(hdul[0].data, dtype=float))
        header = hdul[0].header

    if image.ndim != 2:
        raise ValueError(
            "Expected a 2D MFS image after squeezing, got shape {}".format(
                image.shape))

    cell_arcsec = abs(float(header['CDELT1'])) * 3600.0
    beam_major_deg = header.get('BMAJ')
    if beam_major_deg is None and beam_fallback_fits:
        with fits.open(beam_fallback_fits, memmap=True) as hdul:
            beam_major_deg = hdul[0].header.get('BMAJ')
    if beam_major_deg is None:
        raise KeyError("BMAJ not found in MFS or cube FITS header")
    if cell_arcsec <= 0:
        raise ValueError("Invalid spatial pixel size in {}".format(mfs_fits))

    beam_major_pix = abs(float(beam_major_deg)) * 3600.0 / cell_arcsec
    search_radius_pix = max(1.0, radius_beams * beam_major_pix)
    ny, nx = image.shape
    if not (0 <= initial_x < nx and 0 <= initial_y < ny):
        raise ValueError(
            "Initial pixel ({:.2f}, {:.2f}) is outside image {}x{}".format(
                initial_x, initial_y, nx, ny))

    x_min = max(0, int(np.floor(initial_x - search_radius_pix)))
    x_max = min(nx, int(np.ceil(initial_x + search_radius_pix)) + 1)
    y_min = max(0, int(np.floor(initial_y - search_radius_pix)))
    y_max = min(ny, int(np.ceil(initial_y + search_radius_pix)) + 1)
    y_grid, x_grid = np.ogrid[y_min:y_max, x_min:x_max]
    local_mask = (
        (x_grid - initial_x)**2 + (y_grid - initial_y)**2
        <= search_radius_pix**2)
    subimage = image[y_min:y_max, x_min:x_max]
    valid_local = local_mask & np.isfinite(subimage)
    if not np.any(valid_local):
        raise ValueError("No finite pixels in the recenter search region")

    peak_flat_index = np.nanargmax(np.where(valid_local, subimage, np.nan))
    peak_local_y, peak_local_x = np.unravel_index(
        peak_flat_index, subimage.shape)
    peak_x = int(x_min + peak_local_x)
    peak_y = int(y_min + peak_local_y)
    peak_value = float(image[peak_y, peak_x])

    finite_image = image[np.isfinite(image)]
    if finite_image.size < 10:
        raise ValueError("Not enough finite pixels to estimate MFS image noise")
    _, background_median, background_rms = sigma_clipped_stats(
        finite_image, sigma=3.0)
    background_median = float(background_median)
    background_rms = float(background_rms)
    if not np.isfinite(background_rms) or background_rms <= 0:
        raise ValueError("Could not estimate a positive MFS image RMS")

    peak_snr = (peak_value - background_median) / background_rms
    shift_pix = float(np.hypot(peak_x - initial_x, peak_y - initial_y))
    return {
        'accepted': bool(np.isfinite(peak_snr) and peak_snr >= min_peak_snr),
        'pixel_x': peak_x,
        'pixel_y': peak_y,
        'peak_value': peak_value,
        'peak_snr': float(peak_snr),
        'shift_pix': shift_pix,
        'beam_major_pix': beam_major_pix,
        'search_radius_pix': search_radius_pix,
    }


def recenter_csv_positions(csv_file, target_name=None, position_id=None,
                           radius_beams=1.5, min_peak_snr=5.0,
                           work_dir='.'):
    """用 MFS dirty 图的局部峰更新 CSV 像素坐标，并保留原坐标。"""
    if not os.path.exists(csv_file):
        raise IOError("CSV file not found: {}".format(csv_file))

    with open(csv_file, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required = ['name', 'pixel_x', 'pixel_y']
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError("CSV is missing required column(s): {}".format(
            ', '.join(missing)))

    diagnostic_fields = [
        'pixel_x_original', 'pixel_y_original',
        'recenter_peak_snr', 'recenter_shift_pix'
    ]
    for key in diagnostic_fields:
        if key not in fieldnames:
            fieldnames.append(key)

    selected_count = 0
    updated_count = 0
    for row in rows:
        name = (row.get('name') or '').strip()
        row_position_id = (row.get('position_id') or 'default').strip() or 'default'
        if target_name is not None and name != target_name:
            continue
        if position_id is not None and row_position_id != position_id:
            continue

        selected_count += 1
        try:
            current_x = float(row['pixel_x'])
            current_y = float(row['pixel_y'])
            original_x = (row.get('pixel_x_original') or '').strip()
            original_y = (row.get('pixel_y_original') or '').strip()
            search_x = float(original_x) if original_x else current_x
            search_y = float(original_y) if original_y else current_y
            mfs_fits, cube_fits = resolve_mfs_dirty_fits(name, row, work_dir)
            if not os.path.exists(mfs_fits):
                raise IOError("MFS dirty FITS not found: {}".format(mfs_fits))

            result = find_local_dirty_peak(
                mfs_fits, search_x, search_y,
                radius_beams=radius_beams,
                min_peak_snr=min_peak_snr,
                beam_fallback_fits=cube_fits if os.path.exists(cube_fits) else None)
            print(
                "Recenter {} [{}]: ({:.2f}, {:.2f}) -> ({}, {}), "
                "shift={:.2f} pix, peak SNR={:.2f}".format(
                    name, row_position_id, search_x, search_y,
                    result['pixel_x'], result['pixel_y'],
                    result['shift_pix'], result['peak_snr']))

            row['recenter_peak_snr'] = '{:.3f}'.format(result['peak_snr'])
            row['recenter_shift_pix'] = '{:.3f}'.format(result['shift_pix'])
            if not result['accepted']:
                print("  Peak is below {:.1f} sigma; keeping the original position".format(
                    min_peak_snr))
                continue

            if not (row.get('pixel_x_original') or '').strip():
                row['pixel_x_original'] = row['pixel_x']
            if not (row.get('pixel_y_original') or '').strip():
                row['pixel_y_original'] = row['pixel_y']
            row['pixel_x'] = str(result['pixel_x'])
            row['pixel_y'] = str(result['pixel_y'])
            updated_count += 1
        except Exception as exc:
            print("Warning: could not recenter {} [{}]: {}".format(
                name, row_position_id, str(exc)))

    if selected_count == 0:
        print("Warning: no CSV rows matched the recenter selection")
        return 0

    temp_csv = csv_file + '.recenter.tmp'
    try:
        with open(temp_csv, 'w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_csv, csv_file)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

    print("Recentered {}/{} selected CSV row(s)".format(
        updated_count, selected_count))
    return updated_count


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



def extract_circle_aperture(cent_x, cent_y, freq_len, x_len, y_len, aperture_pix):
    """创建圆形孔径mask"""
    ra_grid, dec_grid = np.meshgrid(np.arange(x_len), np.arange(y_len))
    distance_grid = np.sqrt((ra_grid - cent_x)**2 + (dec_grid - cent_y)**2)
    
    circle_mask = distance_grid <= aperture_pix
    mask_expanded = np.expand_dims(circle_mask, axis=(0, 1))
    spec_mask = np.repeat(mask_expanded, freq_len, axis=1)
    
    return distance_grid, spec_mask


# def extract_spectrum(fits_file, pixel_x, pixel_y, aperture_factor, output_dir):
#     """从FITS cube中提取光谱（4种孔径）"""
#     print("Extracting spectrum from: {}".format(fits_file))
    
#     results = {}
    
#     with fits.open(fits_file, memmap=True) as hdul:  # 使用 memmap 避免全量加载
#         data = hdul[0].data
#         header = hdul[0].header
        
#         cell = np.abs(header['CDELT1'] * 3600)
#         nchan = header['NAXIS3']
#         ref_freq = header['CRVAL3']
#         freq_step = header['CDELT3']
#         ref_freq_chanid = header['CRPIX3']
        
#         if len(hdul) == 1: # or 'BMAJ' in header:
#             print("  Using common beam from header")
#             beam_maj_single = header['BMAJ'] * 3600
#             beam_min_single = header['BMIN'] * 3600
            
#             beam_area_pixel = np.pi * beam_maj_single * beam_min_single / 4 / np.log(2) / cell**2
            
#             for i in range(nchan):
#                 data[0, i, :, :] = data[0, i, :, :] / beam_area_pixel
            
#             beam_maj_ap = beam_maj_single
            
#         else:
#             print("  Using per-channel beams ({} channels)".format(nchan))
#             beam_info = hdul[1].data
#             beam_maj = np.array([beam_[0] for beam_ in beam_info])
#             beam_min = np.array([beam_[1] for beam_ in beam_info])
#             beam_maj_ap = np.max(beam_maj)
            
#             beam_area_pixel = np.pi * beam_maj * beam_min / 4 / np.log(2) / cell**2
            
#             for i in range(nchan):
#                 data[0, i, :, :] = data[0, i, :, :] / beam_area_pixel[i]
        
#         freq_list = np.array([ref_freq + freq_step * (i + 1 - ref_freq_chanid) 
#                               for i in range(nchan)]) / 1e9
        
#         aperture_factors = [0.5, 1.0, 1.5, 2.0]
        
#         for n in aperture_factors:
#             print("  Aperture factor: {}x".format(n))
            
#             aperture_pix = beam_maj_ap / cell / 2 * n
#             print("    Aperture radius: {:.2f} pixels".format(aperture_pix))
            
#             distance_grid, spec_mask = extract_circle_aperture(
#                 pixel_x, pixel_y, data.shape[1], data.shape[2], data.shape[3], aperture_pix
#             )
#             target_spectrum = np.nansum(data * spec_mask, axis=(2, 3))[0]
            
#             print("    Sampling background noise...")
#             bkg_mask = distance_grid > aperture_pix * 2
#             bkg_y, bkg_x = np.where(bkg_mask)
            
#             if len(bkg_y) < 100:
#                 print("    Warning: Not enough background pixels")
#                 n_samples = len(bkg_y)
#             else:
#                 n_samples = 100
            
#             indices = np.random.choice(len(bkg_y), size=n_samples, replace=False)
#             sample_x = bkg_x[indices]
#             sample_y = bkg_y[indices]
            
#             bkg_spectra = []
#             for sx, sy in zip(sample_x, sample_y):
#                 _, bkg_spec_mask = extract_circle_aperture(
#                     sx, sy, data.shape[1], data.shape[2], data.shape[3], aperture_pix
#                 )
#                 bkg_spectrum = np.nansum(data * bkg_spec_mask, axis=(2, 3))[0]
#                 bkg_spectra.append(bkg_spectrum)
            
#             bkg_spectra = np.array(bkg_spectra)
#             mean, bkg_flux, bkg_noise = sigma_clipped_stats(bkg_spectra, axis=0)
            
#             results[n] = {
#                 'freq': freq_list,
#                 'flux': target_spectrum,
#                 'error': bkg_noise,
#                 'bkg': bkg_flux,
#                 'aperture_pix': aperture_pix
#             }
    
#     return results
def extract_spectrum(fits_file, pixel_x, pixel_y, extraction_mode='aperture'):
    """从 FITS cube 提取孔径光谱和/或中心像素光谱。"""
    print("Extracting spectrum from: {}".format(fits_file))

    extraction_mode = (extraction_mode or 'aperture').lower()
    if extraction_mode not in ['aperture', 'point', 'both']:
        raise ValueError("Unknown extraction mode: {}".format(extraction_mode))

    extract_apertures = extraction_mode in ['aperture', 'both']
    extract_point = extraction_mode in ['point', 'both']
    
    results = {}
    
    with fits.open(fits_file, memmap=True) as hdul:  # 使用 memmap 避免全量加载
        header = hdul[0].header
        
        cell = np.abs(header['CDELT1'] * 3600)
        nchan = header['NAXIS3']
        ny = header['NAXIS2']
        nx = header['NAXIS1']
        ref_freq = header['CRVAL3']
        freq_step = header['CDELT3']
        ref_freq_chanid = header['CRPIX3']
        
        # 计算频率列表（不需要读取数据）
        freq_list = np.array([ref_freq + freq_step * (i + 1 - ref_freq_chanid) 
                              for i in range(nchan)]) / 1e9
        
        # 获取 beam 信息
        if len(hdul) == 1:
            print("  Using common beam from header")
            beam_maj_single = header['BMAJ'] * 3600
            beam_min_single = header['BMIN'] * 3600
            beam_area_pixel = np.pi * beam_maj_single * beam_min_single / 4 / np.log(2) / cell**2
            beam_maj_ap = beam_maj_single
            beam_per_channel = None
        else:
            print("  Using per-channel beams ({} channels)".format(nchan))
            beam_info = hdul[1].data
            beam_maj = np.array([beam_[0] for beam_ in beam_info])
            beam_min = np.array([beam_[1] for beam_ in beam_info])
            beam_maj_ap = np.max(beam_maj)
            beam_area_pixel = np.pi * beam_maj * beam_min / 4 / np.log(2) / cell**2
            beam_per_channel = beam_area_pixel
        
        aperture_factors = [0.5, 1.0, 1.5, 2.0]
        max_aperture = max(aperture_factors)
        max_aperture_pix = beam_maj_ap / cell / 2 * max_aperture
        
        # 确定需要读取的区域（比最大孔径稍大）
        buffer = int(np.ceil(max_aperture_pix * 2.5))
        x_min = max(0, int(pixel_x) - buffer)
        x_max = min(nx, int(pixel_x) + buffer + 1)
        y_min = max(0, int(pixel_y) - buffer)
        y_max = min(ny, int(pixel_y) + buffer + 1)
        
        print("  Reading subregion: x=[{},{}], y=[{},{}] ({}x{} pixels)".format(
            x_min, x_max, y_min, y_max, x_max-x_min, y_max-y_min))
        
        # 只读取需要的子区域；原始单位通常为 Jy/beam。
        data_sub = hdul[0].data[0, :, y_min:y_max, x_min:x_max].astype(np.float32)
        
        # 调整像素坐标到子区域
        pixel_x_sub = pixel_x - x_min
        pixel_y_sub = pixel_y - y_min
        
        if extract_point:
            point_x = int(round(pixel_x_sub))
            point_y = int(round(pixel_y_sub))
            if not (0 <= point_x < data_sub.shape[2] and
                    0 <= point_y < data_sub.shape[1]):
                raise ValueError(
                    "Point pixel ({:.2f}, {:.2f}) is outside the FITS image".format(
                        pixel_x, pixel_y))

            # 不做 peak search：直接取用户给定坐标的最近像素，避免噪声选择偏差。
            point_spectrum = np.array(data_sub[:, point_y, point_x], dtype=float)
            y_grid, x_grid = np.ogrid[:data_sub.shape[1], :data_sub.shape[2]]
            distance_grid = np.sqrt(
                (x_grid - pixel_x_sub)**2 + (y_grid - pixel_y_sub)**2)
            point_exclusion_pix = 2.0 * beam_maj_ap / cell
            point_bkg_mask = distance_grid > point_exclusion_pix

            if np.any(point_bkg_mask):
                point_bkg_pixels = data_sub[:, point_bkg_mask]
                _, point_bkg, point_noise = sigma_clipped_stats(
                    point_bkg_pixels, sigma=3.0, axis=1)
                point_bkg = np.asarray(point_bkg, dtype=float)
                point_noise = np.asarray(point_noise, dtype=float)
            else:
                print("    Warning: Not enough background pixels for point extraction")
                point_bkg = np.full(nchan, np.nan)
                point_noise = np.full(nchan, np.nan)

            bunit = str(header.get('BUNIT', 'Jy/beam')).strip() or 'Jy/beam'
            if 'jy/beam' not in bunit.lower().replace(' ', ''):
                print("    Warning: point spectrum keeps FITS unit '{}'".format(bunit))
            print("  Point extraction: nearest pixel ({}, {})".format(
                int(round(pixel_x)), int(round(pixel_y))))
            print("    Background exclusion radius: {:.2f} pixels".format(
                point_exclusion_pix))
            results['point'] = {
                'freq': freq_list,
                'flux': point_spectrum,
                'error': point_noise,
                'bkg': point_bkg,
                'aperture_pix': 0.0,
                'unit': bunit,
                'description': 'Point (nearest pixel)'
            }

        if not extract_apertures:
            return results

        # 孔径积分需要先将每像素的 Jy/beam 转成 Jy/pixel。
        if beam_per_channel is None:
            data_sub = data_sub / beam_area_pixel
        else:
            data_sub = data_sub / beam_per_channel[:, np.newaxis, np.newaxis]
        
        for n in aperture_factors:
            print("  Aperture factor: {}x".format(n))
            
            aperture_pix = beam_maj_ap / cell / 2 * n
            print("    Aperture radius: {:.2f} pixels".format(aperture_pix))
            
            # 在子区域上创建 mask
            y_grid, x_grid = np.ogrid[:data_sub.shape[1], :data_sub.shape[2]]
            distance_grid = np.sqrt((x_grid - pixel_x_sub)**2 + (y_grid - pixel_y_sub)**2)
            
            circle_mask = distance_grid <= aperture_pix
            
            # 提取目标光谱（向量化操作）
            target_spectrum = np.array([np.nansum(data_sub[i][circle_mask]) for i in range(nchan)])
            
            # 背景采样
            print("    Sampling background noise...")
            bkg_mask = distance_grid > aperture_pix * 2
            bkg_y, bkg_x = np.where(bkg_mask)
            
            if len(bkg_y) < 100:
                print("    Warning: Not enough background pixels")
                n_samples = min(len(bkg_y), 50)
            else:
                n_samples = 100
            
            if n_samples > 0:
                indices = np.random.choice(len(bkg_y), size=n_samples, replace=False)
                sample_x = bkg_x[indices]
                sample_y = bkg_y[indices]
                
                # 批量提取背景光谱（向量化）
                bkg_spectra = []
                for sx, sy in zip(sample_x, sample_y):
                    bkg_distance = np.sqrt((x_grid - sx)**2 + (y_grid - sy)**2)
                    bkg_circle = bkg_distance <= aperture_pix
                    bkg_spectrum = np.array([np.nansum(data_sub[i][bkg_circle]) for i in range(nchan)])
                    bkg_spectra.append(bkg_spectrum)
                
                bkg_spectra = np.array(bkg_spectra)
                mean, bkg_flux, bkg_noise = sigma_clipped_stats(bkg_spectra, axis=0)
            else:
                bkg_flux = np.zeros(nchan)
                bkg_noise = np.ones(nchan) * np.nan
            
            results[n] = {
                'freq': freq_list,
                'flux': target_spectrum,
                'error': bkg_noise,
                'bkg': bkg_flux,
                'aperture_pix': aperture_pix,
                'unit': 'Jy',
                'description': 'Aperture: {}x beam_maj/2'.format(n)
            }
    
    return results

def gaussian_plus_const(freq, amplitude, center, sigma, constant):
    """高斯线 + 常数背景模型"""
    return amplitude * np.exp(-0.5 * ((freq - center) / sigma)**2) + constant


def fit_gaussian(freq, flux, error, line_freq_prior, fit_range_ghz=4.0):
    """对光谱进行高斯拟合"""
    half_range = fit_range_ghz / 2.0

    freq = np.asarray(freq, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    finite_freq = np.isfinite(freq)
    valid_channel = (
        finite_freq & np.isfinite(flux) & (flux != 0)
        & np.isfinite(error) & (error > 0))

    valid_freq = freq[valid_channel]
    if valid_freq.size == 0:
        print("    Warning: Spectrum contains no valid channels")
        return None, None, np.nan, np.nan, np.nan, np.nan

    # 用通道边缘而不是首末通道中心定义物理覆盖范围，否则一个实际
    # 4 GHz 的 cube 会因为少算首末两个半通道而显示成略小于 4 GHz。
    sorted_freq = np.unique(np.sort(valid_freq))
    freq_diffs = np.diff(sorted_freq)
    freq_diffs = freq_diffs[np.isfinite(freq_diffs) & (freq_diffs > 0)]
    channel_width_ghz = float(np.median(freq_diffs)) if freq_diffs.size else 0.0
    coverage_min = float(np.min(valid_freq) - 0.5 * channel_width_ghz)
    coverage_max = float(np.max(valid_freq) + 0.5 * channel_width_ghz)
    available_width = coverage_max - coverage_min

    if available_width <= fit_range_ghz:
        fit_freq_min = coverage_min
        fit_freq_max = coverage_max
    else:
        desired_min = line_freq_prior - half_range
        latest_start = coverage_max - fit_range_ghz
        fit_freq_min = float(np.clip(desired_min, coverage_min, latest_start))
        fit_freq_max = fit_freq_min + fit_range_ghz

    sigma_low = 100/3/10**5*line_freq_prior/(8*np.log(2))**0.5    # 100 km/s 对应的频率宽度（下限）
    sigma_max = 1000/3/10**5*line_freq_prior/(8*np.log(2))**0.5   # 1000 km/s 对应的频率宽度（上限）
    sigma_typical = 300/3/10**5*line_freq_prior/(8*np.log(2))**0.5  # 300 km/s 对应的频率宽度（初始值）

    print("    Fit frequency range: {:.6f}-{:.6f} GHz ({:.3f} GHz total)".format(
        fit_freq_min, fit_freq_max, fit_freq_max - fit_freq_min))
    if available_width + 1e-9 < fit_range_ghz:
        print("    Warning: requested {:.3f} GHz, but only {:.3f} GHz of "
              "valid cube coverage is available".format(
                  fit_range_ghz, available_width))

    window_mask = finite_freq & (freq >= fit_freq_min) & (freq <= fit_freq_max)
    fit_mask = window_mask & valid_channel
    n_invalid = int(np.sum(window_mask & ~valid_channel))
    if n_invalid > 0:
        print("Filtered {} invalid data points (NaN/Inf/zero-error)".format(n_invalid))

    freq_fit = freq[fit_mask]
    flux_fit = flux[fit_mask]
    error_fit = error[fit_mask]
    
    if len(freq_fit) < 5:
        print("    Warning: Not enough valid data points in fit range ({})".format(len(freq_fit)))
        return None, None, np.nan, np.nan, fit_freq_min, fit_freq_max
    
    flux_range = np.max(flux_fit) - np.min(flux_fit)
    # 扩大数据窗口是为了增加连续谱通道，不应同时放宽线心到整段窗口。
    # 最多保留旧版 2 GHz 窗口对应的 line prior +/- 1 GHz 搜索范围。
    center_half_range = min(half_range, 1.0)
    center_low = max(np.min(freq_fit), line_freq_prior - center_half_range)
    center_high = min(np.max(freq_fit), line_freq_prior + center_half_range)
    if center_low >= center_high:
        center_low = np.min(freq_fit)
        center_high = np.max(freq_fit)
    initial_center = np.clip(line_freq_prior, center_low, center_high)
    p0 = [flux_range, initial_center, sigma_typical, np.median(flux_fit)]
    
    bounds = ([0, center_low, sigma_low, -np.inf],
              [np.inf, center_high, sigma_max, np.inf])
    
    try:
        popt, pcov = curve_fit(
            gaussian_plus_const, freq_fit, flux_fit, 
            p0=p0, sigma=error_fit, absolute_sigma=True, bounds=bounds, maxfev=5000
        )
        
        perr = np.sqrt(np.diag(pcov))
        
        model = gaussian_plus_const(freq_fit, *popt)
        residuals = flux_fit - model
        chi2 = np.sum((residuals / error_fit)**2)
        dof = len(freq_fit) - len(popt)
        chi2_dof = chi2 / dof if dof > 0 else np.nan
        
        snr = popt[0] / np.median(error_fit)
        
        return popt, perr, chi2_dof, snr, fit_freq_min, fit_freq_max
        
    except Exception as e:
        print("    Fit failed: {}".format(str(e)))
        return None, None, np.nan, np.nan, fit_freq_min, fit_freq_max


def integrate_line_window(freq, flux, error, fit_params, window_factor=0.7):
    """Integrate continuum-subtracted flux within +/- window_factor*FWHM.

    The returned line flux is in Jy km/s.  The uncertainty follows the
    conventional independent-channel estimate, sqrt(sum((sigma_i*dV)^2)).
    The fitted constant continuum level is subtracted before integration.
    """
    if fit_params is None or len(fit_params) < 4:
        return {
            'flux': np.nan, 'error': np.nan, 'snr': np.nan,
            'fwhm_ghz': np.nan, 'nchan': 0,
            'freq_min': np.nan, 'freq_max': np.nan,
        }

    center = float(fit_params[1])
    sigma = float(fit_params[2])
    continuum = float(fit_params[3])
    fwhm_ghz = 2.354820045 * abs(sigma)
    half_width_ghz = window_factor * fwhm_ghz

    freq = np.asarray(freq, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    finite_freq = np.isfinite(freq)
    window_mask = (
        finite_freq
        & (freq >= center - half_width_ghz)
        & (freq <= center + half_width_ghz)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > 0)
    )

    nchan = int(np.sum(window_mask))
    if nchan == 0 or not np.isfinite(center) or center == 0:
        return {
            'flux': np.nan, 'error': np.nan, 'snr': np.nan,
            'fwhm_ghz': fwhm_ghz, 'nchan': nchan,
            'freq_min': center - half_width_ghz,
            'freq_max': center + half_width_ghz,
        }

    # The cubes are not intentionally binned, so use the native channel
    # spacing.  abs() handles either increasing or decreasing FITS axes.
    freq_diffs = np.abs(np.diff(freq[finite_freq]))
    freq_diffs = freq_diffs[np.isfinite(freq_diffs) & (freq_diffs > 0)]
    if len(freq_diffs) == 0:
        return {
            'flux': np.nan, 'error': np.nan, 'snr': np.nan,
            'fwhm_ghz': fwhm_ghz, 'nchan': nchan,
            'freq_min': center - half_width_ghz,
            'freq_max': center + half_width_ghz,
        }

    channel_width_ghz = float(np.median(freq_diffs))
    channel_width_kms = 299792.458 * channel_width_ghz / abs(center)
    line_flux = np.sum((flux[window_mask] - continuum) * channel_width_kms)
    line_error = np.sqrt(np.sum((error[window_mask] * channel_width_kms) ** 2))
    line_snr = line_flux / line_error if line_error > 0 else np.nan

    return {
        'flux': float(line_flux),
        'error': float(line_error),
        'snr': float(line_snr),
        'fwhm_ghz': fwhm_ghz,
        'nchan': nchan,
        'freq_min': center - half_width_ghz,
        'freq_max': center + half_width_ghz,
    }


def invalid_channel_spans(freq, valid_mask):
    """返回连续无效 channel 对应的频率边界。"""
    freq = np.asarray(freq, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    finite_indices = np.flatnonzero(np.isfinite(freq))
    if len(finite_indices) == 0:
        return []

    order = finite_indices[np.argsort(freq[finite_indices])]
    sorted_freq = freq[order]
    sorted_invalid = ~valid_mask[order]
    if not np.any(sorted_invalid):
        return []

    if len(sorted_freq) == 1:
        return [(sorted_freq[0] - 0.5, sorted_freq[0] + 0.5)]

    channel_edges = np.empty(len(sorted_freq) + 1, dtype=float)
    channel_edges[1:-1] = 0.5 * (sorted_freq[:-1] + sorted_freq[1:])
    channel_edges[0] = sorted_freq[0] - 0.5 * (
        sorted_freq[1] - sorted_freq[0])
    channel_edges[-1] = sorted_freq[-1] + 0.5 * (
        sorted_freq[-1] - sorted_freq[-2])

    invalid_indices = np.flatnonzero(sorted_invalid)
    spans = []
    run_start = invalid_indices[0]
    run_end = invalid_indices[0]
    for index in invalid_indices[1:]:
        if index == run_end + 1:
            run_end = index
        else:
            spans.append((channel_edges[run_start], channel_edges[run_end + 1]))
            run_start = index
            run_end = index
    spans.append((channel_edges[run_start], channel_edges[run_end + 1]))
    return spans


def plot_spectrum_with_fit(freq, flux, error, fit_params, extraction_label,
                        flux_unit, output_file, fit_freq_min=None,
                        fit_freq_max=None):
    """绘制光谱和拟合结果（只显示拟合范围）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # 只使用拟合范围内的数据来确定y轴范围
    if fit_freq_min is not None and fit_freq_max is not None:
        mask = (freq >= fit_freq_min) & (freq <= fit_freq_max)
        freq_plot = freq[mask]
        flux_plot = flux[mask]
        error_plot = error[mask]
    else:
        freq_plot = freq
        flux_plot = flux
        error_plot = error

    # 与拟合使用相同的有效条件。零值或无效误差 channel 不作为数据点
    # 绘制，而是在主图和残差图中以浅灰色频率区段标记。
    valid_mask = (
        np.isfinite(freq_plot)
        & np.isfinite(flux_plot)
        & (flux_plot != 0)
        & np.isfinite(error_plot)
        & (error_plot > 0)
    )
    freq_valid = freq_plot[valid_mask]
    flux_valid = flux_plot[valid_mask]
    error_valid = error_plot[valid_mask]

    for span_min, span_max in invalid_channel_spans(freq_plot, valid_mask):
        ax1.axvspan(span_min, span_max, color='0.92', linewidth=0, zorder=0)
        ax2.axvspan(span_min, span_max, color='0.92', linewidth=0, zorder=0)

    # 使用 sigma-clipping 排除异常值，确定y轴范围
    if len(flux_valid) > 0:
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(flux_valid, sigma=3.0)
        
        # y轴范围：median ± 5*std（可以捕获大部分有效数据）
        y_min = median - 5 * std
        y_max = median + 5 * std
        
        # 如果有拟合结果，确保拟合峰值在范围内
        if fit_params is not None:
            peak_flux = fit_params[0] + fit_params[3]  # amplitude + constant
            y_max = max(y_max, peak_flux * 1.2)
    else:
        y_min, y_max = None, None
    
    ax1.errorbar(freq_valid, flux_valid, yerr=error_valid, fmt='o', markersize=3,
                 alpha=0.6, label='Data', color='black',zorder=2)
    flux_line = np.where(valid_mask, flux_plot, np.nan)
    ax1.plot(freq_plot, flux_line, c='grey', linewidth=1, alpha=0.8, zorder=1)
    if fit_params is not None:
        freq_model = np.linspace(np.min(freq_plot), np.max(freq_plot), 500)
        flux_model = gaussian_plus_const(freq_model, *fit_params)
        ax1.plot(freq_model, flux_model, 'r-', linewidth=2, label='Gaussian fit')
        ax1.axvline(fit_params[1], color='blue', linestyle='--', 
                   alpha=0.5, label='Fitted center')
    
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Flux ({})'.format(flux_unit), fontsize=12)
    ax1.set_title(extraction_label, fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 设置x轴范围为拟合区域
    if fit_freq_min is not None and fit_freq_max is not None:
        ax1.set_xlim(fit_freq_min, fit_freq_max)
        ax2.set_xlim(fit_freq_min, fit_freq_max)
    
    # 设置y轴范围，排除异常值
    if y_min is not None and y_max is not None:
        ax1.set_ylim(y_min, y_max)
    
    if fit_params is not None and len(freq_valid) > 0:
        model = gaussian_plus_const(freq_valid, *fit_params)
        residuals = flux_valid - model
        ax2.errorbar(freq_valid, residuals, yerr=error_valid, fmt='o', markersize=3,
                    alpha=0.6, color='black')
        ax2.axhline(0, color='red', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Residuals ({})'.format(flux_unit), fontsize=12)
        
        # 残差图也使用sigma-clipping确定y轴范围
        valid_res_mask = np.isfinite(residuals)
        if np.sum(valid_res_mask) > 0:
            res_mean, res_median, res_std = sigma_clipped_stats(residuals[valid_res_mask], sigma=3.0)
            res_y_min = res_median - 5 * res_std
            res_y_max = res_median + 5 * res_std
            ax2.set_ylim(res_y_min, res_y_max)
    
    ax2.set_xlabel('Frequency (GHz)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved plot: {}".format(output_file))

def load_spectrum_result(spec_file):
    """从已写出的 spectrum 文本中读取频谱数据。"""
    if not os.path.exists(spec_file):
        print("  Error: Spectrum file not found: {}".format(spec_file))
        return None

    freq_list = []
    flux_list = []
    error_list = []
    bkg_list = []

    with open(spec_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            try:
                freq_list.append(float(parts[0]))
                flux_list.append(float(parts[1]))
                error_list.append(float(parts[2]))
                bkg_list.append(float(parts[3]))
            except ValueError:
                continue

    if len(freq_list) == 0:
        print("  Error: No valid data found in spectrum file: {}".format(spec_file))
        return None

    return {
        'freq': np.array(freq_list),
        'flux': np.array(flux_list),
        'error': np.array(error_list),
        'bkg': np.array(bkg_list),
    }

def get_extraction_keys(extraction_mode):
    """返回需要处理的抽谱结果键，顺序保持稳定。"""
    keys = []
    if extraction_mode in ['point', 'both']:
        keys.append('point')
    if extraction_mode in ['aperture', 'both']:
        keys.extend([0.5, 1.0, 1.5, 2.0])
    return keys


def extraction_output_info(uid, extraction_key):
    """生成抽谱类型对应的文件名标签。"""
    if extraction_key == 'point':
        return {
            'label': 'point',
            'description': 'Point (nearest pixel)',
            'spec_name': '{}_point.result.txt'.format(uid),
            'plot_name': '{}_point_specfit.pdf'.format(uid),
            'unit': 'Jy/beam',
        }

    return {
        'label': '{:.1f}x'.format(extraction_key),
        'description': 'Aperture: {}x beam_maj/2'.format(extraction_key),
        'spec_name': '{}_{}_no_Apcor.result.txt'.format(uid, extraction_key),
        'plot_name': '{}_{}_specfit.pdf'.format(uid, extraction_key),
        'unit': 'Jy',
    }


def parse_value_error(value):
    """解析 summary 中的 value+/-error 字段。"""
    if '\u00b1' in value:
        parts = value.split('\u00b1', 1)
    elif '+/-' in value:
        parts = value.split('+/-', 1)
    else:
        return float(value), np.nan
    return float(parts[0]), float(parts[1])


def read_fit_summary_rows(summary_file):
    """读取新版或旧版 gaussian fit summary 的数据行。"""
    rows = []
    if not os.path.exists(summary_file):
        return rows

    with open(summary_file, 'r') as handle:
        for line in handle:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                amplitude, amplitude_err = parse_value_error(parts[1])
                center, center_err = parse_value_error(parts[2])
                sigma, sigma_err = parse_value_error(parts[3])
                continuum, continuum_err = parse_value_error(parts[4])
                rows.append({
                    'extraction': parts[0],
                    'amplitude': amplitude,
                    'amplitude_err': amplitude_err,
                    'center': center,
                    'center_err': center_err,
                    'sigma': sigma,
                    'sigma_err': sigma_err,
                    'continuum': continuum,
                    'continuum_err': continuum_err,
                    'chi2_dof': float(parts[5]),
                    'snr_int': float(parts[6]),
                    'line_flux': float(parts[7]),
                    'line_flux_err': float(parts[8]),
                    'peak_snr': float(parts[9]),
                    'nchan': int(parts[10]),
                    'flux_unit': parts[11] if len(parts) >= 12 else 'Jy',
                })
            except (ValueError, IndexError):
                continue
    return rows


def write_detection_catalog(csv_file='target_line_list.csv',
                            output_file=DEFAULT_CATALOG_FILE,
                            min_snr=3.0, work_dir='.'):
    """汇总每个目标积分 SNR 最高且超过阈值的高斯拟合结果。"""
    targets = load_target_list(csv_file)
    catalog_rows = []
    missing_summaries = 0

    for target in targets:
        name = target['name']
        uid = target['uid']
        summary_file = os.path.join(
            work_dir, 'Each_target_img', name, 'cubes',
            '{}_gaussian_fit_summary.txt'.format(uid))
        fit_rows = read_fit_summary_rows(summary_file)
        if not fit_rows:
            missing_summaries += 1
            continue

        finite_rows = [
            row for row in fit_rows
            if np.isfinite(row['snr_int'])
            and np.isfinite(row['center']) and row['center'] > 0
            and np.isfinite(row['sigma']) and row['sigma'] > 0
        ]
        if not finite_rows:
            continue

        best = max(finite_rows, key=lambda row: row['snr_int'])
        if best['snr_int'] <= min_snr:
            continue

        fwhm_ghz = 2.354820045 * abs(best['sigma'])
        fwhm_ghz_err = 2.354820045 * abs(best['sigma_err'])
        fwhm_kms = C_KMS * fwhm_ghz / abs(best['center'])
        if (np.isfinite(fwhm_ghz_err) and fwhm_ghz > 0
                and np.isfinite(best['center_err'])):
            fwhm_kms_err = fwhm_kms * np.sqrt(
                (fwhm_ghz_err / fwhm_ghz) ** 2
                + (best['center_err'] / best['center']) ** 2)
        else:
            fwhm_kms_err = np.nan

        flux_unit = best['flux_unit']
        catalog_rows.append({
            'name': name,
            'position_id': target.get('position_id', 'default'),
            'band': target.get('band', ''),
            'uid': uid,
            'extraction': best['extraction'],
            'line_freq_prior_GHz': target.get('line_freq_GHz', ''),
            'line_center_GHz': '{:.6f}'.format(best['center']),
            'line_center_err_GHz': '{:.6f}'.format(best['center_err']),
            'FWHM_GHz': '{:.6f}'.format(fwhm_ghz),
            'FWHM_err_GHz': '{:.6f}'.format(fwhm_ghz_err),
            'FWHM_kms': '{:.3f}'.format(fwhm_kms),
            'FWHM_err_kms': '{:.3f}'.format(fwhm_kms_err),
            'line_flux': '{:.6e}'.format(best['line_flux']),
            'line_flux_err': '{:.6e}'.format(best['line_flux_err']),
            'line_flux_unit': '{} km/s'.format(flux_unit),
            'integration_window': 'fitted center +/- 0.7 FWHM',
            'SNR_int': '{:.3f}'.format(best['snr_int']),
            'peak_SNR': '{:.3f}'.format(best['peak_snr']),
            'chi2_dof': '{:.3f}'.format(best['chi2_dof']),
            'nchan_integrated': best['nchan'],
        })

    fieldnames = [
        'name', 'position_id', 'band', 'uid', 'extraction',
        'line_freq_prior_GHz', 'line_center_GHz', 'line_center_err_GHz',
        'FWHM_GHz', 'FWHM_err_GHz', 'FWHM_kms', 'FWHM_err_kms',
        'line_flux', 'line_flux_err', 'line_flux_unit', 'integration_window',
        'SNR_int', 'peak_SNR', 'chi2_dof', 'nchan_integrated'
    ]
    with open(output_file, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog_rows)

    print("Wrote emission-line catalog: {}".format(output_file))
    print("  Selection: best integrated-SNR extraction per target, SNR_int > {}".format(
        min_snr))
    print("  Detections: {} / {} target(s)".format(
        len(catalog_rows), len(targets)))
    if missing_summaries > 0:
        print("  Targets without a readable fit summary: {}".format(
            missing_summaries))
    return catalog_rows


def process_target(target_name=None, position_id=None,
                   csv_file='target_line_list.csv', mode='full',
                   extraction_mode='aperture', fit_range_ghz=4.0,
                   recenter=False, recenter_radius_beams=1.5,
                   recenter_min_snr=5.0, catalog_min_snr=3.0,
                   catalog_output=DEFAULT_CATALOG_FILE):
    """处理单个或所有目标源。

    mode:
      - full: 抽谱 + 拟合
      - extract: 只抽谱
      - fit: 只拟合
      - catalog: 只从已有 summary 重建总表
    """
    mode = (mode or 'full').lower()
    if mode not in ['full', 'extract', 'fit', 'catalog']:
        raise ValueError("Unknown mode: {}".format(mode))

    if mode == 'catalog':
        write_detection_catalog(
            csv_file=csv_file, output_file=catalog_output,
            min_snr=catalog_min_snr)
        return

    extraction_mode = (extraction_mode or 'aperture').lower()
    if extraction_mode not in ['aperture', 'point', 'both']:
        raise ValueError("Unknown extraction mode: {}".format(extraction_mode))
    if not np.isfinite(fit_range_ghz) or fit_range_ghz <= 0:
        raise ValueError("fit_range_ghz must be greater than zero")

    need_extract = mode in ['full', 'extract']
    need_fit = mode in ['full', 'fit']

    if recenter and need_extract:
        print("Recentering target pixels from local MFS dirty-image peaks...")
        recenter_csv_positions(
            csv_file, target_name=target_name, position_id=position_id,
            radius_beams=recenter_radius_beams,
            min_peak_snr=recenter_min_snr)
    elif recenter:
        print("Recenter requested in fit-only mode; using coordinates already stored in CSV")

    targets = load_target_list(csv_file)

    if len(targets) == 0:
        print("No targets found in CSV file.")
        return

    if target_name is not None:
        targets = [t for t in targets if t['name'] == target_name]
        if len(targets) == 0:
            print("Target '{}' not found in CSV file.".format(target_name))
            return

    if position_id is not None:
        targets = [t for t in targets if t.get('position_id') == position_id]
        if len(targets) == 0:
            if target_name is not None:
                print("Target '{}' with position_id '{}' not found in CSV file.".format(target_name, position_id))
            else:
                print("No targets found for position_id '{}' in CSV file.".format(position_id))
            return

    print("\n" + "=" * 60)
    print("Phase 2: Extract spectra and/or fit Gaussian")
    print("Mode: {}".format(mode))
    print("Extraction mode: {}".format(extraction_mode))
    print("Fit window: {:.3f} GHz total".format(fit_range_ghz))
    print("Processing {} target(s)".format(len(targets)))
    print("=" * 60)

    for tgt in targets:
        base_name = tgt['name']
        uid = tgt['uid']
        row_position_id = tgt.get('position_id', 'default')
        pixel_x = float(tgt['pixel_x'])
        pixel_y = float(tgt['pixel_y'])
        line_freq = float(tgt['line_freq_GHz'])
        band = infer_band(tgt, base_name)

        print("\n" + "=" * 60)
        print("Target: {}".format(base_name))
        print("UID: {}".format(uid))
        print("Position ID: {}".format(row_position_id))
        print("Position: pixel ({:.2f}, {:.2f})".format(pixel_x, pixel_y))
        print("Line frequency prior: {:.2f} GHz".format(line_freq))
        if band:
            print("Band: {}".format(band))
        print("=" * 60)

        image_file = resolve_image_file(base_name, tgt)
        fits_file = image_file + '.fits'

        if need_extract:
            if not os.path.exists(fits_file):
                print("Error: FITS file not found: {}".format(fits_file))
                print("Please run Phase 1 first: ./alma_project_level_6_emission_line_process.sh --export-only")
                print("Skipping target: {}".format(uid))
                continue

            print("FITS file exists, extracting spectrum...")
            output_dir = 'Each_target_img/{}/cubes'.format(base_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            try:
                spectra = extract_spectrum(
                    fits_file, pixel_x, pixel_y, extraction_mode
                )
            except Exception as e:
                print("Error extracting spectrum: {}".format(str(e)))
                import traceback
                traceback.print_exc()
                continue
        else:
            output_dir = 'Each_target_img/{}/cubes'.format(base_name)
            spectra = None

        fit_summary = [] if need_fit else None

        for extraction_key in get_extraction_keys(extraction_mode):
            output_info = extraction_output_info(uid, extraction_key)
            print("\nProcessing extraction: {}".format(output_info['description']))

            spec_file = os.path.join(output_dir, output_info['spec_name'])

            if need_extract:
                spec_data = spectra[extraction_key]
                flux_unit = spec_data.get('unit', output_info['unit'])

                with open(spec_file, 'w') as f:
                    f.write("# Target: {}\n".format(base_name))
                    f.write("# UID: {}\n".format(uid))
                    f.write("# Position ID: {}\n".format(row_position_id))
                    f.write("# Extraction: {}\n".format(spec_data['description']))
                    if extraction_key != 'point':
                        f.write("# Aperture radius: {:.2f} pixels\n".format(
                            spec_data['aperture_pix']))
                    f.write("# Line frequency prior: {:.4f} GHz\n".format(line_freq))
                    f.write("#\n")
                    f.write("# Frequency(GHz)\tFlux({0})\tError({0})\tBackground({0})\n".format(
                        flux_unit))

                    for freq, flux, err, bkg in zip(
                        spec_data['freq'],
                        spec_data['flux'],
                        spec_data['error'],
                        spec_data['bkg']
                    ):
                        f.write("{:.6f}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(freq, flux, err, bkg))

                print("  Saved spectrum: {}".format(spec_file))

                if not need_fit:
                    continue
            else:
                spec_data = load_spectrum_result(spec_file)
                if spec_data is None:
                    print("  Skipping {} due to missing spectrum file".format(
                        output_info['description']))
                    continue
                flux_unit = output_info['unit']

            print("  Fitting Gaussian...")
            fit_params, fit_errors, chi2_dof, snr, fit_freq_min, fit_freq_max = fit_gaussian(
                spec_data['freq'], spec_data['flux'], spec_data['error'],
                line_freq, fit_range_ghz=fit_range_ghz
            )

            line_integral = integrate_line_window(
                spec_data['freq'], spec_data['flux'], spec_data['error'],
                fit_params, window_factor=0.7
            )
            if fit_params is not None:
                print(
                    "  Integrated line flux (±0.7 FWHM): "
                    "{:.4e} ± {:.4e} {} km/s; SNR={:.2f} (N={})".format(
                        line_integral['flux'], line_integral['error'],
                        flux_unit, line_integral['snr'], line_integral['nchan'])
                )

            plot_file = os.path.join(output_dir, output_info['plot_name'])

            plot_spectrum_with_fit(
                spec_data['freq'], spec_data['flux'], spec_data['error'],
                fit_params, output_info['description'], flux_unit, plot_file,
                fit_freq_min, fit_freq_max
            )

            if fit_params is not None:
                fit_summary.append({
                    'extraction': output_info['label'],
                    'unit': flux_unit,
                    'amplitude': fit_params[0],
                    'amplitude_err': fit_errors[0],
                    'center': fit_params[1],
                    'center_err': fit_errors[1],
                    'sigma': fit_params[2],
                    'sigma_err': fit_errors[2],
                    'constant': fit_params[3],
                    'constant_err': fit_errors[3],
                    'chi2_dof': chi2_dof,
                    # Keep the integrated SNR in the historical SNR column;
                    # retain peak SNR separately for diagnostics.
                    'snr': line_integral['snr'],
                    'peak_snr': snr,
                    'line_flux': line_integral['flux'],
                    'line_flux_err': line_integral['error'],
                    'line_fwhm_ghz': line_integral['fwhm_ghz'],
                    'line_nchan': line_integral['nchan']
                })
            else:
                fit_summary.append({
                    'extraction': output_info['label'],
                    'unit': flux_unit,
                    'amplitude': np.nan,
                    'amplitude_err': np.nan,
                    'center': np.nan,
                    'center_err': np.nan,
                    'sigma': np.nan,
                    'sigma_err': np.nan,
                    'constant': np.nan,
                    'constant_err': np.nan,
                    'chi2_dof': np.nan,
                    'snr': np.nan,
                    'peak_snr': np.nan,
                    'line_flux': np.nan,
                    'line_flux_err': np.nan,
                    'line_fwhm_ghz': np.nan,
                    'line_nchan': 0
                })

        if need_fit:
            summary_file = os.path.join(
                output_dir,
                '{}_gaussian_fit_summary.txt'.format(uid)
            )

            with open(summary_file, 'w') as f:
                f.write("# Gaussian fit results for {}\n".format(uid))
                f.write("# Base target: {}\n".format(base_name))
                f.write("# Position ID: {}\n".format(row_position_id))
                f.write("# Model: flux = A * exp(-0.5*((freq-f0)/sigma)^2) + C\n")
                f.write("# Line frequency prior: {:.4f} GHz\n".format(line_freq))
                f.write("# Fit window: {:.3f} GHz total (clipped to cube coverage)\n".format(
                    fit_range_ghz))
                f.write("# Line-center search: line prior +/- min(fit window / 2, 1 GHz)\n")
                f.write("# FWHM bounds: 100-1000 km/s; initial value: 300 km/s\n")
                f.write("# Integrated line flux: direct sum of (flux - fitted constant continuum)\n")
                f.write("# Integration window: fitted center +/- 0.7 FWHM; uncertainty assumes independent native channels\n")
                f.write("#\n")
                f.write("# Extraction\tAmplitude\tCenter(GHz)\tSigma(GHz)\tConstant\tChi2/dof\tSNR_int\tLineFlux\te_LineFlux\tPeakSNR\tNchan\tFluxUnit\n")

                for fit in fit_summary:
                    f.write("{}\t{:.4e}±{:.4e}\t{:.6f}±{:.6f}\t{:.4f}±{:.4f}\t{:.4e}±{:.4e}\t{:.2f}\t{:.2f}\t{:.4e}\t{:.4e}\t{:.2f}\t{}\t{}\n".format(
                        fit['extraction'],
                        fit['amplitude'], fit['amplitude_err'],
                        fit['center'], fit['center_err'],
                        fit['sigma'], fit['sigma_err'],
                        fit['constant'], fit['constant_err'],
                        fit['chi2_dof'],
                        fit['snr'],
                        fit['line_flux'], fit['line_flux_err'],
                        fit['peak_snr'], fit['line_nchan'], fit['unit']
                    ))

            print("\nSaved fit summary: {}".format(summary_file))

        print("=" * 60)
        print("Completed: {}".format(uid))
        print("=" * 60)

    if need_fit:
        print("")
        write_detection_catalog(
            csv_file=csv_file, output_file=catalog_output,
            min_snr=catalog_min_snr)

def positive_float(value):
    """argparse 类型：只接受正浮点数。"""
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number")
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract and fit ALMA spectra from Level 5 image cubes')
    parser.add_argument('target_name', nargs='?', help='process only this target')
    parser.add_argument('--position-id', help='process only this position_id')
    parser.add_argument(
        '--fit-range-ghz', type=positive_float, default=4.0, metavar='GHZ',
        help='total frequency window used for fitting (default: %(default)s GHz)')
    parser.add_argument(
        '--extraction-mode', choices=['aperture', 'point', 'both'],
        default='aperture',
        help='spectrum extraction method (default: %(default)s)')
    parser.add_argument(
        '--recenter', action='store_true',
        help='update CSV pixels from a local peak in the 2D MFS dirty image')
    parser.add_argument(
        '--recenter-radius-beams', type=positive_float, default=1.5,
        metavar='BEAMS',
        help='local peak search radius in synthesized beams (default: %(default)s)')
    parser.add_argument(
        '--recenter-min-snr', type=positive_float, default=5.0,
        metavar='SNR',
        help='minimum MFS dirty peak SNR required to update CSV (default: %(default)s)')
    parser.add_argument(
        '--catalog-min-snr', type=positive_float, default=3.0,
        metavar='SNR',
        help='strict integrated-SNR threshold for the catalog (default: > %(default)s)')
    parser.add_argument(
        '--catalog-output', default=DEFAULT_CATALOG_FILE, metavar='FILE',
        help='project-level emission-line catalog (default: %(default)s)')

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--extract-only', action='store_const',
                            dest='mode', const='extract',
                            help='extract spectra without fitting')
    mode_group.add_argument('--fit-only', action='store_const',
                            dest='mode', const='fit',
                            help='fit previously extracted spectra')
    mode_group.add_argument('--specfit-only', action='store_const',
                            dest='mode', const='full',
                            help='extract spectra and fit them')
    mode_group.add_argument('--catalog-only', action='store_const',
                            dest='mode', const='catalog',
                            help='rebuild the catalog from existing fit summaries')
    parser.set_defaults(mode='full')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments()

    if args.target_name is not None:
        print("Processing single target: {}".format(args.target_name))
    else:
        print("Processing all targets from CSV file")

    if args.position_id is not None:
        print("Filtering by position_id: {}".format(args.position_id))

    print("Processing mode: {}".format(args.mode))
    process_target(
        args.target_name, args.position_id, mode=args.mode,
        extraction_mode=args.extraction_mode,
        fit_range_ghz=args.fit_range_ghz,
        recenter=args.recenter,
        recenter_radius_beams=args.recenter_radius_beams,
        recenter_min_snr=args.recenter_min_snr,
        catalog_min_snr=args.catalog_min_snr,
        catalog_output=args.catalog_output)

    print("\nAll done!")
