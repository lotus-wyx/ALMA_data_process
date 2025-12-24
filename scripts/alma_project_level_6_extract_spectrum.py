#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Phase 2: Spectral Extraction and Gaussian Fitting

从FITS cube中提取目标源光谱并进行高斯拟合（纯Python，不需要CASA）
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


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
def extract_spectrum(fits_file, pixel_x, pixel_y, aperture_factor, output_dir):
    """从FITS cube中提取光谱（优化版：只读取需要的区域）"""
    print("Extracting spectrum from: {}".format(fits_file))
    
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
        
        # 只读取需要的子区域
        data_sub = hdul[0].data[0, :, y_min:y_max, x_min:x_max].astype(np.float32)
        
        # 调整像素坐标到子区域
        pixel_x_sub = pixel_x - x_min
        pixel_y_sub = pixel_y - y_min
        
        # Beam 归一化（一次性处理所有通道）
        if beam_per_channel is None:
            data_sub = data_sub / beam_area_pixel
        else:
            for i in range(nchan):
                data_sub[i] = data_sub[i] / beam_per_channel[i]
        
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
                'aperture_pix': aperture_pix
            }
    
    return results

def gaussian_plus_const(freq, amplitude, center, sigma, constant):
    """高斯线 + 常数背景模型"""
    return amplitude * np.exp(-0.5 * ((freq - center) / sigma)**2) + constant


def fit_gaussian(freq, flux, error, line_freq_prior, fit_range_ghz=4.0):
    """对光谱进行高斯拟合"""
    half_range = fit_range_ghz / 2.0
    
    freq_min = np.min(freq)
    freq_max = np.max(freq)
    
    fit_freq_min = line_freq_prior - half_range
    fit_freq_max = line_freq_prior + half_range
    
    if fit_freq_min < freq_min:
        deficit = freq_min - fit_freq_min
        fit_freq_min = freq_min
        fit_freq_max = min(line_freq_prior + half_range + deficit, freq_max)
    
    if fit_freq_max > freq_max:
        deficit = fit_freq_max - freq_max
        fit_freq_max = freq_max
        fit_freq_min = max(line_freq_prior - half_range - deficit, freq_min)
    
    mask = (freq >= fit_freq_min) & (freq <= fit_freq_max)
    freq_fit = freq[mask]
    flux_fit = flux[mask]
    error_fit = error[mask]
    
    # 过滤掉NaN/Inf和error<=0的数据点
    valid_mask = np.isfinite(flux_fit) & (flux_fit!=0) & np.isfinite(error_fit) & (error_fit > 0)
    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print("Filtered {} invalid data points (NaN/Inf/zero-error)".format(n_invalid))
    
    freq_fit = freq_fit[valid_mask]
    flux_fit = flux_fit[valid_mask]
    error_fit = error_fit[valid_mask]
    
    if len(freq_fit) < 5:
        print("    Warning: Not enough valid data points in fit range ({})".format(len(freq_fit)))
        return None, None, np.nan, np.nan, np.min(freq_fit), np.max(freq_fit)
    
    flux_range = np.max(flux_fit) - np.min(flux_fit)
    p0 = [flux_range, line_freq_prior, 0.127, np.median(flux_fit)]
    
    bounds = ([0, np.min(freq_fit), 0.05, -np.inf], [np.inf, np.max(freq_fit), 0.5, np.inf])
    
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
        
        return popt, perr, chi2_dof, snr, np.min(freq_fit), np.max(freq_fit)
        
    except Exception as e:
        print("    Fit failed: {}".format(str(e)))
        return None, None, np.nan, np.nan, np.min(freq_fit), np.max(freq_fit)


def plot_spectrum_with_fit(freq, flux, error, fit_params, aperture_factor, 
                        output_file, fit_freq_min=None, fit_freq_max=None):
    """绘制光谱和拟合结果（只显示拟合范围）"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), 
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
    
    # 使用 sigma-clipping 排除异常值，确定y轴范围
    valid_mask = np.isfinite(flux_plot)
    if np.sum(valid_mask) > 0:
        from astropy.stats import sigma_clipped_stats
        mean, median, std = sigma_clipped_stats(flux_plot[valid_mask], sigma=3.0)
        
        # y轴范围：median ± 5*std（可以捕获大部分有效数据）
        y_min = median - 5 * std
        y_max = median + 5 * std
        
        # 如果有拟合结果，确保拟合峰值在范围内
        if fit_params is not None:
            peak_flux = fit_params[0] + fit_params[3]  # amplitude + constant
            y_max = max(y_max, peak_flux * 1.2)
    else:
        y_min, y_max = None, None
    
    ax1.errorbar(freq_plot, flux_plot, yerr=error_plot, fmt='o', markersize=3, 
                 alpha=0.6, label='Data', color='black')
    
    if fit_params is not None:
        freq_model = np.linspace(np.min(freq_plot), np.max(freq_plot), 500)
        flux_model = gaussian_plus_const(freq_model, *fit_params)
        ax1.plot(freq_model, flux_model, 'r-', linewidth=2, label='Gaussian fit')
        ax1.axvline(fit_params[1], color='blue', linestyle='--', 
                   alpha=0.5, label='Fitted center')
    
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Flux (Jy)', fontsize=12)
    ax1.set_title('Aperture: {}x beam_maj/2'.format(aperture_factor), fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 设置x轴范围为拟合区域
    if fit_freq_min is not None and fit_freq_max is not None:
        ax1.set_xlim(fit_freq_min, fit_freq_max)
        ax2.set_xlim(fit_freq_min, fit_freq_max)
    
    # 设置y轴范围，排除异常值
    if y_min is not None and y_max is not None:
        ax1.set_ylim(y_min, y_max)
    
    if fit_params is not None:
        model = gaussian_plus_const(freq_plot, *fit_params)
        residuals = flux_plot - model
        ax2.errorbar(freq_plot, residuals, yerr=error_plot, fmt='o', markersize=3, 
                    alpha=0.6, color='black')
        ax2.axhline(0, color='red', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Residuals (Jy)', fontsize=12)
        
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

def process_target(target, csv_file='target_line_list.csv'):
    """处理单个或所有目标源"""
    targets = load_target_list(csv_file)
    
    if len(targets) == 0:
        print("No targets found in CSV file.")
        return
    
    if target is not None:
        targets = [t for t in targets if t['name'] == target]
        if len(targets) == 0:
            print("Target '{}' not found in CSV file.".format(target))
            return
    
    print("\n" + "=" * 60)
    print("Phase 2: Extracting spectra and fitting")
    print("Processing {} target(s)".format(len(targets)))
    print("=" * 60)
    
    for tgt in targets:
        name = tgt['name']
        pixel_x = float(tgt['pixel_x'])
        pixel_y = float(tgt['pixel_y'])
        line_freq = float(tgt['line_freq_GHz'])

        print("\n" + "=" * 60)
        print("Target: {}".format(name))
        print("Position: pixel ({:.2f}, {:.2f})".format(pixel_x, pixel_y))
        print("Line frequency prior: {:.2f} GHz".format(line_freq))
        print("=" * 60)
        
        image_file = 'Each_target_img/{}/cubes/{}.image'.format(name, name)
        fits_file = image_file + '.fits'
        
        if not os.path.exists(fits_file):
            print("Error: FITS file not found: {}".format(fits_file))
            print("Please run Phase 1 first: ./alma_project_level_6_extract_spectrum.sh --export-only")
            print("Skipping target: {}".format(name))
            continue
        
        print("FITS file exists, processing...")
        
        output_dir = 'Each_target_img/{}/cubes'.format(name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            spectra = extract_spectrum(
                fits_file, pixel_x, pixel_y, 
                [0.5, 1.0, 1.5, 2.0], output_dir
            )
        except Exception as e:
            print("Error extracting spectrum: {}".format(str(e)))
            import traceback
            traceback.print_exc()
            continue
        
        fit_summary = []
        
        for aper_factor in [0.5, 1.0, 1.5, 2.0]:
            print("\nProcessing aperture: {}x".format(aper_factor))
            
            spec_data = spectra[aper_factor]
            
            spec_file = os.path.join(
                output_dir, 
                '{}_{}_no_Apcor.result.txt'.format(name, aper_factor)
            )
            
            with open(spec_file, 'w') as f:
                f.write("# Target: {}\n".format(name))
                f.write("# Aperture: {:.1f} x beam_maj/2\n".format(aper_factor))
                f.write("# Aperture radius: {:.2f} pixels\n".format(spec_data['aperture_pix']))
                f.write("# Line frequency prior: {:.4f} GHz\n".format(line_freq))
                f.write("#\n")
                f.write("# Frequency(GHz)\tFlux(Jy)\tError(Jy)\tBackground(Jy)\n")
                
                for freq, flux, err, bkg in zip(spec_data['freq'], spec_data['flux'], 
                                                 spec_data['error'], spec_data['bkg']):
                    f.write("{:.6f}\t{:.6e}\t{:.6e}\t{:.6e}\n".format(freq, flux, err, bkg))
            
            print("  Saved spectrum: {}".format(spec_file))
            
            print("  Fitting Gaussian...")
            fit_params, fit_errors, chi2_dof, snr, fit_freq_min, fit_freq_max = fit_gaussian(
                spec_data['freq'], spec_data['flux'], spec_data['error'], 
                line_freq, fit_range_ghz=8.0
            )
            
            plot_file = os.path.join(
                output_dir,
                '{}_{}_specfit.pdf'.format(name, aper_factor)
            )
            
            plot_spectrum_with_fit(
                spec_data['freq'], spec_data['flux'], spec_data['error'],
                fit_params, aper_factor, plot_file,
                fit_freq_min, fit_freq_max
            )
            
            if fit_params is not None:
                fit_summary.append({
                    'aperture': aper_factor,
                    'amplitude': fit_params[0],
                    'amplitude_err': fit_errors[0],
                    'center': fit_params[1],
                    'center_err': fit_errors[1],
                    'sigma': fit_params[2],
                    'sigma_err': fit_errors[2],
                    'constant': fit_params[3],
                    'constant_err': fit_errors[3],
                    'chi2_dof': chi2_dof,
                    'snr': snr
                })
            else:
                fit_summary.append({
                    'aperture': aper_factor,
                    'amplitude': np.nan,
                    'amplitude_err': np.nan,
                    'center': np.nan,
                    'center_err': np.nan,
                    'sigma': np.nan,
                    'sigma_err': np.nan,
                    'constant': np.nan,
                    'constant_err': np.nan,
                    'chi2_dof': np.nan,
                    'snr': np.nan
                })
        
        summary_file = os.path.join(
            output_dir,
            '{}_gaussian_fit_summary.txt'.format(name)
        )
        
        with open(summary_file, 'w') as f:
            f.write("# Gaussian fit results for {}\n".format(name))
            f.write("# Model: flux = A * exp(-0.5*((freq-f0)/sigma)^2) + C\n")
            f.write("# Line frequency prior: {:.4f} GHz\n".format(line_freq))
            f.write("# Sigma prior: 0.127 GHz (FWHM = 0.3 GHz)\n")
            f.write("#\n")
            f.write("# Aperture\tAmplitude(Jy)\tCenter(GHz)\tSigma(GHz)\tConstant(Jy)\tChi2/dof\tSNR\n")
            
            for fit in fit_summary:
                f.write("{:.1f}x\t{:.4e}±{:.4e}\t{:.6f}±{:.6f}\t{:.4f}±{:.4f}\t{:.4e}±{:.4e}\t{:.2f}\t{:.1f}\n".format(
                    fit['aperture'],
                    fit['amplitude'], fit['amplitude_err'],
                    fit['center'], fit['center_err'],
                    fit['sigma'], fit['sigma_err'],
                    fit['constant'], fit['constant_err'],
                    fit['chi2_dof'],
                    fit['snr']
                ))
        
        print("\nSaved fit summary: {}".format(summary_file))
        print("=" * 60)
        print("Completed: {}".format(name))
        print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_name = sys.argv[1]
        print("Processing single target: {}".format(target_name))
        process_target(target_name)
    else:
        print("Processing all targets from CSV file")
        process_target(None)
    
    print("\nAll done!")
