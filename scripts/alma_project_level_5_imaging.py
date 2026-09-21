#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ALMA Project Level 5: Imaging with tclean
功能：对每个源目录下按频段拆分的 MS 文件分别进行 tclean 成像
"""

import os
import sys
import glob
import argparse
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
imsize_set = 512  # 默认图像大小

# 输出 cube 的目标速度宽度。tclean 仍按整数个原始频道合并，因此实际
# 速度宽度会取最接近该目标值的整数倍。
DEFAULT_TARGET_CHANNEL_WIDTH_KMS = 20.0
DEFAULT_AUTOMASK_PBMASK = 0.5

# 与 Level 4 保持一致：只有相邻 MS 的频率 gap 超过该值才拆组。
# Level 4/5 可通过同一个环境变量统一调整。
try:
    FREQUENCY_GAP_GHZ = float(os.environ.get('ALMA_FREQUENCY_GAP_GHZ', '100.0'))
except ValueError:
    FREQUENCY_GAP_GHZ = 100.0
    print("Warning: invalid ALMA_FREQUENCY_GAP_GHZ; using 100.0 GHz")
BAND_PREFIX = 'band'
# =================================================

C_KMS = 299792.458

def positive_float(value):
    """argparse 类型：只接受正浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a number")
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number

def pbmask_float(value):
    """argparse 类型：PB 响应阈值必须位于 (0, 1]。"""
    number = positive_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be less than or equal to 1")
    return number

def env_float(name, default, validator=positive_float):
    """读取浮点环境变量；无效时回退到默认值。"""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return validator(value)
    except argparse.ArgumentTypeError as exc:
        print("Warning: invalid {}={}: {}; using {}".format(
            name, value, exc, default))
        return default

def parse_arguments(argv=None):
    """解析 CASA 脚本参数。命令行值优先于环境变量。"""
    default_width = env_float(
        'ALMA_TARGET_CHANNEL_WIDTH_KMS',
        DEFAULT_TARGET_CHANNEL_WIDTH_KMS)
    default_pbmask = env_float(
        'ALMA_AUTOMASK_PBMASK',
        DEFAULT_AUTOMASK_PBMASK,
        validator=pbmask_float)

    parser = argparse.ArgumentParser(
        description='ALMA Level 5 cube imaging with configurable spectral binning')
    parser.add_argument(
        '--channel-width-kms',
        type=positive_float,
        default=default_width,
        metavar='KM/S',
        help=('target output channel width in km/s; the nearest integer '
              'number of native channels is used (default: %(default)s)'))

    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        '--mask', dest='mask_mode', action='store_const', const='fixed',
        help='use a fixed 2-beam ellipse centered on the MFS dirty peak (default)')
    mask_group.add_argument(
        '--auto-mask', dest='mask_mode', action='store_const', const='auto',
        help='use the previous per-channel auto-multithresh mask')
    mask_group.add_argument(
        '--no-mask', dest='mask_mode', action='store_const', const='none',
        help='disable masking and allow CLEAN over the full image')
    parser.set_defaults(mask_mode='fixed')

    parser.add_argument(
        '--pbmask',
        type=pbmask_float,
        default=default_pbmask,
        metavar='LEVEL',
        help=('primary-beam cutoff for auto-multithresh, in (0, 1] '
              '(default: %(default)s)'))

    dirty_mfs_group = parser.add_mutually_exclusive_group()
    dirty_mfs_group.add_argument(
        '--dirty-mfs', dest='make_dirty_mfs', action='store_true',
        help='create a fast all-channel 2D MFS dirty image (default)')
    dirty_mfs_group.add_argument(
        '--no-dirty-mfs', dest='make_dirty_mfs', action='store_false',
        help='do not create the 2D MFS dirty image used for recentering')
    parser.set_defaults(make_dirty_mfs=True)
    parser.add_argument(
        '--mask-radius-beams', type=positive_float, default=2.0,
        metavar='BEAMS',
        help=('semi-major/minor radius of the fixed ellipse in beam FWHM '
              '(default: %(default)s)'))
    parser.add_argument(
        '--mask-search-radius-beams', type=positive_float, default=2.0,
        metavar='BEAMS',
        help=('MFS peak search radius around image center in beam FWHM '
              '(default: %(default)s)'))
    return parser.parse_args(argv)

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

def get_ms_frequency_span_ghz(ms_file):
    """读取 MS 的实际频率覆盖范围（GHz）。"""
    try:
        tb.open(ms_file + os.sep + 'SPECTRAL_WINDOW')
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

def calculate_channel_binning(ms_file, target_width_kms):
    """计算最接近目标速度宽度的整数频道 bin 数。"""
    spw_table = ms_file + os.sep + 'SPECTRAL_WINDOW'
    spw_widths = []
    try:
        tb.open(spw_table)
        for row in range(tb.nrows()):
            channel_frequency = np.asarray(
                tb.getcell('CHAN_FREQ', row), dtype=float).ravel()
            channel_frequency = channel_frequency[np.isfinite(channel_frequency)]
            if channel_frequency.size < 2:
                continue

            frequency_steps = np.abs(np.diff(channel_frequency))
            frequency_steps = frequency_steps[
                np.isfinite(frequency_steps) & (frequency_steps > 0)]
            if frequency_steps.size == 0:
                continue

            reference_frequency = float(np.median(np.abs(channel_frequency)))
            if reference_frequency <= 0:
                continue

            channel_step_hz = float(np.median(frequency_steps))
            channel_width_kms = C_KMS * channel_step_hz / reference_frequency
            spw_widths.append(channel_width_kms)
            print("    SPW {} native channel spacing: {:.6f} MHz "
                  "({:.3f} km/s at {:.6f} GHz)".format(
                      row, channel_step_hz / 1e6, channel_width_kms,
                      reference_frequency / 1e9))
    finally:
        try:
            tb.close()
        except Exception:
            pass

    if not spw_widths:
        raise ValueError("could not determine channel spacing from CHAN_FREQ")

    native_width_kms = float(np.median(spw_widths))
    width_ratio = target_width_kms / native_width_kms
    # 显式使用 half-up，避免 Python round() 在 x.5 时采用偶数舍入。
    nbin = max(1, int(np.floor(width_ratio + 0.5)))
    actual_width_kms = nbin * native_width_kms

    if len(spw_widths) > 1:
        spread = max(spw_widths) - min(spw_widths)
        if spread > 0.01 * native_width_kms:
            print("    Warning: native channel widths differ across SPWs "
                  "({:.3f}-{:.3f} km/s); using their median.".format(
                      min(spw_widths), max(spw_widths)))

    print("    Target channel width: {:.3f} km/s".format(target_width_kms))
    print("    Spectral binning: {} native channel(s) per output channel".format(nbin))
    print("    Approximate output channel width: {:.3f} km/s".format(
        actual_width_kms))
    return nbin

def find_band_ms_files(target_dir, target):
    """查找每个频段的输入 MS。

    优先使用 Level 4 生成的 band MS；若尚未运行 Level 4，则直接使用
    DataSet_*.ms（本项目每个频段只有一个文件）。最后才回退到旧的混合 MS。
    """
    result = {}
    for band_ms in sorted(glob.glob(os.path.join(target_dir, '{}_{}_*.ms'.format(target, BAND_PREFIX)))):
        suffix = os.path.splitext(os.path.basename(band_ms))[0]
        prefix = '{}_'.format(target)
        if suffix.startswith(prefix):
            result[suffix[len(prefix):]] = band_ms

    raw_ms_files = sorted(glob.glob(os.path.join(target_dir, 'DataSet_*.ms')))
    if raw_ms_files:
        spans = []
        for ms_file in raw_ms_files:
            freq_min, freq_max = get_ms_frequency_span_ghz(ms_file)
            spans.append((freq_min, freq_max, ms_file))
        spans.sort(key=lambda item: item[0])
        groups = []
        for freq_min, freq_max, ms_file in spans:
            if not groups or freq_min - groups[-1]['max_ghz'] > FREQUENCY_GAP_GHZ:
                groups.append({'min_ghz': freq_min, 'max_ghz': freq_max, 'files': []})
            groups[-1]['files'].append(ms_file)
            groups[-1]['max_ghz'] = max(groups[-1]['max_ghz'], freq_max)

        for index, group in enumerate(groups, start=1):
            band = '{}_{:02d}'.format(BAND_PREFIX, index)
            if band in result:
                continue
            if len(group['files']) == 1:
                result[band] = group['files'][0]
            else:
                print("  WARNING: {} has {} raw MS files in {}. "
                      "Run Level 4 first to concat this group.".format(
                          target, len(group['files']), band))

    if result:
        return result

    # 兼容旧项目：只有在没有任何 band/raw MS 时才使用可能包含混合频段的旧文件。
    legacy_ms = os.path.join(target_dir, '{}.ms'.format(target))
    if os.path.exists(legacy_ms):
        return {'legacy': legacy_ms}
    return {}

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

def create_dirty_mfs(msfile, image_path, imcell, field_='', imsize_=imsize_set):
    """创建包含 continuum + line 的快速二维 MFS dirty 图。"""
    dirty_mfs = image_path + '_mfs_dirty'
    dirty_mfs_image = dirty_mfs + '.image'
    if os.path.exists(dirty_mfs_image):
        print("  MFS dirty image already exists: {}".format(
            os.path.basename(dirty_mfs_image)))
        return True

    print("\n  Creating all-channel 2D MFS dirty image...")
    try:
        tclean(
            vis=msfile,
            imagename=dirty_mfs,
            field=field_,
            datacolumn='corrected',
            specmode='mfs',
            imsize=imsize_,
            cell=imcell,
            deconvolver='hogbom',
            weighting='natural',
            niter=0
        )
    except Exception as e:
        print("Error during 2D MFS dirty imaging: {}".format(str(e)))
        return False

    if not os.path.exists(dirty_mfs_image):
        print("Error: 2D MFS dirty image was not created")
        return False
    print("  Created: {}".format(os.path.basename(dirty_mfs_image)))
    return True


def angle_to_arcsec(quantity):
    """将 CASA quantity 字典转换为 arcsec。"""
    value = float(quantity['value'])
    unit = str(quantity.get('unit', 'arcsec')).lower()
    factors = {
        'arcsec': 1.0,
        'arcmin': 60.0,
        'deg': 3600.0,
        'rad': 206264.806247,
        'mas': 0.001,
    }
    if unit not in factors:
        raise ValueError("Unsupported angular unit: {}".format(unit))
    return value * factors[unit]


def angle_to_degrees(quantity):
    """将 CASA quantity 字典转换为 degrees。"""
    value = float(quantity['value'])
    unit = str(quantity.get('unit', 'deg')).lower()
    factors = {
        'deg': 1.0,
        'rad': 180.0 / np.pi,
        'arcmin': 1.0 / 60.0,
        'arcsec': 1.0 / 3600.0,
    }
    if unit not in factors:
        raise ValueError("Unsupported position-angle unit: {}".format(unit))
    return value * factors[unit]


def build_fixed_ellipse_mask(image_path, radius_beams=2.0,
                             search_radius_beams=2.0, min_peak_snr=5.0):
    """由二维 MFS dirty 图构造跨所有 channel 的固定椭圆 region。"""
    dirty_mfs_image = image_path + '_mfs_dirty.image'
    if not os.path.exists(dirty_mfs_image):
        raise IOError("MFS dirty image not found: {}".format(dirty_mfs_image))

    ia.open(dirty_mfs_image)
    try:
        shape = ia.shape()
        restoring_beam = ia.restoringbeam()
        coordsys = ia.coordsys()
        increments = np.asarray(coordsys.increment()['numeric'], dtype=float)
        try:
            coordsys.done()
        except Exception:
            pass
    finally:
        ia.close()

    if len(shape) < 2 or len(increments) < 2:
        raise ValueError("Invalid MFS image dimensions")
    beam_major_arcsec = angle_to_arcsec(restoring_beam['major'])
    beam_minor_arcsec = angle_to_arcsec(restoring_beam['minor'])
    beam_pa_deg = angle_to_degrees(restoring_beam['positionangle'])
    cell_x_arcsec = abs(float(increments[0])) * 206264.806247
    cell_y_arcsec = abs(float(increments[1])) * 206264.806247
    if cell_x_arcsec <= 0 or cell_y_arcsec <= 0:
        raise ValueError("Invalid MFS image cell size")

    center_x = (float(shape[0]) - 1.0) / 2.0
    center_y = (float(shape[1]) - 1.0) / 2.0
    search_x_pix = search_radius_beams * beam_major_arcsec / cell_x_arcsec
    search_y_pix = search_radius_beams * beam_major_arcsec / cell_y_arcsec
    x0 = max(0, int(np.floor(center_x - search_x_pix)))
    x1 = min(int(shape[0]) - 1, int(np.ceil(center_x + search_x_pix)))
    y0 = max(0, int(np.floor(center_y - search_y_pix)))
    y1 = min(int(shape[1]) - 1, int(np.ceil(center_y + search_y_pix)))

    local_stats = imstat(
        imagename=dirty_mfs_image,
        box='{},{},{},{}'.format(x0, y0, x1, y1))
    full_stats = imstat(imagename=dirty_mfs_image)
    if len(local_stats.get('maxpos', [])) < 2:
        raise ValueError("Could not determine an MFS dirty peak position")

    peak_x = int(local_stats['maxpos'][0])
    peak_y = int(local_stats['maxpos'][1])
    peak_value = float(local_stats['max'][0])
    image_rms = float(full_stats['rms'][0])
    peak_snr = peak_value / image_rms if image_rms > 0 else np.nan
    if not np.isfinite(peak_snr) or peak_snr < min_peak_snr:
        print("    MFS local peak SNR={:.2f} is below {:.1f}; "
              "using image center".format(peak_snr, min_peak_snr))
        peak_x = int(round(center_x))
        peak_y = int(round(center_y))
    else:
        print("    MFS local peak: ({}, {}), SNR={:.2f}".format(
            peak_x, peak_y, peak_snr))

    major_radius_arcsec = radius_beams * beam_major_arcsec
    minor_radius_arcsec = radius_beams * beam_minor_arcsec
    mask_region = (
        'ellipse[[{:.3f}pix,{:.3f}pix],'
        '[{:.6f}arcsec,{:.6f}arcsec],{:.6f}deg]'.format(
            float(peak_x), float(peak_y),
            major_radius_arcsec, minor_radius_arcsec, beam_pa_deg))
    print("    Fixed mask: {}".format(mask_region))
    print("    Mask semi-axes: {:.2f} x {:.2f} arcsec "
          "({:.1f} x beam FWHM)".format(
              major_radius_arcsec, minor_radius_arcsec, radius_beams))
    return mask_region


def tclean_for_msfile(msfile, image_path, target_width_kms,
                      mask_mode='fixed', automask_pbmask=DEFAULT_AUTOMASK_PBMASK,
                      mask_radius_beams=2.0, mask_search_radius_beams=2.0,
                      make_dirty_mfs=True, field_='', imsize_=imsize_set):
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
        channel_bin = calculate_channel_binning(msfile, target_width_kms)
    except Exception as e:
        print("Error calculating imaging parameters: {}".format(str(e)))
        return False


    if make_dirty_mfs and not create_dirty_mfs(
            msfile, image_path, imcell, field_=field_, imsize_=imsize_):
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
            width=channel_bin,
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

    if mask_mode == 'fixed':
        try:
            fixed_mask = build_fixed_ellipse_mask(
                image_path,
                radius_beams=mask_radius_beams,
                search_radius_beams=mask_search_radius_beams)
        except Exception as e:
            print("Error creating fixed MFS ellipse mask: {}".format(str(e)))
            return False
        mask_parameters = {'usemask': 'user', 'mask': fixed_mask}
        print("    Masking: fixed MFS-centered ellipse, same for all channels")
    elif mask_mode == 'auto':
        mask_parameters = {
            'usemask': 'auto-multithresh',
            'pbmask': automask_pbmask,
            'sidelobethreshold': 3.0,
            'noisethreshold': 5.0,
            'lownoisethreshold': 1.5,
            'minbeamfrac': 0.3,
            'growiterations': 75,
            'dogrowprune': True,
        }
        print("    Masking: auto-multithresh (pbmask={:.3f})".format(
            automask_pbmask))
    elif mask_mode == 'none':
        mask_parameters = {'usemask': 'user', 'mask': ''}
        print("    Masking: disabled (CLEAN may place components anywhere in the image)")
    else:
        print("Error: unknown mask mode '{}'".format(mask_mode))
        return False
    
    try:
        tclean(
            vis=msfile,
            imagename=clean_image,
            field=field_,
            datacolumn='corrected',
            specmode='cube',
            width=channel_bin,
            start=1,
            nchan=-1,
            veltype='radio',
            imsize=imsize_,
            cell=imcell,
            deconvolver='hogbom',
            weighting='natural',
            #restoringbeam='common',
            threshold=2 * threshold_clean,
            niter=300000,
            **mask_parameters
        )
        print("\n  Imaging completed!")
        return True
    except Exception as e:
        print("Error during clean: {}".format(str(e)))
        return False

def main():
    """主函数"""
    args = parse_arguments()
    print("="*60)
    print("ALMA Project Level 5: Imaging with tclean")
    print("="*60)
    print("Working directory: {}".format(base_dir))
    print("Target channel width: {:.3f} km/s".format(
        args.channel_width_kms))
    if args.mask_mode == 'fixed':
        print("Masking: fixed MFS-centered ellipse "
              "({:.1f} beam semi-axes)".format(args.mask_radius_beams))
    elif args.mask_mode == 'auto':
        print("Masking: auto-multithresh (pbmask={:.3f})".format(args.pbmask))
    else:
        print("Masking: disabled")
    print("2D MFS dirty image: {}".format(
        'enabled' if args.make_dirty_mfs else 'disabled'))
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
        
        # 查找动态频率组的 MS；不会把 gap 过大的组一起送进 tclean
        band_ms_files = find_band_ms_files(target_dir, target)

        if not band_ms_files:
            print("Warning: No band MS files found. Skipping.")
            skipped_count += 1
            continue
        
        # 创建输出目录
        output_dir = os.path.join(target_dir, OUTPUT_SUBDIR)
        os.makedirs(output_dir, exist_ok=True)
        
        # 对每个频率组分别成像；legacy 保持旧的 <target>.image 命名
        for band, ms_file in sorted(band_ms_files.items()):
            if band == 'legacy':
                image_stem = target
            else:
                image_stem = '{}_{}'.format(target, band)
            image_path = os.path.join(output_dir, image_stem)

            final_image = image_path + '.image'
            if os.path.exists(final_image):
                dirty_mfs_image = image_path + '_mfs_dirty.image'
                if args.make_dirty_mfs and not os.path.exists(dirty_mfs_image):
                    try:
                        imcell = calculate_cell(ms_file)
                        if not create_dirty_mfs(
                                ms_file, image_path, imcell,
                                field_='', imsize_=imsize_set):
                            failed_count += 1
                    except Exception as e:
                        print("Error creating missing MFS dirty image: {}".format(
                            str(e)))
                        failed_count += 1
                print("Output image '{}' already exists. Skipping.".format(
                    os.path.basename(final_image)))
                skipped_count += 1
                continue

            if tclean_for_msfile(
                    ms_file, image_path,
                    target_width_kms=args.channel_width_kms,
                    mask_mode=args.mask_mode,
                    automask_pbmask=args.pbmask,
                    mask_radius_beams=args.mask_radius_beams,
                    mask_search_radius_beams=args.mask_search_radius_beams,
                    make_dirty_mfs=args.make_dirty_mfs,
                    field_='', imsize_=imsize_set):
                success_count += 1
            else:
                failed_count += 1
    
    # 输出总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Total targets: {}".format(len(targets)))
    print("Successful images: {}".format(success_count))
    print("Failed images: {}".format(failed_count))
    print("Skipped images: {}".format(skipped_count))
    print("="*60)
    
    if failed_count > 0:
        sys.exit(1)
    else:
        print("\nAll done!")

if __name__ == '__main__':
    main()
