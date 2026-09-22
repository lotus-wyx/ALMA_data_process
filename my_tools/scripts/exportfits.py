#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Phase 1: FITS Export

批量检查并导出FITS文件（需要CASA环境）
"""

import os
import csv
import json

def infer_band(target_row, name=None, work_dir='.'):
    """从 CSV 或 MS 分组 manifest 推断动态频率组。"""
    band = (target_row.get('band') or '').strip().lower()
    if band:
        return band

    if name:
        manifest = os.path.join(
            work_dir, 'Each_target_img', name, '{}_ms_groups.json'.format(name))
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
    return None

def resolve_image_file(name, target_row, work_dir='.'):
    """优先找到按频段命名的新图像，找不到时回退到旧图像。"""
    band = infer_band(target_row, name, work_dir)
    cube_dir = os.path.join(work_dir, 'Each_target_img', name, 'cubes')
    candidates = []
    if band:
        candidates.append(os.path.join(
            cube_dir, '{}_{}.image'.format(
                name, band if band.startswith('band_') else 'band_{}'.format(band))))
    candidates.append(os.path.join(cube_dir, '{}.image'.format(name)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # 返回新路径以便错误信息明确指出期望的 band 图像
    return candidates[0] if candidates else candidates[-1]


def load_target_list(csv_file):
    """从CSV文件读取目标源列表"""
    targets = []
    if not os.path.exists(csv_file):
        print("Error: CSV file '{}' not found.".format(csv_file))
        return targets

    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        targets = list(reader)

    print("Loaded {} targets from {}".format(len(targets), csv_file))
    return targets


def check_and_export_fits(image_file):
    """检查FITS文件是否存在，如果不存在则从.image导出"""
    fits_file = image_file + '.fits'

    if os.path.exists(fits_file):
        print("  FITS already exists: {}".format(fits_file))
        return 'skip'

    if not os.path.exists(image_file):
        print("  Error: Image file not found: {}".format(image_file))
        return 'fail'

    print("  Exporting FITS from image...")
    try:
        exportfits(imagename=image_file, fitsimage=fits_file, overwrite=True)
        print("  Successfully exported: {}".format(fits_file))
        return 'success'
    except Exception as e:
        print("  Error exporting FITS: {}".format(str(e)))
        return 'fail'


def resolve_mfs_dirty_image(image_file):
    """由 clean cube 路径得到同一频率组的二维 MFS dirty 图路径。"""
    suffix = '.image'
    if not image_file.endswith(suffix):
        return image_file + '_mfs_dirty.image'
    return image_file[:-len(suffix)] + '_mfs_dirty.image'


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 1: FITS Export (CASA Environment)")
    print("=" * 60)

    csv_file = 'target_line_list.csv'
    if not os.path.exists(csv_file):
        fallback_csv = 'target_line_list_radec.csv'
        if os.path.exists(fallback_csv):
            csv_file = fallback_csv
            print("target_line_list.csv not found, fallback to {}".format(fallback_csv))

    targets = load_target_list(csv_file)

    if len(targets) == 0:
        print("No targets found. Exiting.")
        import sys
        sys.exit(1)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for tgt in targets:
        name = tgt['name']
        image_file = resolve_image_file(name, tgt)

        print("\nTarget: {}".format(name))
        band = infer_band(tgt, name)
        if band:
            print("  Band: {}".format(band))
        result = check_and_export_fits(image_file)

        mfs_dirty_image = resolve_mfs_dirty_image(image_file)
        if os.path.exists(mfs_dirty_image):
            check_and_export_fits(mfs_dirty_image)
        else:
            print("  MFS dirty image not found (recenter unavailable): {}".format(
                mfs_dirty_image))

        if result == 'success':
            success_count += 1
        elif result == 'skip':
            skip_count += 1
        elif result == 'fail':
            fail_count += 1

    print("\n" + "=" * 60)
    print("Phase 1 Summary:")
    print("  Newly exported: {}".format(success_count))
    print("  Already exists: {}".format(skip_count))
    print("  Failed: {}".format(fail_count))
    print("  Total: {}".format(len(targets)))
    print("=" * 60)

    if fail_count > 0:
        print("\nWarning: {} target(s) failed to export".format(fail_count))

    print("\nPhase 1 completed.")
