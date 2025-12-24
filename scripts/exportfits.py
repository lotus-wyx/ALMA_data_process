#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Phase 1: FITS Export

批量检查并导出FITS文件（需要CASA环境）
"""

import os
import csv


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


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Phase 1: FITS Export (CASA Environment)")
    print("=" * 60)
    
    csv_file = 'target_line_list.csv'
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
        image_file = 'Each_target_img/{}/cubes/{}.image'.format(name, name)
        
        print("\nTarget: {}".format(name))
        result = check_and_export_fits(image_file)
        
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
