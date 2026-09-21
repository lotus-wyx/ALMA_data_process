#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALMA Project Level 6 - Prepare Step
将 CSV 中的 RA/Dec 转为 Level 6 所需 pixel_x/pixel_y。

转换方式参考历史脚本：
with fits.open(fits_f) as hdul:
    ref_ra = header['CRVAL1']
    ref_dec = header['CRVAL2']
    delta_ra = header['CDELT1']
    delta_dec = header['CDELT2']
    ra_values = ref_ra + (np.arange(nx) - header['CRPIX1']) * delta_ra
    dec_values = ref_dec + (np.arange(ny) - header['CRPIX2']) * delta_dec
    ra_index = np.abs(ra_values - target_ra).argmin()
    dec_index = np.abs(dec_values - target_dec).argmin()
"""

import os
import sys
import csv
import argparse
import json

import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u

def infer_band(row, name=None, work_dir='.'):
    """从 CSV 或 MS 分组 manifest 推断动态频率组。"""
    band = (row.get("band") or "").strip().lower()
    if band:
        return band

    if name:
        manifest = os.path.join(
            work_dir, "Each_target_img", name, "{}_ms_groups.json".format(name))
        if os.path.exists(manifest):
            try:
                with open(manifest) as handle:
                    groups = json.load(handle)
                frequency_ghz = float(row.get("line_freq_GHz", ""))
                containing = [group for group in groups
                              if group["min_freq_GHz"] <= frequency_ghz <= group["max_freq_GHz"]]
                if containing:
                    return containing[0]["group"]
                if groups:
                    return min(groups, key=lambda group: abs(
                        group["center_freq_GHz"] - frequency_ghz))["group"]
            except (TypeError, ValueError, IOError, KeyError):
                pass

    # 没有 manifest 且 CSV 未指定 band 时，不按绝对频率强行拆分。
    return None


def parse_ra_deg(value):
    s = str(value).strip()
    if not s:
        raise ValueError("empty RA value")

    s_lower = s.lower()
    if (":" in s) or ("h" in s_lower and "m" in s_lower):
        return float(Angle(s, unit=u.hourangle).degree)

    return float(s)


def parse_dec_deg(value):
    s = str(value).strip()
    if not s:
        raise ValueError("empty Dec value")

    s_lower = s.lower()
    if (":" in s) or ("d" in s_lower) or ("m" in s_lower) or ("s" in s_lower):
        return float(Angle(s, unit=u.deg).degree)

    return float(s)


def radec_to_pixel_index(fits_file, target_ra_deg, target_dec_deg):
    """按最近像素索引法将 RA/Dec 转为 0-based 像素坐标。"""
    with fits.open(fits_file, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        for key in ["CRVAL1", "CDELT1", "CRPIX1", "CRVAL2", "CDELT2", "CRPIX2"]:
            if key not in header:
                raise KeyError("{} not found in header: {}".format(key, fits_file))

        ref_ra = float(header["CRVAL1"])
        ref_dec = float(header["CRVAL2"])
        delta_ra = float(header["CDELT1"])
        delta_dec = float(header["CDELT2"])
        crpix1 = float(header["CRPIX1"])
        crpix2 = float(header["CRPIX2"])

        if data is None or data.ndim < 2:
            raise ValueError("Invalid FITS data shape in {}".format(fits_file))

        # 尽量贴近历史脚本的数据维度处理方式
        if data.ndim >= 4:
            nx = int(data.shape[2])
            ny = int(data.shape[3])
        else:
            nx = int(header.get("NAXIS1", data.shape[-1]))
            ny = int(header.get("NAXIS2", data.shape[-2]))

        ra_values = ref_ra + (np.arange(nx) - crpix1) * delta_ra
        dec_values = ref_dec + (np.arange(ny) - crpix2) * delta_dec

        ra_index = int(np.abs(ra_values - target_ra_deg).argmin())
        dec_index = int(np.abs(dec_values - target_dec_deg).argmin())

        return ra_index, dec_index


def build_output_fields(include_band=False):
    """输出为 Level 6 下游固定需要的字段。"""
    fields = ["name", "pixel_x", "pixel_y", "line_freq_GHz", "position_id"]
    if include_band:
        fields.append("band")
    return fields


def main():
    parser = argparse.ArgumentParser(description="Convert RA/Dec CSV to pixel_x/pixel_y for Level 6")
    parser.add_argument("--input-csv", default="target_line_list_radec.csv", help="Input CSV with name,ra,dec")
    parser.add_argument("--output-csv", default="target_line_list.csv", help="Output CSV for Level 6")
    parser.add_argument("--work-dir", default=".", help="Working directory")
    parser.add_argument(
        "--image-template",
        default="Each_target_img/{name}/cubes/{name}_{band}.image.fits",
        help="Image FITS path template, supports {name}, {position_id}, and {band}"
    )
    parser.add_argument("--target-name", default=None, help="Process only one target name")
    parser.add_argument("--position-id", default=None, help="Process only one position_id")
    parser.add_argument("--strict", action="store_true", help="Abort on first row error")

    args = parser.parse_args()

    input_csv = os.path.abspath(args.input_csv)
    output_csv = os.path.abspath(args.output_csv)
    work_dir = os.path.abspath(args.work_dir)

    if not os.path.exists(input_csv):
        print("Error: input CSV not found: {}".format(input_csv))
        sys.exit(1)

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        print("Error: input CSV header is empty")
        sys.exit(1)

    for key in ["name", "ra", "dec", "line_freq_GHz"]:
        if key not in fieldnames:
            print("Error: missing required column '{}' in input CSV".format(key))
            sys.exit(1)

    output_rows = []
    ok_count = 0
    err_count = 0

    print("=" * 50)
    print("Level6 Prepare: RA/Dec -> pixel_x/pixel_y")
    print("Input : {}".format(input_csv))
    print("Output: {}".format(output_csv))
    print("=" * 50)

    for row_id, row in enumerate(rows, start=2):
        try:
            name = (row.get("name") or "").strip()
            if not name:
                continue

            position_id = (row.get("position_id") or "default").strip() or "default"

            if args.target_name is not None and name != args.target_name:
                continue
            if args.position_id is not None and position_id != args.position_id:
                continue

            target_ra = parse_ra_deg(row.get("ra", ""))
            target_dec = parse_dec_deg(row.get("dec", ""))

            band = infer_band(row, name, work_dir)
            format_values = {
                "name": name,
                "position_id": position_id,
                "band": band or "",
            }
            rel_path = args.image_template.format(**format_values)
            fits_file = rel_path if os.path.isabs(rel_path) else os.path.join(work_dir, rel_path)

            # 兼容旧的 <target>.image.fits；也兼容用户传入不含 {band} 的模板。
            if not os.path.exists(fits_file):
                legacy_file = os.path.join(
                    work_dir, "Each_target_img", name, "cubes", "{}.image.fits".format(name))
                if os.path.exists(legacy_file):
                    fits_file = legacy_file

            if not os.path.exists(fits_file):
                if fits_file.endswith(".fits"):
                    casa_image = fits_file[:-5]
                    if os.path.exists(casa_image):
                        raise FileNotFoundError(
                            "FITS not found: {} ; CASA image exists: {} ; please run exportfits first".format(
                                fits_file, casa_image
                            )
                        )
                raise FileNotFoundError("FITS not found: {}".format(fits_file))

            pixel_x, pixel_y = radec_to_pixel_index(fits_file, target_ra, target_dec)

            output_row = {
                "name": name,
                "pixel_x": str(pixel_x),
                "pixel_y": str(pixel_y),
                "line_freq_GHz": str(row.get("line_freq_GHz", "")).strip(),
                "position_id": position_id,
            }
            if "band" in fieldnames:
                output_row["band"] = band or ""
            output_rows.append(output_row)
            ok_count += 1

            print(
                "Line {}: {} [{}] RA/Dec=({:.8f},{:.8f}) -> pixel=({}, {})".format(
                    row_id, name, position_id, target_ra, target_dec, pixel_x, pixel_y
                )
            )

        except Exception as e:
            err_count += 1
            print("Line {}: ERROR: {}".format(row_id, str(e)))
            if args.strict:
                sys.exit(1)

    if len(output_rows) == 0:
        print("Error: no valid rows converted")
        sys.exit(1)

    out_fields = build_output_fields(include_band=("band" in fieldnames))

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 50)
    print("Done. converted={}, errors={}".format(ok_count, err_count))
    print("Output CSV: {}".format(output_csv))
    print("=" * 50)


if __name__ == "__main__":
    main()
