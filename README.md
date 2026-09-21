# ALMA Project Pipeline (`my_tools`)

End-to-end scripts for ALMA archival data: reorganize delivery → calibrate → split → concat → image → extract/fit lines → export UVFITS → collect GILDAS size fits.

All level scripts are meant to be run **from the project root** (the directory that will contain `Level_2_Calib/`, `Each_target_img/`, and your input CSVs).

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [System requirements](#2-system-requirements)
3. [Installation and environment](#3-installation-and-environment)
4. [Download ALMA data](#4-download-alma-data)
5. [Level-by-level guide](#5-level-by-level-guide)
6. [Input file formats](#6-input-file-formats)
7. [Environment variables](#7-environment-variables)
8. [Typical full run](#8-typical-full-run)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Pipeline overview

| Step | Script | What it does |
|------|--------|--------------|
| L1 | `alma_project_level_1_ReorgnizeDirForCalib.sh` | Move archive `member.uid*` dirs into `Level_2_Calib/DataSet_NN` |
| L2 | `alma_project_level_2_calib.bash` | Run ALMA `scriptForPI` calibration (correct CASA version per dataset) |
| L3 | `alma_project_level_3_SplitTargetFromCalib.sh` | Split science targets out of `calibrated.ms` |
| L4 | `alma_project_level_4_ConcatMS.sh` | Concat MS files per target, split by frequency band |
| L5 | `alma_project_level_5_imaging.sh` | Image spectral cubes with `tclean` |
| L6 | `alma_project_level_6_emission_line_process.sh` | Export FITS → RA/Dec→pixel → extract/fit spectra → line maps |
| L7 | `alma_project_level_7_generate_line_uvfits.sh` | Build `line_info.csv` and export continuum-subtracted UVFITS |
| — | *(GILDAS, manual)* | `fits_to_uvt` / `uv_fit` / save logs |
| L8 | `alma_project_level_8_read_gildas_fit_results.sh` | Collect GILDAS fit results into one CSV |

```text
ALMA Archive delivery
        │
        ▼
   Level 1  reorganize  →  Level_2_Calib/DataSet_*
        │
        ▼
   Level 2  calibrate   →  calibrated/calibrated.ms
        │
        ▼
   Level 3  split       →  Each_target_img/<src>/DataSet_*.ms
        │
        ▼
   Level 4  concat      →  Each_target_img/<src>/<src>_band_NN.ms
        │
        ▼
   Level 5  imaging     →  Each_target_img/<src>/cubes/*.image
        │
        ▼
   Level 6  lines       →  spectra, fits, line maps, optional GILDAS maps
        │
        ▼
   Level 7  UVFITS      →  line_info.csv + line.uvfits
        │
        ▼
   GILDAS uv_fit        →  size_gildas/test_log/*.gildas
        │
        ▼
   Level 8  collect     →  size_gildas/line_size_results.csv
```

---

## 2. System requirements

| Component | Role | Notes |
|-----------|------|--------|
| Linux | Host OS | Scripts assume bash + standard GNU tools |
| CASA (multiple versions) | Calibration & imaging | Put **all** needed versions under `~/Software/CASA/Portable/` |
| Python 3 | L6 prepare/extract, L7 phase 1, helpers | `python3` on `PATH` |
| Astropy | L6 RA/Dec → pixel | `pip install astropy` |
| BeautifulSoup4 | CASA-version lookup from QA weblog/HTML | `pip install beautifulsoup4` |
| GILDAS | Size fitting after L7 | Needed for the GILDAS step before L8 |
| MPI (optional) | Faster L5 imaging | Used by `alma_project_level_5_imaging.sh` |

---

## 3. Installation and environment

### 3.1 Clone / place the tools

Recommended layout:

```text
~/Software/
├── CASA/
│   ├── SETUP.bash
│   └── Portable/
│       ├── bin/bin_setup.bash
│       ├── casa-6.6.1-17-pipeline-2024.1.0.8/
│       ├── casa-6.5.4-9-pipeline-2023.1.0.124/
│       └── … (other versions as needed)
└── my_tools/          ← this repository
    ├── config/
    ├── scripts/
    └── README.md
```

### 3.2 Download CASA versions

1. Open the official download page: [https://casa.nrao.edu/casa_obtaining.shtml](https://casa.nrao.edu/casa_obtaining.shtml)  
   (Pipeline packages are listed under ALMA pipeline releases.)
2. Download the **pipeline** builds that match your archive deliveries (check each member’s README / QA weblog for “CASA version”).
3. Unpack each tarball into:

```text
~/Software/CASA/Portable/
```

Directory names should look like:

```text
casa-6.6.1-17-pipeline-2024.1.0.8
casa-6.5.4-9-pipeline-2023.1.0.124
casa-release-5.4.0-70.el7
…
```

**Why multiple versions?** Level 2 auto-selects the CASA version written in the delivery README/QA. Imaging (L3–L7) usually uses one modern default (see below).

### 3.3 CASA setup helpers

Level 2 expects:

| File | Location |
|------|----------|
| `SETUP.bash` | `~/Software/CASA/SETUP.bash` |
| `bin_setup.bash` | `~/Software/CASA/Portable/bin/bin_setup.bash` |

If you already have these files (as on a shared machine), leave them. Otherwise copy/adapt the versions used by your group and make them executable:

```bash
chmod +x ~/Software/CASA/SETUP.bash
chmod +x ~/Software/CASA/Portable/bin/bin_setup.bash
```

Manual switch example:

```bash
source ~/Software/CASA/SETUP.bash 6.6.1
casa --version
```

### 3.4 Edit `~/.bashrc`

Add (adjust paths if your home layout differs):

```bash
# Pipeline scripts
export PATH="$PATH:$HOME/Software/my_tools/scripts"

# Default CASA for Levels 3–7 (imaging / export / uvfits)
export CASA_DIR="$HOME/Software/CASA/Portable/casa-6.6.1-17-pipeline-2024.1.0.8"
export PATH="$CASA_DIR/bin:$PATH"
```

Then:

```bash
source ~/.bashrc
which casa
casa --version
```

### 3.5 Hardcoded CASA paths in wrappers

Some shell wrappers still contain absolute CASA paths. **Edit them once** after install:

| File | What to set |
|------|-------------|
| `alma_project_level_3_SplitTargetFromCalib.sh` | `CASA_CMD=…/casa` |
| `alma_project_level_4_ConcatMS.sh` | `CASA_CMD=…/casa` |
| `alma_project_level_5_imaging.sh` | `MPICASA`, `CASA_CMD`, optional `NPROC` |
| `alma_project_level_5_imaging.single_core.sh` | `CASA_CMD` (default `casa` on PATH) |

Prefer `$HOME/Software/CASA/Portable/<your-default>/bin/casa` so the same setup works for every user.

### 3.6 Python packages

```bash
python3 -m pip install --user astropy beautifulsoup4
```

Optional: set another interpreter with `export PYTHON_CMD=/path/to/python3`.

### 3.7 GILDAS (for size fitting)

Install GILDAS separately and ensure `mapping` / `class` are available before the post–Level-7 size-fitting step. Level 8 only **reads** existing `.gildas` logs; it does not run GILDAS itself.

---

## 4. Download ALMA data

1. Go to the [ALMA Science Archive](https://almascience.eso.org/aq/).
2. Search by project code (e.g. `2019.1.01634.L`).
3. Request / download the **raw + calibration scripts** delivery (member packages with `raw/`, `script/`, `qa/`, etc.).
4. Unpack into an empty project directory, for example:

```text
/data/alma/2019.1.01634.L/
├── science_goal.uid___A001_.../
│   └── group.uid___A001_.../
│       └── member.uid___A001_.../
│           ├── raw/
│           ├── script/
│           ├── qa/
│           └── …
```

You will run **all** subsequent commands from this project root.

---

## 5. Level-by-level guide

### Level 1 — Reorganize directories

**Prepare**

| Item | Required? |
|------|-----------|
| Unpacked ALMA delivery under current directory | Yes |
| Extra input files | No |

**Run**

```bash
cd /path/to/project
alma_project_level_1_ReorgnizeDirForCalib.sh
```

No CLI parameters. The script moves every `science*/group*/member.uid*` into `Level_2_Calib/DataSet_01`, `DataSet_02`, … and optionally asks to delete leftover top-level `science*` folders.

**Output**

```text
Level_2_Calib/
├── DataSet_01/
├── DataSet_02/
└── …
```

---

### Level 2 — Calibration (`scriptForPI`)

**Prepare**

| Item | Required? |
|------|-----------|
| `Level_2_Calib/DataSet_*` from Level 1 | Yes |
| Matching CASA pipeline builds under `~/Software/CASA/Portable/` | Yes |
| `SETUP.bash` + `bin_setup.bash` | Yes |

**Run**

```bash
alma_project_level_2_calib.bash PROJECT_CODE
alma_project_level_2_calib.bash PROJECT_CODE -dataset DataSet_01
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `PROJECT_CODE` (positional) | *required* | Project label for logs (e.g. `2019.1.01634.L`) |
| `-dataset NAME` | all `DataSet_*` | Repeatable; process only named datasets |

This calls `alma_archive_run_alma_pipeline_scriptForPI.sh`, which detects the CASA version from README / QA and runs calibration (or concatenates existing calibrated MS).

**Useful low-level call**

```bash
alma_archive_run_alma_pipeline_scriptForPI.sh [--nogui] Level_2_Calib/DataSet_01
```

**Output**

```text
Level_2_Calib/DataSet_NN/calibrated/calibrated.ms
Level_2_Calib/DataSet_NN/script/README_CASA_VERSION
```

---

### Level 3 — Split targets

**Prepare**

| Item | Required? |
|------|-----------|
| `Level_2_Calib/*/calibrated/calibrated.ms` | Yes |
| `target_list.txt` in project root | Yes |

Example `target_list.txt`:

```text
# one target name per line (must match FIELD names in listobs)
J0142-3327
J1223+0257
```

**Run**

```bash
alma_project_level_3_SplitTargetFromCalib.sh
```

No CLI flags (edit `CASA_CMD` in the wrapper if needed).

**Output**

```text
Each_target_img/<target>/DataSet_NN.ms
calibrated.listobs   (under each calibrated dir, as produced)
alma_level3_split.log
```

---

### Level 4 — Concat MS by frequency group

**Prepare**

| Item | Required? |
|------|-----------|
| `Each_target_img/<target>/DataSet_*.ms` | Yes |
| `target_list.txt` | Recommended (else directories under `Each_target_img/` are used) |

**Run**

```bash
alma_project_level_4_ConcatMS.sh

# optional: change band-splitting gap (GHz)
ALMA_FREQUENCY_GAP_GHZ=80 alma_project_level_4_ConcatMS.sh
```

| Parameter / env | Default | Meaning |
|-----------------|---------|---------|
| `ALMA_FREQUENCY_GAP_GHZ` | `100.0` | If the frequency gap between adjacent MS files exceeds this (GHz), start a new `band_NN` group |

**Output**

```text
Each_target_img/<target>/<target>_band_01.ms
Each_target_img/<target>/<target>_band_02.ms
Each_target_img/<target>/<target>_ms_groups.json
alma_level4_concat.log
```

---

### Level 5 — Imaging

**Prepare**

| Item | Required? |
|------|-----------|
| Concatenated `<target>_band_NN.ms` (preferred) or Level-3 MS | Yes |
| `target_list.txt` | Yes |
| Enough disk / cores for cubes | Yes |

**Run (MPI, recommended)**

```bash
alma_project_level_5_imaging.sh --channel-width-kms 20 --mask
```

**Run (single core)**

```bash
alma_project_level_5_imaging.single_core.sh --channel-width-kms 20 --mask
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--channel-width-kms` | `20` | Target cube channel width (km/s); rounded to nearest integer number of native channels |
| `--mask` | **on** | Fixed ellipse = `mask-radius-beams` × beam around MFS dirty peak |
| `--auto-mask` | | Per-channel auto-multithresh |
| `--no-mask` | | No mask (CLEAN over full image) |
| `--pbmask` | `0.5` | PB cutoff for auto-mask, in `(0, 1]` |
| `--dirty-mfs` / `--no-dirty-mfs` | dirty-mfs **on** | Build 2D MFS dirty image for peak/mask |
| `--mask-radius-beams` | `2.0` | Ellipse semi-axes in units of beam FWHM |
| `--mask-search-radius-beams` | `2.0` | Peak search radius around image center (beam FWHM) |

Environment alternatives: `ALMA_TARGET_CHANNEL_WIDTH_KMS`, `ALMA_AUTOMASK_PBMASK`, `ALMA_FREQUENCY_GAP_GHZ`.

Fixed in code (edit `alma_project_level_5_imaging.py` if needed): `imsize=512`, natural weighting, Hogbom, threshold ≈ 2×RMS. MPI wrapper uses `NPROC=16` by default.

**Output**

```text
Each_target_img/<target>/cubes/<target>_band_NN.image
Each_target_img/<target>/cubes/<target>_band_NN_mfs_dirty.image   # if dirty-mfs on
alma_level5_imaging_linecont_cube.log
```

---

### Level 6 — Emission-line processing

Orchestrator: `alma_project_level_6_emission_line_process.sh`  
(phases: export FITS → optional RA/Dec→pixel → extract/fit → linemap → optional GILDAS scripts)

**Prepare**

| Item | Required? | Notes |
|------|-----------|--------|
| Level-5 `.image` cubes | Yes | |
| `target_line_list_radec.csv` **or** `target_line_list.csv` | Yes | Prefer RA/Dec CSV; prepare step writes pixel CSV |
| `casa` on PATH | Yes | For export + linemap |
| `python3` (+ Astropy) | Yes | For prepare + extract |

**Run (full pipeline)**

```bash
alma_project_level_6_emission_line_process.sh
alma_project_level_6_emission_line_process.sh J0142-3327
alma_project_level_6_emission_line_process.sh --fit-range-ghz 4 --extraction-mode aperture
alma_project_level_6_emission_line_process.sh --recenter --recenter-min-snr 5
```

**Mode flags** (pick one)

| Flag | What runs |
|------|-----------|
| *(none)* | Full: export → prepare (if RA/Dec CSV) → extract+fit → linemap |
| `--export-only` | CASA `exportfits.py` only |
| `--prepare-only` | Export + RA/Dec → pixel |
| `--extract-only` | Spectrum extraction only |
| `--fit-only` | Gaussian fit only |
| `--specfit-only` | Extract + fit |
| `--catalog-only` | Rebuild project catalog from fit summaries |
| `--linemap-only` | CASA line maps |
| `--gildas-only` | Generate GILDAS `uv_fit` map scripts (`--phase4`) |

**Tuning parameters**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--fit-range-ghz` | `4` | Total frequency window for fitting (GHz) |
| `--extraction-mode` | `aperture` | `aperture`, `point`, or `both` |
| `--recenter` | off | Update pixel coords from local MFS dirty peak |
| `--recenter-radius-beams` | `1.5` | Peak search radius (beam FWHM) |
| `--recenter-min-snr` | `5` | Minimum peak SNR to accept recenter |
| `--catalog-min-snr` | `3` | Keep catalog rows with integrated SNR **>** this |
| `--catalog-output` | `emission_line_fit_catalog.csv` | Catalog filename |
| `TARGET_NAME` | all | Optional positional: process one source |

Standalone prepare (optional):

```bash
alma_project_level_6_prepare_radec_to_pixel.sh \
  --input-csv target_line_list_radec.csv \
  --output-csv target_line_list.csv
```

| Prepare flag | Default | Meaning |
|--------------|---------|---------|
| `--input-csv` | `target_line_list_radec.csv` | Input RA/Dec list |
| `--output-csv` | `target_line_list.csv` | Pixel list for extract/linemap |
| `--image-template` | `Each_target_img/{name}/cubes/{name}_{band}.image.fits` | FITS path pattern |
| `--target-name` / `--position-id` | | Restrict rows |
| `--strict` | off | Abort on first row error |
| `--work-dir` | `.` | Project root |

**Main outputs**

```text
Each_target_img/<name>/cubes/*.fits
Each_target_img/<name>/cubes/*_*.result.txt
Each_target_img/<name>/cubes/*_specfit.pdf
Each_target_img/<name>/cubes/*_gaussian_fit_summary.txt
emission_line_fit_catalog.csv
…/line_map / imfit products
size_gildas/uv_fit_line_*.map          # if --gildas-only / phase4
```

---

### Level 7 — Generate line UVFITS

**Prepare**

| Item | Required? |
|------|-----------|
| Level-6 `*_gaussian_fit_summary.txt` | Yes (Phase 1) |
| Level-4/5 MS for the same sources | Yes (Phase 2) |
| Editable `line_info.csv` after Phase 1 | Recommended before Phase 2 |

**Run**

```bash
alma_project_level_7_generate_line_uvfits.sh              # Phase 1 + 2
alma_project_level_7_generate_line_uvfits.sh --phase-1-only
alma_project_level_7_generate_line_uvfits.sh --phase-2-only
```

| Option | Meaning |
|--------|---------|
| *(default)* | Phase 1 then Phase 2 |
| `--phase-1-only` | Pure Python → `line_info.csv` |
| `--phase-2-only` | CASA: `uvcontsub` / split / concat / `exportuvfits` |

After Phase 1, review/edit `line_info.csv` (line centre / FWHM used for continuum subtraction) before Phase 2.

**Output**

```text
line_info.csv
Each_target_img/.../uvfits_output/line.uvfits
```

---

### GILDAS size fit (between Level 7 and 8)

Typical flow (details depend on your GILDAS setup):

1. Convert UVFITS → UVT and run `uv_fit` in `mapping`.
2. Save fit logs under `size_gildas/test_log/` as `line_result_<name>.gildas` (or consistent naming expected by Level 8).
3. Optional: export uvfit tables with `export_uvfit.map`:

```bash
# inside GILDAS mapping
@/path/to/my_tools/scripts/export_uvfit.map xx
```

Optional line-search helper (separate from L1–L8):

```bash
line_search_step3 -t TARGET -f '*.result.txt' -l 200 -h 600 -s true -z 3.98
```

See also `readme.txt` and `config/line_candidates`.

---

### Level 8 — Collect GILDAS results

**Prepare**

| Item | Required? |
|------|-----------|
| `size_gildas/test_log/*.gildas` | Yes |

**Run**

```bash
alma_project_level_8_read_gildas_fit_results.sh
```

No parameters.

**Output**

```text
size_gildas/line_size_results.csv
```

Columns: `src_name,line_flux,e_line_flux,line_flux_unit,CII_maj,e_CII_maj,CII_min,e_CII_min,CII_nu,e_CII_nu`.

---

## 6. Input file formats

### `target_list.txt` (Levels 3–5)

```text
# comments allowed
J0142-3327
J1223+0257
```

Names must match FIELD names in the calibrated MS (`listobs`).

### `target_line_list_radec.csv` (Level 6 prepare)

| Column | Required | Description |
|--------|----------|-------------|
| `name` | Yes | Target folder name under `Each_target_img/` |
| `ra` | Yes | Sexagesimal or degrees |
| `dec` | Yes | Sexagesimal or degrees |
| `line_freq_GHz` | Yes | Expected line frequency (GHz) |
| `position_id` | No | e.g. `default` (multi-position support) |
| `band` | No | e.g. `band_01` (selects which cube) |

```csv
name,ra,dec,line_freq_GHz,position_id,band
J0142-3327,01:42:43.73,-33:27:45.47,199.1213,default,band_01
J0142-3327,01:42:43.73,-33:27:45.47,335.1618,default,band_02
```

### `target_line_list.csv` (Level 6 extract / linemap)

Produced by prepare, or write by hand:

```csv
name,pixel_x,pixel_y,line_freq_GHz,position_id,band
```

### `line_info.csv` (Level 7)

Created by Phase 1. Typical columns include:

`name,band,position_id,uid,line_cen_GHz,FWHM_GHz,e_FWHM_GHz,line_cen_for_contsub,FWHM_for_contsub,aperture,snr`

Edit `*_for_contsub` columns if you need custom continuum windows.

### `config/line_candidates`

Rest-frequency list for optional line search:

```csv
line_name,rest_freq
CO21,230.538
CII,1900.537
```

---

## 7. Environment variables

| Variable | Used by | Default | Purpose |
|----------|---------|---------|---------|
| `ALMA_FREQUENCY_GAP_GHZ` | L4, L5 | `100.0` | Band grouping gap (GHz) |
| `ALMA_TARGET_CHANNEL_WIDTH_KMS` | L5 | `20.0` | Default `--channel-width-kms` |
| `ALMA_AUTOMASK_PBMASK` | L5 | `0.5` | Default `--pbmask` |
| `PYTHON_CMD` | L6 wrappers | `python3` | Python interpreter |
| `CASA_DIR` / `PATH` | shell | your bashrc | Default `casa` binary |
| `PYTHONPATH` | archive CASA finders | cleared during lookup | Avoid polluting CASA’s Python |

---

## 8. Typical full run

```bash
cd /path/to/2019.1.01634.L

# 0. Environment already set in ~/.bashrc
which casa && which alma_project_level_1_ReorgnizeDirForCalib.sh

# 1. Reorganize archive tree
alma_project_level_1_ReorgnizeDirForCalib.sh

# 2. Calibrate (needs correct Portable CASA builds)
alma_project_level_2_calib.bash 2019.1.01634.L

# 3. Create target_list.txt, then split
alma_project_level_3_SplitTargetFromCalib.sh

# 4. Concat by band
alma_project_level_4_ConcatMS.sh

# 5. Image
alma_project_level_5_imaging.sh --channel-width-kms 20 --mask

# 6. Prepare target_line_list_radec.csv, then process lines
alma_project_level_6_emission_line_process.sh

# 7. UVFITS (inspect line_info.csv between phases if needed)
alma_project_level_7_generate_line_uvfits.sh --phase-1-only
# edit line_info.csv if necessary
alma_project_level_7_generate_line_uvfits.sh --phase-2-only

# 8. Run GILDAS uv_fit; place logs in size_gildas/test_log/
alma_project_level_8_read_gildas_fit_results.sh
```

---

## 9. Troubleshooting

| Problem | What to check |
|---------|----------------|
| `casa: command not found` | `~/.bashrc` `CASA_DIR` / `PATH`; `source ~/.bashrc` |
| Level 2 cannot find CASA version | Member has `README` or `qa/*.tgz` / `qa/*.html`; install matching Portable build |
| Level 2: missing `SETUP.bash` | Place `~/Software/CASA/SETUP.bash` and `Portable/bin/bin_setup.bash` |
| Level 3 finds no targets | FIELD names in MS vs `target_list.txt` (run `listobs`) |
| Level 5 MPI fails | Use `alma_project_level_5_imaging.single_core.sh`, or fix `MPICASA` / `NPROC` |
| Level 6 prepare: FITS not found | Run export first (`--export-only` or full L6) |
| Level 6 extract needs Astropy | `python3 -m pip install --user astropy` |
| Level 7 Phase 2 fails on `uvcontsub` | Need CASA ≳ 6.5 with `fitspec=`-style `uvcontsub` |
| Level 8: directory missing | Create `size_gildas/test_log/` and put `*.gildas` logs there |
| Scripts not found | `export PATH="$PATH:$HOME/Software/my_tools/scripts"` |

---

## Repository layout

```text
my_tools/
├── README.md
├── readme.txt                          # short GILDAS / line-search notes
├── config/
│   └── line_candidates
└── scripts/
    ├── alma_project_level_1_….sh
    ├── alma_project_level_2_….bash
    ├── …
    ├── alma_project_level_8_….sh
    ├── alma_archive_*.sh / *.py
    ├── exportfits.py
    ├── export_uvfit.map
    ├── line_detection.py
    └── line_search_step3
```

---

## Notes

- Always `cd` to the **project root** before calling level scripts.
- Level 2 may need **many** historical CASA builds; Levels 3–7 typically use one modern default (e.g. `casa-6.6.1-17-pipeline-2024.1.0.8`).
- For one-off tests, prefer mode flags (`--export-only`, `--phase-1-only`, …) instead of re-running the full chain.
- Legacy scripts (`prepare_for_gildas.py`, `spec_extraction_from_ALMA_img.py`) are superseded by Levels 6–7; keep them only for reference.
