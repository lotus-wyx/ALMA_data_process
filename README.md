# ALMA Data Process

End-to-end tools for ALMA archival data: reorganize delivery → calibrate → split → concat → image → extract/fit lines → export UVFITS → collect GILDAS size fits.

This repository is meant to live under **`~/Software/`** and provides two top-level folders:

| Folder | What this repo ships | What you add yourself |
|--------|----------------------|------------------------|
| `CASA/` | `SETUP.bash` + `Portable/bin/` (version switcher) | Pipeline CASA builds under `Portable/` |
| `my_tools/` | Level 1–8 scripts + config | Nothing (optional local edits) |

All level scripts are run from your **ALMA project root** (the directory that will contain `Level_2_Calib/`, `Each_target_img/`, and your input CSVs)—not from `~/Software/`.

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
| CASA (multiple versions) | Calibration & imaging | Install under `~/Software/CASA/Portable/` |
| Python 3 | L6 prepare/extract, L7 phase 1, helpers | `python3` on `PATH` |
| Astropy | L6 RA/Dec → pixel | `pip install astropy` |
| BeautifulSoup4 | CASA-version lookup from QA weblog/HTML | `pip install beautifulsoup4` |
| GILDAS | Size fitting after L7 | Install separately; needed before L8 |
| MPI (optional) | Faster L5 imaging | Used by `alma_project_level_5_imaging.sh` |

---

## 3. Installation and environment

### 3.1 Create `~/Software` and clone this repository

```bash
mkdir -p ~/Software
cd ~/Software
git clone https://github.com/lotus-wyx/ALMA_data_process.git .
```

If `~/Software` already has other files, clone into a temp dir and copy `CASA/` + `my_tools/` (and this `README.md`) into `~/Software/` instead:

```bash
git clone https://github.com/lotus-wyx/ALMA_data_process.git /tmp/ALMA_data_process
cp -a /tmp/ALMA_data_process/CASA /tmp/ALMA_data_process/my_tools /tmp/ALMA_data_process/README.md ~/Software/
```

After a successful clone, your home layout should look like:

```text
~/Software/
├── README.md                 ← this file
├── CASA/
│   ├── SETUP.bash            ← provided by this repo
│   └── Portable/
│       ├── bin/              ← provided (bin_setup.bash)
│       │   ├── bin_setup.bash
│       │   └── bin_setup.readme
│       └── (CASA builds you install yourself; see §3.2)
└── my_tools/
    ├── config/
    ├── scripts/              ← Level 1–8 + archive helpers
    └── readme.txt
```

Make the setup helpers executable:

```bash
chmod +x ~/Software/CASA/SETUP.bash
chmod +x ~/Software/CASA/Portable/bin/bin_setup.bash
```

### 3.2 Install CASA versions into `Portable/`

This repository does **not** ship the multi‑GB CASA binaries. Download them from NRAO and unpack into:

```text
~/Software/CASA/Portable/
```

1. Open [https://casa.nrao.edu/casa_obtaining.shtml](https://casa.nrao.edu/casa_obtaining.shtml) (use the ALMA **pipeline** packages when available).
2. For each ALMA delivery, check the member README / QA weblog for the required “CASA version”, then install a matching build.
3. Keep the unpacked directory names as distributed (examples below).

**Reference list — versions commonly used with this pipeline** (install what you need; you do not need all of them):

| Directory name under `Portable/` | Typical use |
|----------------------------------|-------------|
| `casa-6.6.6-18-pipeline-2025.1.0.36-py3.10.el8` | Recent Cycle pipeline |
| `casa-6.6.6-17-pipeline-2025.1.0.35-py3.10.el8` | Recent Cycle pipeline |
| `casa-6.6.1-17-pipeline-2024.1.0.8` | **Default for Levels 3–7** in this repo |
| `casa-6.5.4-9-pipeline-2023.1.0.124` | Cycle ~10 pipeline |
| `casa-6.4.1-12-pipeline-2022.2.0.68` | Cycle ~9 pipeline |
| `casa-6.4.1-12-pipeline-2022.2.0.64` | Cycle ~9 pipeline |
| `casa-6.2.1-7-pipeline-2021.2.0.128` | Cycle ~8 pipeline |
| `casa-6.1.1-15-pipeline-2020.1.0.40` | Cycle ~7 pipeline |
| `casa-pipeline-release-5.6.1-8.el7` | CASA 5.6 pipeline |
| `casa-release-5.7.2-4.el7` | CASA 5.7 |
| `casa-release-5.4.0-70.el7` | CASA 5.4 |
| `casa-release-5.4.0-68.el7` | CASA 5.4 |
| `casa-release-5.1.1-5.el7` | CASA 5.1 |
| `casa-release-4.7.2-el7` | CASA 4.7 |
| `casa-release-4.7.0-1-el7` | CASA 4.7 |
| `casa-release-4.5.3-el6` | CASA 4.5 |
| `casa-release-4.5.2-el6` | CASA 4.5 |
| `casa-release-4.5.1-el6` | CASA 4.5 |
| `casa-release-4.4.0-el6` | CASA 4.4 |
| `casa-release-4.3.1-pipe-el6` | CASA 4.3 pipeline |
| `casapy-4.2.2.30986-pipe-1-64b` | CASA 4.2 pipeline |
| `casapy-42.1.29047-001-1-64b` | Older CASA 4.2 |
| `casapy-41.0.24668-001-64b-2` | Older CASA 4.1 |

**Why multiple versions?** Level 2 auto-selects the CASA version written in each delivery’s README/QA via `SETUP.bash`. Imaging (Levels 3–7) usually uses one modern default (see §3.4).

Manual version switch:

```bash
source ~/Software/CASA/SETUP.bash 6.6.1
casa --version
```

### 3.3 Edit `~/.bashrc`

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
ls ~/Software/my_tools/scripts/alma_project_level_1_ReorgnizeDirForCalib.sh
```

### 3.4 CASA paths in shell wrappers

Wrappers prefer `$HOME/Software/CASA/Portable/<version>/bin/casa` (or `casa` on `PATH`). After install, confirm these match your default build:

| File | Setting |
|------|---------|
| `my_tools/scripts/alma_project_level_3_SplitTargetFromCalib.sh` | `CASA_CMD=$HOME/Software/CASA/Portable/casa-6.6.1-17-pipeline-2024.1.0.8/bin/casa` |
| `my_tools/scripts/alma_project_level_4_ConcatMS.sh` | same pattern |
| `my_tools/scripts/alma_project_level_5_imaging.sh` | `MPICASA` + `CASA_CMD` (+ optional `NPROC`) |
| `my_tools/scripts/alma_project_level_5_imaging.single_core.sh` | `CASA_CMD=casa` (uses `PATH`) |
| `my_tools/scripts/alma_project_level_6_emission_line_process.sh` | `CASA_CMD=casa` |
| `my_tools/scripts/alma_project_level_7_generate_line_uvfits.sh` | `CASA_CMD=casa` |

### 3.5 Python packages

```bash
python3 -m pip install --user astropy beautifulsoup4
```

Optional: `export PYTHON_CMD=/path/to/python3`.

### 3.6 GILDAS (for size fitting)

Install GILDAS separately and ensure `mapping` / `class` are available before the post–Level-7 size-fitting step. Level 8 only **reads** existing `.gildas` logs; it does not run GILDAS itself.

---

## 4. Download ALMA data

1. Go to the [ALMA Science Archive](https://almascience.eso.org/aq/).
2. Search by project code (e.g. `2019.1.01634.L`).
3. Request / download the **raw + calibration scripts** delivery (member packages with `raw/`, `script/`, `qa/`, etc.).
4. Unpack into an empty **project** directory (anywhere; not inside `~/Software/` unless you prefer that), for example:

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

Run **all** subsequent level commands from this project root.

---

## 5. Level-by-level guide

Scripts live in `~/Software/my_tools/scripts/` and should be on your `PATH` after §3.3.

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

No CLI parameters. Moves every `science*/group*/member.uid*` into `Level_2_Calib/DataSet_01`, `DataSet_02`, … and optionally asks to delete leftover top-level `science*` folders.

**Output:** `Level_2_Calib/DataSet_NN/`

---

### Level 2 — Calibration (`scriptForPI`)

**Prepare**

| Item | Required? |
|------|-----------|
| `Level_2_Calib/DataSet_*` from Level 1 | Yes |
| Matching CASA builds under `~/Software/CASA/Portable/` | Yes |
| `SETUP.bash` + `bin_setup.bash` (from this repo) | Yes |

**Run**

```bash
alma_project_level_2_calib.bash PROJECT_CODE
alma_project_level_2_calib.bash PROJECT_CODE -dataset DataSet_01
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `PROJECT_CODE` (positional) | *required* | Project label for logs (e.g. `2019.1.01634.L`) |
| `-dataset NAME` | all `DataSet_*` | Repeatable; process only named datasets |

Low-level call:

```bash
alma_archive_run_alma_pipeline_scriptForPI.sh [--nogui] Level_2_Calib/DataSet_01
```

**Output:** `Level_2_Calib/DataSet_NN/calibrated/calibrated.ms`, `README_CASA_VERSION`

---

### Level 3 — Split targets

**Prepare:** calibrated MS + `target_list.txt` in the project root.

```text
# target_list.txt — one FIELD name per line
J0142-3327
J1223+0257
```

```bash
alma_project_level_3_SplitTargetFromCalib.sh
```

**Output:** `Each_target_img/<target>/DataSet_NN.ms`

---

### Level 4 — Concat MS by frequency group

```bash
alma_project_level_4_ConcatMS.sh
ALMA_FREQUENCY_GAP_GHZ=80 alma_project_level_4_ConcatMS.sh
```

| Parameter / env | Default | Meaning |
|-----------------|---------|---------|
| `ALMA_FREQUENCY_GAP_GHZ` | `100.0` | Start a new `band_NN` if adjacent MS gap exceeds this (GHz) |

**Output:** `<target>_band_NN.ms`, `<target>_ms_groups.json`

---

### Level 5 — Imaging

```bash
alma_project_level_5_imaging.sh --channel-width-kms 20 --mask
alma_project_level_5_imaging.single_core.sh --channel-width-kms 20 --mask
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--channel-width-kms` | `20` | Target cube channel width (km/s) |
| `--mask` | **on** | Fixed ellipse around MFS dirty peak |
| `--auto-mask` | | Per-channel auto-multithresh |
| `--no-mask` | | No mask |
| `--pbmask` | `0.5` | PB cutoff for auto-mask, in `(0, 1]` |
| `--dirty-mfs` / `--no-dirty-mfs` | dirty-mfs **on** | 2D MFS dirty image for peak/mask |
| `--mask-radius-beams` | `2.0` | Ellipse size in beam FWHM |
| `--mask-search-radius-beams` | `2.0` | Peak search radius (beam FWHM) |

Env alternatives: `ALMA_TARGET_CHANNEL_WIDTH_KMS`, `ALMA_AUTOMASK_PBMASK`, `ALMA_FREQUENCY_GAP_GHZ`.

**Output:** `Each_target_img/<target>/cubes/*.image`

---

### Level 6 — Emission-line processing

**Prepare:** Level-5 cubes + `target_line_list_radec.csv` (or `target_line_list.csv`).

```bash
alma_project_level_6_emission_line_process.sh
alma_project_level_6_emission_line_process.sh J0142-3327
alma_project_level_6_emission_line_process.sh --fit-range-ghz 4 --extraction-mode aperture
alma_project_level_6_emission_line_process.sh --recenter
```

**Mode flags** (pick one): `--export-only`, `--prepare-only`, `--extract-only`, `--fit-only`, `--specfit-only`, `--catalog-only`, `--linemap-only`, `--gildas-only`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--fit-range-ghz` | `4` | Fit window (GHz) |
| `--extraction-mode` | `aperture` | `aperture` \| `point` \| `both` |
| `--recenter` | off | Update pixels from MFS dirty peak |
| `--recenter-radius-beams` | `1.5` | Search radius |
| `--recenter-min-snr` | `5` | Min peak SNR to accept recenter |
| `--catalog-min-snr` | `3` | Catalog rows with integrated SNR **>** this |
| `--catalog-output` | `emission_line_fit_catalog.csv` | Catalog filename |

---

### Level 7 — Generate line UVFITS

```bash
alma_project_level_7_generate_line_uvfits.sh
alma_project_level_7_generate_line_uvfits.sh --phase-1-only
# review/edit line_info.csv
alma_project_level_7_generate_line_uvfits.sh --phase-2-only
```

**Output:** `line_info.csv`, `.../uvfits_output/line.uvfits`

---

### GILDAS size fit (between Level 7 and 8)

1. Convert UVFITS → UVT and run `uv_fit` in `mapping`.
2. Place logs under `size_gildas/test_log/` as `line_result_<name>.gildas`.
3. Optional: `@~/Software/my_tools/scripts/export_uvfit.map xx` inside GILDAS.

Optional line search: see `my_tools/readme.txt` and `my_tools/config/line_candidates`.

---

### Level 8 — Collect GILDAS results

```bash
alma_project_level_8_read_gildas_fit_results.sh
```

**Output:** `size_gildas/line_size_results.csv`

---

## 6. Input file formats

### `target_list.txt` (Levels 3–5)

```text
# comments allowed
J0142-3327
J1223+0257
```

### `target_line_list_radec.csv` (Level 6 prepare)

| Column | Required | Description |
|--------|----------|-------------|
| `name` | Yes | Folder name under `Each_target_img/` |
| `ra`, `dec` | Yes | Sexagesimal or degrees |
| `line_freq_GHz` | Yes | Expected line frequency (GHz) |
| `position_id` | No | e.g. `default` |
| `band` | No | e.g. `band_01` |

```csv
name,ra,dec,line_freq_GHz,position_id,band
J0142-3327,01:42:43.73,-33:27:45.47,199.1213,default,band_01
```

### `target_line_list.csv` (Level 6 extract / linemap)

```csv
name,pixel_x,pixel_y,line_freq_GHz,position_id,band
```

### `line_info.csv` (Level 7)

Created by Phase 1; edit `*_for_contsub` columns before Phase 2 if needed.

---

## 7. Environment variables

| Variable | Used by | Default | Purpose |
|----------|---------|---------|---------|
| `ALMA_FREQUENCY_GAP_GHZ` | L4, L5 | `100.0` | Band grouping gap (GHz) |
| `ALMA_TARGET_CHANNEL_WIDTH_KMS` | L5 | `20.0` | Default `--channel-width-kms` |
| `ALMA_AUTOMASK_PBMASK` | L5 | `0.5` | Default `--pbmask` |
| `PYTHON_CMD` | L6 wrappers | `python3` | Python interpreter |
| `CASA_DIR` / `PATH` | shell | your bashrc | Default `casa` binary |

---

## 8. Typical full run

```bash
cd /path/to/2019.1.01634.L

alma_project_level_1_ReorgnizeDirForCalib.sh
alma_project_level_2_calib.bash 2019.1.01634.L
# create target_list.txt
alma_project_level_3_SplitTargetFromCalib.sh
alma_project_level_4_ConcatMS.sh
alma_project_level_5_imaging.sh --channel-width-kms 20 --mask
# create target_line_list_radec.csv
alma_project_level_6_emission_line_process.sh
alma_project_level_7_generate_line_uvfits.sh --phase-1-only
# edit line_info.csv if needed
alma_project_level_7_generate_line_uvfits.sh --phase-2-only
# GILDAS uv_fit → size_gildas/test_log/*.gildas
alma_project_level_8_read_gildas_fit_results.sh
```

---

## 9. Troubleshooting

| Problem | What to check |
|---------|----------------|
| `casa: command not found` | `~/.bashrc` `CASA_DIR` / `PATH`; install a build under `Portable/` |
| Level 2 cannot find CASA version | Member README/QA present; matching Portable folder name installed |
| Level 2: missing `SETUP.bash` | Re-clone so `~/Software/CASA/SETUP.bash` and `Portable/bin/bin_setup.bash` exist |
| Level 3 finds no targets | FIELD names in MS vs `target_list.txt` |
| Level 5 MPI fails | Use `alma_project_level_5_imaging.single_core.sh` |
| Level 6 prepare: FITS not found | Run `--export-only` (or full L6) first |
| Scripts not found | `export PATH="$PATH:$HOME/Software/my_tools/scripts"` |
| Accidentally committed a huge CASA build | Builds must stay gitignored under `CASA/Portable/casa*` |

---

## Repository layout

```text
.                          ← clone into ~/Software/
├── README.md
├── .gitignore             ← excludes CASA Portable installs, gildas, etc.
├── CASA/
│   ├── SETUP.bash
│   └── Portable/
│       └── bin/
│           ├── bin_setup.bash
│           └── bin_setup.readme
└── my_tools/
    ├── config/line_candidates
    ├── readme.txt
    └── scripts/
        ├── alma_project_level_1_….sh … _8_….sh
        ├── alma_archive_*.sh / *.py
        ├── exportfits.py
        ├── export_uvfit.map
        ├── line_detection.py
        └── line_search_step3
```

---

## Notes

- Always `cd` to the **ALMA project root** before calling level scripts.
- Level 2 may need many historical CASA builds; Levels 3–7 typically use `casa-6.6.1-17-pipeline-2024.1.0.8`.
- For one-off tests, prefer mode flags (`--export-only`, `--phase-1-only`, …) instead of re-running the full chain.
- Legacy scripts (`prepare_for_gildas.py`, `spec_extraction_from_ALMA_img.py`) are superseded by Levels 6–7; keep them only for reference.
