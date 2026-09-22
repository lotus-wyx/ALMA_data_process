#!/bin/bash

# 脚本名称：alma_project_level_6_emission_line_process.sh
# 功能：从ALMA成图结果中提取光谱并进行高斯拟合，生成line map
# 可通过修改target_line_list.csv来指定不同的目标和谱线
# 使用方法：
#   ./script.sh                    # 完整流程（export + prepare + specfit + linemap）
#   ./script.sh --export-only      # 仅Phase 1（导出FITS）
#   ./script.sh --prepare-only     # 仅Phase 2 (导出FITS,并RA/Dec转pixel)
#   ./script.sh --extract-only     # 仅Phase 3a（抽谱）"
#   ./script.sh --fit-only         # 仅Phase 3b（谱线拟合）"
#   ./script.sh --specfit-only     # 兼容旧版：抽谱 + 拟合"
#   ./script.sh --catalog-only     # 从已有拟合summary生成项目总表
#   ./script.sh --linemap-only     # 仅Phase 4（生成line map）
#   ./script.sh --gildas-only      # 仅Phase 5（生成GILDAS脚本）
#   ./script.sh --fit-range-ghz 4  # 设置拟合总频率窗口（GHz）
#   ./script.sh --extraction-mode point  # 使用中心像素点抽谱
#   ./script.sh --recenter         # 从局部 MFS dirty 峰更新 CSV 像素位置
#   ./script.sh TARGET_NAME        # 处理单个目标

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/exportfits.py"
SPECFIT_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_extract_spectrum.py"
LINEMAP_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_generate_linemap.py"
PREPARE_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_prepare_radec_to_pixel.sh"
WORK_DIR="$(pwd)"

CASA_CMD="casa"
# 默认使用 python3；可通过 PYTHON_CMD 指定其他 Python 解释器。
PYTHON_CMD="${PYTHON_CMD:-python3}"

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
    echo "错误: 找不到 Python 解释器: ${PYTHON_CMD}"
    echo "请确认 python3 已加入 PATH，或设置 PYTHON_CMD=/path/to/python3"
    exit 1
fi

# 解析参数
MODE="full"  # full, prepare, export, extract, fit, specfit, catalog, linemap, gildas
TARGET_NAME=""
SPECFIT_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --prepare-only)
            MODE="prepare"
            shift
            ;;
        --export-only)
            MODE="export"
            shift
            ;;
        --extract-only)
            MODE="extract"
            shift
            ;;
        --fit-only)
            MODE="fit"
            shift
            ;;
        --specfit-only)
            MODE="specfit"
            shift
            ;;
        --catalog-only)
            MODE="catalog"
            shift
            ;;
        --linemap-only)
            MODE="linemap"
            shift
            ;;
        --gildas-only)
            MODE="gildas"
            shift
            ;;
        --fit-range-ghz|--extraction-mode|--recenter-radius-beams|--recenter-min-snr|--catalog-min-snr|--catalog-output)
            if [ "$#" -lt 2 ]; then
                echo "错误: $1 需要一个参数"
                exit 1
            fi
            SPECFIT_ARGS+=("$1" "$2")
            shift 2
            ;;
        --fit-range-ghz=*|--extraction-mode=*|--recenter-radius-beams=*|--recenter-min-snr=*|--catalog-min-snr=*|--catalog-output=*)
            SPECFIT_ARGS+=("$1")
            shift
            ;;
        --recenter)
            SPECFIT_ARGS+=("$1")
            shift
            ;;
        --help|-h)
            echo "使用方法："
            echo "  $0                         # 完整流程（export + prepare + specfit + linemap）"
            echo "  $0 --export-only           # 仅Phase 1（导出FITS）"
            echo "  $0 --prepare-only          # 仅Phase 2 (导出FITS并执行RA/Dec转pixel)"
            echo "  $0 --extract-only          # 仅Phase 3a（抽谱）"
            echo "  $0 --fit-only              # 仅Phase 3b（谱线拟合）"
            echo "  $0 --specfit-only          # 兼容旧版：抽谱 + 拟合"
            echo "  $0 --catalog-only          # 从已有拟合summary生成项目总表"
            echo "  $0 --linemap-only          # 仅Phase 4（生成line map）"
            echo "  $0 --gildas-only           # 仅Phase 5（生成GILDAS脚本）"
            echo "  $0 --fit-range-ghz GHZ     # 拟合总频率窗口（默认4 GHz）"
            echo "  $0 --extraction-mode MODE  # aperture、point 或 both（默认aperture）"
            echo "  $0 --recenter              # 用局部MFS dirty峰更新CSV坐标"
            echo "  $0 --recenter-radius-beams N # 搜索半径（默认1.5 beam）"
            echo "  $0 --recenter-min-snr SNR  # 更新坐标所需最低峰值SNR（默认5）"
            echo "  $0 --catalog-min-snr SNR   # 总表积分SNR严格下限（默认>3）"
            echo "  $0 --catalog-output FILE   # 总表文件名"
            echo "  $0 TARGET_NAME             # 处理单个目标"
            exit 0
            ;;
        --*)
            echo "错误: 未知参数 $1"
            exit 1
            ;;
        *)
            if [ -z "${TARGET_NAME}" ]; then
                TARGET_NAME="$1"
            else
                echo "错误: 只能指定一个目标名称"
                exit 1
            fi
            shift
            ;;
    esac
done

echo "=================================================="
echo "ALMA Project Level 6 Pipeline"
echo "=================================================="
echo "工作目录: ${WORK_DIR}"
[ -n "${TARGET_NAME}" ] && echo "目标: ${TARGET_NAME}" || echo "模式: 处理所有目标"
echo "运行阶段: ${MODE}"
echo "=================================================="

cd "${WORK_DIR}"
LOG_FILE="alma_level6_fit_line_make_linemap.log"

# Phase 0: 导出FITS
if [ "${MODE}" == "full" ] || [ "${MODE}" == "export" ] || [ "${MODE}" == "prepare" ]; then
    echo "[Phase 1] 导出FITS..."
    stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${EXPORT_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
fi

# Phase 1: RA/Dec 转 pixel（必须在导出FITS之后）
RUN_PREPARE=0
if [ "${MODE}" == "prepare" ]; then
    RUN_PREPARE=1
elif [ "${MODE}" == "full" ] && [ -f "${WORK_DIR}/target_line_list_radec.csv" ]; then
    RUN_PREPARE=1
fi

if [ "${RUN_PREPARE}" -eq 1 ]; then
    echo "[Phase 2] RA/Dec 转 pixel..."
    if [ ! -f "${PREPARE_SCRIPT}" ]; then
        echo "错误: 未找到预处理脚本 ${PREPARE_SCRIPT}"
        exit 1
    fi

    if [ -n "${TARGET_NAME}" ]; then
        bash "${PREPARE_SCRIPT}" --target-name "${TARGET_NAME}"
    else
        bash "${PREPARE_SCRIPT}"
    fi
elif [ "${MODE}" == "full" ]; then
    echo "[Phase 2] 未检测到 target_line_list_radec.csv，跳过 prepare，继续使用现有 target_line_list.csv"
fi

if [ "${MODE}" == "prepare" ]; then
    echo "=================================================="
    echo "完成！日志: ${LOG_FILE}"
    echo "=================================================="
    exit 0
fi

# 检查并修复 CSV 文件的 UTF-8 BOM
CSV_FILE="${WORK_DIR}/target_line_list.csv"
if [ -f "${CSV_FILE}" ]; then
    echo "检查 CSV 文件编码..."
    # 检查文件前3个字节是否为 EF BB BF (UTF-8 BOM)
    BOM_CHECK=$(hexdump -n 3 -e '3/1 "%02X"' "${CSV_FILE}" 2>/dev/null)
    if [ "${BOM_CHECK}" = "EFBBBF" ]; then
        echo "检测到 UTF-8 BOM，正在移除..."
        sed -i '1s/^\xEF\xBB\xBF//' "${CSV_FILE}"
        echo "UTF-8 BOM 已移除"
    else
        echo "CSV 文件编码正常"
    fi
else
    echo "警告: 未找到 ${CSV_FILE}"
fi
echo ""

# 仅从已有逐目标 summary 重建项目级发射线总表
if [ "${MODE}" == "catalog" ]; then
    echo "[Phase 3c] 汇总高信噪比发射线..."
    "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --catalog-only "${SPECFIT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi

# Phase 3a: 抽谱
if [ "${MODE}" == "full" ] || [ "${MODE}" == "extract" ]; then
    echo "[Phase 3a] 抽谱..."
    if [ -n "${TARGET_NAME}" ]; then
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --extract-only "${SPECFIT_ARGS[@]}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --extract-only "${SPECFIT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# Phase 3b: 谱线拟合
if [ "${MODE}" == "full" ] || [ "${MODE}" == "fit" ]; then
    echo "[Phase 3b] 谱线拟合..."
    if [ -n "${TARGET_NAME}" ]; then
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --fit-only "${SPECFIT_ARGS[@]}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --fit-only "${SPECFIT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# 兼容旧用法：一次性抽谱+拟合
if [ "${MODE}" == "specfit" ]; then
    echo "[Phase 3] 抽谱 + 拟合..."
    if [ -n "${TARGET_NAME}" ]; then
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --specfit-only "${SPECFIT_ARGS[@]}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        "${PYTHON_CMD}" -u "${SPECFIT_SCRIPT}" --specfit-only "${SPECFIT_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# Phase 4: 生成Line Map
if [ "${MODE}" == "full" ] || [ "${MODE}" == "linemap" ]; then
    echo "[Phase 4] 生成Line Map..."
    if [ -n "${TARGET_NAME}" ]; then
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# Phase 5: 生成GILDAS脚本
if [ "${MODE}" == "gildas" ]; then
    echo "[Phase 5] 生成GILDAS脚本..."
    LOG_FILE="alma_level6_make_gildas_uvfit_script.log"
    if [ -n "${TARGET_NAME}" ]; then
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" "${TARGET_NAME}" --phase4 2>&1 | tee -a "${LOG_FILE}"
    else
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" --phase4 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

echo "=================================================="
echo "完成！日志: ${LOG_FILE}"
echo "=================================================="
