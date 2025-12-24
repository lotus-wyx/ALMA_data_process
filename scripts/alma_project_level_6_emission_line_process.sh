#!/bin/bash

# 脚本名称：alma_project_level_6_extract_spectrum.sh
# 功能：从ALMA成图结果中提取光谱并进行高斯拟合，生成line map
# 
# 使用方法：
#   ./script.sh                    # 完整流程（Phase 1 + Phase 2 + Phase 3）
#   ./script.sh --export-only      # 仅Phase 1（导出FITS）
#   ./script.sh --specfit-only     # 仅Phase 2（提取光谱和拟合）
#   ./script.sh --linemap-only     # 仅Phase 3（生成line map）
#   ./script.sh --gildas-only      # 仅Phase 4（生成GILDAS脚本）
#   ./script.sh TARGET_NAME        # 处理单个目标

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="${SCRIPT_DIR}/exportfits.py"
SPECFIT_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_extract_spectrum.py"
LINEMAP_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_generate_linemap.py"
WORK_DIR="$(pwd)"

CASA_CMD="casa"

# 解析参数
MODE="full"  # full, export, specfit, linemap, gildas
TARGET_NAME=""

for arg in "$@"; do
    case $arg in
        --export-only)
            MODE="export"
            ;;
        --specfit-only)
            MODE="specfit"
            ;;
        --linemap-only)
            MODE="linemap"
            ;;
        --gildas-only)
            MODE="gildas"
            ;;
        --help|-h)
            echo "使用方法："
            echo "  $0                         # 完整流程（Phase 1 + 2 + 3）"
            echo "  $0 --export-only           # 仅Phase 1（导出FITS）"
            echo "  $0 --specfit-only          # 仅Phase 2（光谱拟合）"
            echo "  $0 --linemap-only          # 仅Phase 3（生成line map）"
            echo "  $0 --gildas-only           # 仅Phase 4（生成GILDAS脚本）"
            echo "  $0 TARGET_NAME             # 处理单个目标"
            exit 0
            ;;
        *)
            if [ -z "${TARGET_NAME}" ] && [[ ! "$arg" == --* ]]; then
                TARGET_NAME="$arg"
            fi
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
LOG_FILE="alma_level6_fit_line_make_linemap.log"

# Phase 1: 导出FITS
if [ "${MODE}" == "full" ] || [ "${MODE}" == "export" ]; then
    echo "[Phase 1] 导出FITS..."
    stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${EXPORT_SCRIPT}" 2>&1 | tee "${LOG_FILE}"
fi

# Phase 2: 光谱拟合
if [ "${MODE}" == "full" ] || [ "${MODE}" == "specfit" ]; then
    echo "[Phase 2] 光谱拟合..."
    if [ -n "${TARGET_NAME}" ]; then
        python -u "${SPECFIT_SCRIPT}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        python -u "${SPECFIT_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# Phase 3: 生成Line Map
if [ "${MODE}" == "full" ] || [ "${MODE}" == "linemap" ]; then
    echo "[Phase 3] 生成Line Map..."
    if [ -n "${TARGET_NAME}" ]; then
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" "${TARGET_NAME}" 2>&1 | tee -a "${LOG_FILE}"
    else
        stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${LINEMAP_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

# Phase 4: 生成GILDAS脚本
if [ "${MODE}" == "gildas" ]; then
    echo "[Phase 4] 生成GILDAS脚本..."
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