#!/bin/bash

# 脚本名称：alma_project_level_6_prepare_radec_to_pixel.sh
# 功能：将 target_line_list_radec.csv 中的 RA/Dec 转换为 target_line_list.csv 的 pixel_x/pixel_y
# 使用方法：
#   ./alma_project_level_6_prepare_radec_to_pixel.sh
#   ./alma_project_level_6_prepare_radec_to_pixel.sh --input-csv xxx.csv --output-csv target_line_list.csv
#   ./alma_project_level_6_prepare_radec_to_pixel.sh --target-name TARGET
#   ./alma_project_level_6_prepare_radec_to_pixel.sh --position-id POS
#   ./alma_project_level_6_prepare_radec_to_pixel.sh --strict

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/alma_project_level_6_prepare_radec_to_pixel.py"

WORK_DIR="$(pwd)"
INPUT_CSV="${WORK_DIR}/target_line_list_radec.csv"
OUTPUT_CSV="${WORK_DIR}/target_line_list.csv"
IMAGE_TEMPLATE="Each_target_img/{name}/cubes/{name}_{band}.image.fits"
# 默认使用 python3；如需指定其他解释器，可运行：
#   PYTHON_CMD=/path/to/python3 bash alma_project_level_6_prepare_radec_to_pixel.sh
PYTHON_CMD="${PYTHON_CMD:-python3}"
TARGET_NAME=""
POSITION_ID=""
STRICT=0

usage() {
    echo "使用方法："
    echo "  $0"
    echo "  $0 --input-csv FILE --output-csv FILE"
    echo "  $0 --target-name NAME"
    echo "  $0 --position-id ID"
    echo "  $0 --image-template TEMPLATE"
    echo "  $0 --strict"
    echo ""
    echo "默认输入: ${WORK_DIR}/target_line_list_radec.csv"
    echo "默认输出: ${WORK_DIR}/target_line_list.csv"
    echo "默认图像: Each_target_img/{name}/cubes/{name}_{band}.image.fits"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-csv)
            INPUT_CSV="$2"
            shift 2
            ;;
        --output-csv)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --image-template)
            IMAGE_TEMPLATE="$2"
            shift 2
            ;;
        --target-name)
            TARGET_NAME="$2"
            shift 2
            ;;
        --position-id)
            POSITION_ID="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ ! -f "${PY_SCRIPT}" ]]; then
    echo "错误: 未找到 ${PY_SCRIPT}"
    exit 1
fi

if ! command -v "${PYTHON_CMD}" >/dev/null 2>&1; then
    echo "错误: 找不到 Python 解释器: ${PYTHON_CMD}"
    echo "请确认 python3 已加入 PATH，或设置 PYTHON_CMD=/path/to/python3"
    exit 1
fi

echo "=================================================="
echo "ALMA Level6 Prepare: RA/Dec -> Pixel"
echo "=================================================="
echo "工作目录: ${WORK_DIR}"
echo "输入CSV: ${INPUT_CSV}"
echo "输出CSV: ${OUTPUT_CSV}"
echo "图像模板: ${IMAGE_TEMPLATE}"
[[ -n "${TARGET_NAME}" ]] && echo "目标过滤: ${TARGET_NAME}"
[[ -n "${POSITION_ID}" ]] && echo "位置过滤: ${POSITION_ID}"
[[ "${STRICT}" -eq 1 ]] && echo "严格模式: 开启"
echo "=================================================="

CMD=("${PYTHON_CMD}" -u "${PY_SCRIPT}"
    --input-csv "${INPUT_CSV}"
    --output-csv "${OUTPUT_CSV}"
    --work-dir "${WORK_DIR}"
    --image-template "${IMAGE_TEMPLATE}"
)

if [[ -n "${TARGET_NAME}" ]]; then
    CMD+=(--target-name "${TARGET_NAME}")
fi
if [[ -n "${POSITION_ID}" ]]; then
    CMD+=(--position-id "${POSITION_ID}")
fi
if [[ "${STRICT}" -eq 1 ]]; then
    CMD+=(--strict)
fi

"${CMD[@]}"

echo "=================================================="
echo "完成！输出: ${OUTPUT_CSV}"
echo "=================================================="
