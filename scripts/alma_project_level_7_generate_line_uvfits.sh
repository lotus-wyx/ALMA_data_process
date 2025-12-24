#!/bin/bash
#
# ALMA Project Level 7 - Generate UVFITS from Level 6 Results
#
# 用法：
#   ./alma_project_level_7_generate_uvfits.sh                   # 运行完整流程（Phase 1 + 2）
#   ./alma_project_level_7_generate_uvfits.sh --phase-1-only    # 只运行 Phase 1
#   ./alma_project_level_7_generate_uvfits.sh --phase-2-only    # 只运行 Phase 2（需要CASA）
#   ./alma_project_level_7_generate_uvfits.sh --help            # 显示帮助信息
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="alma_project_level_7_generate_uvfits.py"
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Python script not found: $SCRIPT_PATH"
    exit 1
fi

# 显示帮助信息
show_help() {
    echo "ALMA Project Level 7 - Generate UVFITS from Level 6 Results"
    echo ""
    echo "用法："
    echo "  $0 [选项]"
    echo ""
    echo "选项："
    echo "  --phase-1-only    只运行 Phase 1（提取线信息，纯Python）"
    echo "  --phase-2-only    只运行 Phase 2（CASA处理，需要CASA环境）"
    echo "  --help            显示此帮助信息"
    echo ""
    echo "说明："
    echo "  Phase 1: 从 gaussian_fit_summary.txt 提取线心和FWHM"
    echo "           输出: line_info.csv"
    echo ""
    echo "  Phase 2: CASA处理 - uvcontsub, split, concat, exportuvfits"
    echo "           输入: line_info.csv"
    echo "           输出: uvfits_output/line.uvfits"
    echo ""
    echo "示例："
    echo "  cd /alma/high-z-qso/2019.1.01634.L"
    echo "  ${SCRIPT_DIR}/$(basename $0)"
    echo ""
}

# 解析命令行参数
PHASE_1_ONLY=0
PHASE_2_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase-1-only)
            PHASE_1_ONLY=1
            shift
            ;;
        --phase-2-only)
            PHASE_2_ONLY=1
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 获取当前工作目录作为项目目录
PROJECT_DIR=$(pwd)

echo "========================================================================"
echo "ALMA Project Level 7 - Generate UVFITS from Level 6 Results"
echo "========================================================================"
echo "Project directory: $PROJECT_DIR"
echo "Script path: $SCRIPT_PATH"
echo ""

# 检查必要文件
if [ ! -f "$PROJECT_DIR/target_line_list.csv" ]; then
    echo "ERROR: target_line_list.csv not found in current directory"
    echo "Please run this script in the project root directory"
    exit 1
fi

LOG_FILE="alma_level7_prepare_uvfits_for_gildas.log"
# 运行 Phase 1（纯 Python）
if [ $PHASE_2_ONLY -eq 0 ]; then
    echo "========================================================================"
    echo "Phase 1: Extracting line information (Pure Python)"
    echo "========================================================================"
    
    python "$SCRIPT_PATH" "$PROJECT_DIR" --phase 1 2>&1 | tee -a "${LOG_FILE}"
    
    PHASE1_STATUS=${PIPESTATUS[0]}
    
    if [ $PHASE1_STATUS -ne 0 ]; then
        echo ""
        echo "ERROR: Phase 1 failed with exit code $PHASE1_STATUS"
        echo "Check phase1_extract_line_info.log for details"
        exit 1
    fi
    
    echo ""
    echo "Phase 1 completed successfully!"
    echo "Log saved to: phase1_extract_line_info.log"
    echo ""
fi

# 如果只运行 Phase 1，到此结束
if [ $PHASE_1_ONLY -eq 1 ]; then
    echo "Phase 1 only mode - Done!"
    exit 0
fi

# 检查 line_info.csv 是否存在（Phase 2 需要）
if [ ! -f "$PROJECT_DIR/line_info.csv" ]; then
    echo "ERROR: line_info.csv not found"
    echo "Please run Phase 1 first!"
    exit 1
fi

# 运行 Phase 2（CASA 环境）
echo "========================================================================"
echo "Phase 2: CASA processing (uvcontsub + exportuvfits)"
echo "========================================================================"
echo ""
echo "Checking CASA availability..."

# CASA 路径 - 根据您的系统配置修改
# 方法1: 如果 casa 在 PATH 中
CASA_CMD="casa"

# 方法2: 如果需要指定完整路径，取消下面的注释并修改路径
# CASA_CMD="/home/wyx/Software/CASA/Portable/casa-6.2.1-7-pipeline-2021.2.0.128/bin/casa"

# 运行 CASA 脚本
PYTHONUNBUFFERED=1 stdbuf -oL ${CASA_CMD} --nologger --nogui -c "$SCRIPT_PATH" "$PROJECT_DIR" --phase 2 2>&1 | tee -a "${LOG_FILE}"

PHASE2_STATUS=${PIPESTATUS[0]}

if [ $PHASE2_STATUS -ne 0 ]; then
    echo ""
    echo "ERROR: Phase 2 failed with exit code $PHASE2_STATUS"
    exit 1
fi

echo ""
echo "========================================================================"
echo "All phases completed successfully!"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  - line_info.csv"
echo "  - uvfits_output/line.uvfits"
echo "========================================================================"
