#!/bin/bash

# 脚本名称：alma_project_level_4_ConcatMS.sh
# 功能：在当前目录下通过 CASA 执行 alma_project_level_4_ConcatMS.py

# 获取脚本所在目录（Python 脚本的位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/alma_project_level_4_ConcatMS.py"

# 获取当前工作目录（用户调用脚本的位置）
WORK_DIR="$(pwd)"

# 检查 Python 脚本是否存在
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误: 找不到 Python 脚本: ${PYTHON_SCRIPT}"
    exit 1
fi

# 检查是否在正确的工作目录（可选）
if [ ! -d "Each_target_img" ]; then
    echo "警告: 当前目录没有 Each_target_img 目录"
    echo "当前工作目录: ${WORK_DIR}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# CASA 路径 - 根据您的系统配置修改
CASA_CMD="/home/wyx/Software/CASA/Portable/casa-6.2.1-7-pipeline-2021.2.0.128/bin/casa" # "casa" 
echo "=================================================="
echo "ALMA Project Level 4: Concat MS files"
echo "=================================================="
echo "工作目录: ${WORK_DIR}"
echo "Python 脚本: ${PYTHON_SCRIPT}"
echo "=================================================="
echo ""

# 在当前目录下运行 CASA
cd "${WORK_DIR}"

# 生成日志文件名
LOG_FILE="alma_level4_concat.log"

echo "日志将保存到: ${LOG_FILE}"
echo ""
# 执行 CASA 并运行 Python 脚本
stdbuf -oL ${CASA_CMD} --nologger --nogui --nologfile -c "${PYTHON_SCRIPT}" 2>&1 | tee "${LOG_FILE}"

# 检查执行结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "脚本执行完成"
    echo "日志已保存到: ${LOG_FILE}"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "脚本执行失败"
    echo "日志已保存到: ${LOG_FILE}"
    echo "=================================================="
    exit 1
fi