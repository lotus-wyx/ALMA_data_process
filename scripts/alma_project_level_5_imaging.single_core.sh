#!/bin/bash

# 脚本名称：alma_project_level_5_imaging.sh
# 功能：在当前目录下通过 CASA 执行 alma_project_level_5_imaging.py

# 获取脚本所在目录（Python 脚本的位置）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/alma_project_level_5_imaging.py"

# 获取当前工作目录（用户调用脚本的位置）
WORK_DIR="$(pwd)"

# 检查 Python 脚本是否存在
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "错误: 找不到 Python 脚本: ${PYTHON_SCRIPT}"
    exit 1
fi

# 检查是否在正确的工作目录（可选）
if [ ! -d "Each_target_img" ]; then
    echo "警告: 当前目录没有 Each_target_img 文件夹"
    echo "当前工作目录: ${WORK_DIR}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# CASA 路径 - 根据您的系统配置修改
# 方法1: 如果 casa 在 PATH 中
CASA_CMD="casa"

# 方法2: 如果需要指定完整路径，取消下面的注释并修改路径
# CASA_CMD="/home/wyx/Software/CASA/Portable/casa-6.x.x/bin/casa"

echo "=================================================="
echo "ALMA Project Level 5: Imaging with tclean"
echo "=================================================="
echo "工作目录: ${WORK_DIR}"
echo "Python 脚本: ${PYTHON_SCRIPT}"
echo "=================================================="
echo ""

# 在当前目录下运行 CASA
cd "${WORK_DIR}"

# 生成日志文件名（带时间戳）
LOG_FILE="alma_imaging_$(date +%Y%m%d_%H%M%S).log"

echo "日志将保存到: ${LOG_FILE}"
echo ""

# 执行 CASA 并运行 Python 脚本，同时输出到终端和日志文件
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
