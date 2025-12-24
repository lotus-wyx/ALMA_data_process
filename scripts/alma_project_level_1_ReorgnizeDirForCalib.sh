#!/bin/bash

# 设置基础目录
BASE_DIR="$(pwd)"
TARGET_DIR="${BASE_DIR}/Level_2_Calib"

# 创建目标目录
mkdir -p "${TARGET_DIR}"

# 计数器
counter=1

echo "=========================================="
echo "开始处理 member 目录"
echo "=========================================="
echo "目标目录: ${TARGET_DIR}"
echo ""

# 查找所有 member 目录（只匹配顶层，不包括子目录）并按路径排序
find "${BASE_DIR}" -type d -regex ".*/science[^/]*/group[^/]*/member\.uid[^/]*$" | sort | while read member_dir; do
    # 生成新的目录名（格式：DataSet_01, DataSet_02, ...）
    new_name=$(printf "DataSet_%02d" ${counter})
    target_path="${TARGET_DIR}/${new_name}"
    
    # 显示操作信息
    echo "[${counter}] 移动:"
    echo "  从: $(echo ${member_dir} | sed "s|${BASE_DIR}/||")"
    echo "  到: Level_2_Calib/${new_name}"
    
    # 执行移动操作
    if [ -d "${member_dir}" ]; then
        mv "${member_dir}" "${target_path}"
        if [ $? -eq 0 ]; then
            echo "  ✓ 成功"
        else
            echo "  ✗ 失败"
            exit 1
        fi
    else
        echo "  ✗ 源目录不存在"
        exit 1
    fi
    
    echo ""
    
    # 计数器加1
    counter=$((counter + 1))
done

echo "=========================================="
echo "处理完成！共处理 $((counter - 1)) 个数据集"
echo "=========================================="

# 显示目标目录的内容
echo ""
echo "Level_2_Calib 目录中的数据集 (前10个):"
ls -1 "${TARGET_DIR}" | head -10
total=$(ls -1 "${TARGET_DIR}" 2>/dev/null | wc -l)
if [ $total -gt 10 ]; then
    echo "..."
fi
echo ""
# 删除剩余的 science 开头的目录
echo ""
echo "=========================================="
echo "清理剩余的 science 目录"
echo "=========================================="

# 查找所有 science 开头的顶层目录
science_dirs=$(find "${BASE_DIR}" -maxdepth 1 -type d -name "science*" 2>/dev/null)

if [ -z "${science_dirs}" ]; then
    echo "没有找到需要删除的 science 目录"
else
    echo "找到以下 science 目录:"
    echo "${science_dirs}" | sed "s|${BASE_DIR}/||g"
    echo ""
    echo -n "是否删除这些目录? (y/N): "
    read -r response
    
    if [[ "${response}" =~ ^[Yy]$ ]]; then
        echo ""
        echo "${science_dirs}" | while read science_dir; do
            if [ -n "${science_dir}" ] && [ -d "${science_dir}" ]; then
                dir_name=$(basename "${science_dir}")
                echo "正在删除: ${dir_name}"
                rm -rf "${science_dir}"
                if [ $? -eq 0 ]; then
                    echo "  ✓ 成功删除"
                else
                    echo "  ✗ 删除失败"
                fi
            fi
        done
        echo ""
        echo "清理完成！"
    else
        echo "已取消删除操作"
    fi
fi

echo ""
echo "=========================================="
echo "所有操作完成"
echo "=========================================="