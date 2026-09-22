#!/bin/bash

# 设置工作目录为当前路径下的 size_gildas/test_log
WORK_DIR="$(pwd)/size_gildas/test_log"
OUTPUT_CSV="$(pwd)/size_gildas/line_size_results.csv"

# 检查目录是否存在
if [ ! -d "$WORK_DIR" ]; then
    echo "Error: Directory $WORK_DIR does not exist"
    exit 1
fi

# 创建或清空CSV文件，写入表头
echo "src_name,line_flux,e_line_flux,line_flux_unit,CII_maj,e_CII_maj,CII_min,e_CII_min,CII_nu,e_CII_nu" > "$OUTPUT_CSV"

# 进入工作目录
cd "$WORK_DIR" || exit 1

# 按字母顺序遍历所有.gildas文件
for filename in $(ls *.gildas 2>/dev/null | sort); do
    # 提取基础源名称（去掉 line_result_ 前缀和 .gildas 后缀）
    base_name="${filename#line_result_}"
    base_name="${base_name%.gildas}"
    
    # 计数器，用于区分同一文件中的多组数据
    count=0
    
    # 初始化变量
    flux="" e_flux="" unit=""
    maj="" e_maj=""
    minor="" e_minor=""
    nu="" e_nu=""
    
    # 读取文件内容并解析
    while IFS= read -r line; do
        if [[ "$line" =~ ^\ E_SPERGEL\ Flux ]]; then
            # 如果已有完整数据，先写入CSV（说明开始新的一组）
            if [ -n "$flux" ] && [ -n "$maj" ] && [ -n "$minor" ] && [ -n "$nu" ]; then
                count=$((count + 1))
                name="${base_name}_${count}"
                echo "$name,$flux,$e_flux,$unit,$maj,$e_maj,$minor,$e_minor,$nu,$e_nu" >> "$OUTPUT_CSV"
                echo "Processed: $filename -> $name"
                # 重置变量
                flux="" e_flux="" unit=""
                maj="" e_maj=""
                minor="" e_minor=""
                nu="" e_nu=""
            fi
            
            # 提取 flux, e_flux, unit
            value_part="${line#*=}"
            flux=$(echo "$value_part" | awk '{print $1}')
            e_flux=$(echo "$value_part" | sed 's/.*(\([^)]*\)).*/\1/' | awk '{print $1}')
            unit=$(echo "$value_part" | sed 's/.*) *//')
        elif [[ "$line" =~ ^\ E_SPERGEL\ Maj\.H\.L\.R ]]; then
            value_part="${line#*=}"
            maj=$(echo "$value_part" | awk '{print $1}')
            e_maj=$(echo "$value_part" | sed 's/.*(\([^)]*\)).*/\1/')
        elif [[ "$line" =~ ^\ E_SPERGEL\ Min\.H\.L\.R ]]; then
            value_part="${line#*=}"
            minor=$(echo "$value_part" | awk '{print $1}')
            e_minor=$(echo "$value_part" | sed 's/.*(\([^)]*\)).*/\1/')
        elif [[ "$line" =~ ^\ E_SPERGEL\ nu ]]; then
            value_part="${line#*=}"
            nu=$(echo "$value_part" | awk '{print $1}')
            e_nu=$(echo "$value_part" | sed 's/.*(\([^)]*\)).*/\1/')
        fi
    done < "$filename"
    
    # 写入最后一组数据
    if [ -n "$flux" ] && [ -n "$maj" ] && [ -n "$minor" ] && [ -n "$nu" ]; then
        count=$((count + 1))
        if [ $count -eq 1 ]; then
            name="$base_name"
        else
            name="${base_name}_${count}"
        fi
        echo "$name,$flux,$e_flux,$unit,$maj,$e_maj,$minor,$e_minor,$nu,$e_nu" >> "$OUTPUT_CSV"
        echo "Processed: $filename -> $name"
    fi
done

echo "Done! Results saved to $OUTPUT_CSV"