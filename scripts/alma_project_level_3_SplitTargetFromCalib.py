import os
import glob

# ================= Configuration =================
# 当前工作目录
base_dir = os.getcwd()
# 目标源列表文件
TARGET_LIST_FILE = 'target_list.txt'
# 输入目录结构
INPUT_DIR_PATTERN = 'Level_2_Calib/DataSet_*/calibrated'
# 输出根目录
OUTPUT_ROOT_DIR = 'Each_target_img'
# =================================================

def get_target_names(filename):
    """读取目标源列表文件"""
    if not os.path.exists(filename):
        print("Error: Target list file '{}' not found.".format(filename))
        return []
    
    targets = []
    with open(filename, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('#'):
                targets.append(name)
    return targets

def parse_listobs_for_targets(listfile, target_list):
    """
    解析 listobs 输出文件，查找包含的目标源。
    返回该 MS 文件中包含的所有目标源名称列表。
    """
    found_targets = []
    
    if not os.path.exists(listfile):
        return found_targets

    in_fields_section = False
    name_start = None
    name_end = None
    header_line = None
    
    with open(listfile, 'r') as f:
        for line in f:
            # 查找 Fields 部分的开始
            if line.startswith('Fields:'):
                in_fields_section = True
                continue
            
            # 查找 Fields 部分的结束
            if 'Spectral Windows:' in line:
                in_fields_section = False
                break
            
            if in_fields_section:
                # 找到列标题行（包含 "Name" 和 "RA"）
                if header_line is None and 'Name' in line and 'RA' in line:
                    header_line = line
                    # 找到 Name 列的起始位置
                    name_start = line.find('Name')
                    # 找到 RA 列的起始位置作为 Name 列的结束位置
                    name_end = line.find('RA')
                    
                    if name_start == -1 or name_end == -1:
                        print("Warning: Could not find Name or SpwId column positions")
                        break
                    
                    print("Found Name column at position {} to {}".format(name_start, name_end))
                    continue
                
                # 如果已经找到列位置，开始解析数据行
                if name_start is not None and name_end is not None:
                    # 确保行足够长
                    if len(line) > name_end:
                        # 提取 Name 列的内容
                        name_field = line[name_start:name_end].strip()
                        
                        if name_field:
                            is_truncated = name_field.endswith('*')
                            if is_truncated:
                                # 去掉末尾的 *
                                name_field = name_field[:-1].strip()
                            for target in target_list:
                                matched = False
                                
                                if is_truncated:
                                    # 如果名称被截断，检查目标源是否以这个截断的名称开头
                                    if target.startswith(name_field):
                                        matched = True
                                else:
                                    # 如果名称完整，使用精确匹配
                                    if target == name_field:
                                        matched = True
                                
                                if matched and target not in found_targets:
                                    found_targets.append(target)
                                    print("  Found target: {} (matched with '{}')".format(target, name_field + ('*' if is_truncated else '')))
                    
    return found_targets

def main():
    # 1. 读取目标源列表
    targets = get_target_names(TARGET_LIST_FILE)
    if not targets:
        print("No targets found or list file missing. Exiting.")
        return

    print("Targets to look for: {}".format(targets))

    # 2. 查找所有 DataSet 文件夹
    dataset_dirs = sorted(glob.glob(INPUT_DIR_PATTERN))
    
    if not dataset_dirs:
        print("No directories found matching '{}'".format(INPUT_DIR_PATTERN))
        return

    # 确保输出根目录存在
    if not os.path.exists(OUTPUT_ROOT_DIR):
        os.makedirs(OUTPUT_ROOT_DIR)

    # 3. 遍历每个 DataSet
    for dataset_path in dataset_dirs:
        dataset_name = os.path.basename(os.path.dirname(dataset_path)) # e.g., DataSet_01
        if not dataset_name.startswith('DataSet'):
            print("Warning: Folder name '{}' does not start with 'DataSet'. Skipping.".format(dataset_name))
            continue
        ms_file = os.path.join(dataset_path, 'calibrated.ms')
        listobs_file = os.path.join(dataset_path, 'calibrated.listobs')

        if not os.path.exists(ms_file):
            print("Skipping {}: calibrated.ms not found.".format(dataset_name))
            continue

        print("-" * 40)
        print("Processing: {}".format(dataset_name))

        # 3.1 运行 listobs
        # 注意: 在 CASA 环境中，listobs 是内置函数
        try:
            print("Running listobs for {}...".format(ms_file))
            listobs(vis=ms_file, listfile=listobs_file, overwrite=True)
        except Exception as e:
            print("Error running listobs: {}".format(e))
            continue

        # 3.2 解析 listobs 文件，找出包含的目标源
        matched_targets = parse_listobs_for_targets(listobs_file, targets)
        
        if not matched_targets:
            print("No target sources found in {}.".format(dataset_name))
            continue
        
        print("Found targets in {}: {}".format(dataset_name, matched_targets))

        # 3.3 对每个匹配的目标源运行 split
        for target in matched_targets:
            # 创建目标源特定的输出目录
            target_out_dir = os.path.join(OUTPUT_ROOT_DIR, target)
            if not os.path.exists(target_out_dir):
                os.makedirs(target_out_dir)
            
            output_vis = os.path.join(target_out_dir, dataset_name + '.ms')
            
            # 检查输出文件是否已存在，避免覆盖或报错
            if os.path.exists(output_vis):
                print("Output file {} already exists. Skipping split.".format(output_vis))
                continue

            print("Splitting target '{}' into {}...".format(target, output_vis))
            
            try:
                # 运行 split
                # field=target 告诉 split 只提取该目标源的数据
                split(vis=ms_file, 
                      outputvis=output_vis, 
                      field=target, 
                      datacolumn='data')
                print("Split finished for {}.".format(target))
            except Exception as e:
                print("Error running split for {}: {}".format(target, e))

    print("-" * 40)
    print("All processing complete.")

# 运行主函数
if __name__ == "__main__":
    main()