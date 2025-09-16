import os
import subprocess
import argparse


def convert_dicom_to_nii(dicom_dir, output_dir, compress=True, keep_metadata=True, custom_name=None):
    """
    单次转换函数
    :param dicom_dir: DICOM文件目录路径
    :param output_dir: 输出目录路径
    :param compress: 是否启用GZIP压缩
    :param keep_metadata: 是否保存JSON元数据
    :param custom_name: 自定义输出文件名模板
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建基础命令
    cmd = ['dcm2niix']
    
    # 添加参数选项
    if compress:
        cmd.append('-z y')  # 启用GZIP压缩
    else:
        cmd.append('-z n')
    
    if keep_metadata:
        cmd.append('-b y')  # 保存JSON元数据
    else:
        cmd.append('-b n')
    
    # 文件名模板设置
    if custom_name:
        cmd.append(f'-f {custom_name}')
    else:
        # 默认命名：患者ID+序列号+随机字符串
        cmd.append('-f %p_%s_%t')
    
    # 添加输出目录和输入目录
    cmd.extend(['-o', f'"{output_dir}"', f'"{dicom_dir}"'])
    
    # 执行命令
    try:
        result = subprocess.run(' '.join(cmd), shell=True, check=True, capture_output=True, text=True)
        print(f"✅ 转换成功: {dicom_dir} -> {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 转换失败: {dicom_dir}\n错误信息: {e.stderr}")
        return False

def batch_convert_dicom(root_dir, output_base_dir, recursive=True):
    """
    批量转换主函数
    :param root_dir: 包含DICOM文件夹的根目录
    :param output_base_dir: 输出根目录
    :param recursive: 是否递归处理子目录
    """
    total_count = 0
    success_count = 0
    
    # 遍历目录结构
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录是否包含DICOM文件
        dicom_files = [f for f in files if f.lower().endswith('.dcm')]
        
        if dicom_files:
            total_count += 1
            # 创建对应的输出子目录
            relative_path = os.path.relpath(root, root_dir)
            # output_subdir = os.path.join(output_base_dir, relative_path)
            names = relative_path.split('/')
            custom_name = os.path.join(output_base_dir, names[0] + '_' + names[2] + '_' + names[3] + '.nii.gz')
            
            # 执行转换
            if convert_dicom_to_nii(root, output_base_dir, custom_name=custom_name):
                success_count += 1
        
        # 如果不递归处理，只处理顶级目录
        if not recursive:
            break
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"转换完成! 总计: {total_count}个目录, 成功: {success_count}个, 失败: {total_count - success_count}个")
    print(f"输出目录: {output_base_dir}")
    print(f"注意事项:")
    print("- 转换后的NIfTI文件可能包含同名的.json元数据文件")
    print("- 文件名格式默认为[患者ID]_[序列号]_[时间戳]")
    print("- 使用MRIcroGL或ITK-SNAP可验证文件方向一致性")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='DICOM到NIfTI批量转换工具')
    parser.add_argument('input_dir', help='包含DICOM文件夹的根目录')
    parser.add_argument('output_dir', help='NIfTI输出根目录')
    parser.add_argument('--no-recurse', action='store_true', help='仅处理顶级目录')
    parser.add_argument('--no-compress', action='store_true', help='禁用GZIP压缩')
    parser.add_argument('--no-json', action='store_true', help='不保存JSON元数据')
    parser.add_argument('--name', help='自定义文件名模板，例如 "patient_%%p_series_%%s"')
    
    args = parser.parse_args()
    
    # 执行批处理
    batch_convert_dicom(
        root_dir=args.input_dir,
        output_base_dir=args.output_dir,
        recursive=not args.no_recurse,
    )