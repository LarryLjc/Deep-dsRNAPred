import os
import pandas as pd
import subprocess
import shutil

# ================= 配置路径 =================
# 【修改】输入路径改为 FASTA 文件
TRAIN_FASTA = "/root/autodl-tmp/data/train_RNA.fasta"
TEST_FASTA = "/root/autodl-tmp/data/test_RNA.fasta"

# 提取完成后，为了方便 PyTorch 读取，依然保存为 Excel
NEW_TRAIN_EXCEL = "/root/autodl-tmp/data/train_RNA_with_struct.xlsx"
NEW_TEST_EXCEL = "/root/autodl-tmp/data/test_RNA_with_struct.xlsx"

# 外部软件路径
RNAFOLD_PATH = "RNAfold"
BPRNA_PATH = "/root/autodl-tmp/Model/Twelve_Feature_Extraction_Methods/bpRNA.pl"
PERL_PATH = "/usr/bin/perl"


TEMP_DIR = os.path.abspath("./temp_rna_processing")
# ============================================

def read_fasta(file_path):
    """自定义 FASTA 读取器，返回 (ID, Sequence) 列表"""
    sequences = []
    with open(file_path, 'r') as f:
        seq_id = ""
        seq_data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    sequences.append((seq_id, "".join(seq_data)))
                seq_id = line[1:] # 移除 '>' 符号
                seq_data = []
            else:
                seq_data.append(line)
        # 把最后一条序列加进去
        if seq_id:
            sequences.append((seq_id, "".join(seq_data)))
    return sequences

def process_sequence(seq, seq_name):
    """带详细报错输出的诊断版"""
    dbn_file = os.path.join(TEMP_DIR, f"{seq_name}.dbn")
    st_file = os.path.join(TEMP_DIR, f"{seq_name}.st")
    structure_string = ""
    
    try:
        # 1. 调用 RNAfold
        cmd_fold = [RNAFOLD_PATH, "--noPS"]
        process_fold = subprocess.run(cmd_fold, input=seq, text=True, capture_output=True, timeout=30)
        
        if process_fold.returncode != 0:
            print(f"\n[DEBUG] {seq_name} RNAfold 运行失败! 报错信息: {process_fold.stderr}")
            return ""
            
        if not process_fold.stdout.strip():
            print(f"\n[DEBUG] {seq_name} RNAfold 没有输出任何内容!")
            return ""
            
        lines = process_fold.stdout.strip().split('\n')
        if len(lines) < 2:
            print(f"\n[DEBUG] {seq_name} RNAfold 输出格式不对: {process_fold.stdout}")
            return ""
            
        dot_bracket = lines[1].split(' ')[0]
        
        # 2. 写入 dbn 文件
        with open(dbn_file, 'w') as f:
            f.write(f">{seq_name}\n{seq}\n{dot_bracket}\n")
            
        # 3. 调用 bpRNA.pl
        cmd_bprna = [PERL_PATH, BPRNA_PATH, dbn_file]
        process_bprna = subprocess.run(cmd_bprna, cwd=TEMP_DIR, capture_output=True, text=True, timeout=30)
        
        if process_bprna.returncode != 0:
            print(f"\n[DEBUG] {seq_name} bpRNA 运行失败! 报错: {process_bprna.stderr}")
            return ""
        
        # 4. 解析 .st 文件
        if os.path.exists(st_file):
            with open(st_file, 'r') as f:
                st_lines = f.readlines()
                if len(st_lines) >= 6:
                    structure_string = st_lines[5].strip()
                else:
                    print(f"\n[DEBUG] {seq_name} .st 文件内容太短: {st_lines}")
        else:
            print(f"\n[DEBUG] {seq_name} bpRNA 没有生成 .st 文件!")
                    
    except subprocess.TimeoutExpired:
        print(f"\n[DEBUG] {seq_name} 处理超时 (>30秒)!")
    except Exception as e:
        print(f"\n[DEBUG] {seq_name} 发生未知错误: {str(e)}")
    finally:
        # 清理临时文件
        if os.path.exists(dbn_file): os.remove(dbn_file)
        if os.path.exists(st_file):  os.remove(st_file)
            
    return structure_string

def process_fasta_to_excel(input_fasta, output_excel):
    print(f"\n=============================================")
    print(f"Start processing: {input_fasta}")
    print(f"=============================================")
    
    if not os.path.exists(input_fasta):
        print(f"File not found: {input_fasta}. Skipping...")
        return
        
    fasta_data = read_fasta(input_fasta)
    print(f"Successfully loaded {len(fasta_data)} sequences from FASTA.\n")
    
    records = []
    failed_count = 0
    
    for idx, (seq_id, seq) in enumerate(fasta_data):
        # 【新增需求 1】: 将所有的 T 转换为 U
        seq = seq.upper().replace('T', 'U')
        
        # 为了防止 FASTA 里的 ID 有奇怪的特殊字符导致文件生成失败，我们生成一个安全的临时名字
        safe_seq_name = f"seq_{idx}"
        
        # 提取结构
        struct_str = process_sequence(seq, safe_seq_name)
        
        if struct_str:
            # 【新增需求 2】: 成功则打印
            print(f"[{idx+1}/{len(fasta_data)}] Success | ID: {seq_id} | Length: {len(seq)}")
            # 保存到记录中，注意这里的列名，以便你的 PyTorch 脚本能正确读取
            records.append({
                'ID': seq_id,
                'Sequence': seq,       # 保存 T转U 后的纯序列
                'Structure': struct_str # 保存 7维结构序列
            })
        else:
            print(f"[{idx+1}/{len(fasta_data)}] FAILED  | ID: {seq_id} (Skipped)")
            failed_count += 1
            
    # 转换为 DataFrame 并保存为 Excel
    df = pd.DataFrame(records)
    df.to_excel(output_excel, index=False)
    
    print(f"\nSaved structured data to {output_excel}.")
    print(f"Total: {len(fasta_data)} | Success: {len(df)} | Failed/Skipped: {failed_count}")

if __name__ == "__main__":
    # 创建临时文件夹
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
        
    # 分别处理训练集和测试集的 FASTA (自动存为 Excel)
    process_fasta_to_excel(TRAIN_FASTA, NEW_TRAIN_EXCEL)
    process_fasta_to_excel(TEST_FASTA, NEW_TEST_EXCEL)
    
    # 清理临时文件夹
    shutil.rmtree(TEMP_DIR)
    print("\nAll preprocessing finished successfully!")