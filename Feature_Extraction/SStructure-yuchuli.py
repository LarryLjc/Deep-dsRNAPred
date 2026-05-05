import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import multiprocessing

# =============================================================================
# 1. 路径配置
# =============================================================================
TRAIN_FASTA = "/root/autodl-tmp/data/train_RNA.fasta"
TEST_FASTA = "/root/autodl-tmp/data/test_RNA.fasta"

TRAIN_OUT_EXCEL = "/root/autodl-tmp/data/train_SSfeatures.xlsx"
TEST_OUT_EXCEL = "/root/autodl-tmp/data/test_SSfeatures.xlsx"

# =============================================================================
# 2. R 语言提取函数初始化 (加入防爆内存和防断连机制)
# =============================================================================
robjects.r('''
    extract_batch_SSfeatures <- function(fastapath, cores){
        # 兜底机制：无论函数是正常结束还是报错退出，都强制清理所有 Socket 连接
        on.exit({
            closeAllConnections()
            gc()
        })
        
        suppressMessages(suppressWarnings(library(LncFinder)))
        demo_DNA.seq <- suppressMessages(seqinr::read.fasta(fastapath))
        
        # 🚀 阶段 1: 多核并发跑 RNAfold
        Seqs <- suppressMessages(LncFinder::run_RNAfold(demo_DNA.seq, RNAfold.path = "RNAfold", parallel.cores = cores))
        
        # ⚠️ 关键修复：主动切断 RNAfold 遗留的 20+ 个 Socket 连接，释放系统资源
        closeAllConnections()
        gc()
        
        # 🚀 阶段 2: 多核并发提取特征
        result_2 <- suppressMessages(LncFinder::extract_features(Seqs, label = NULL, SS.features = TRUE, format = "SS", frequencies.file = "human", parallel.cores = cores))
        
        res2 <- result_2[,c(12:19)]
        return(res2)
    }
''')

FEATURE_COLUMNS = [
    "SLDLD: Structural logarithm distance to lncRNA of acguD", 
    "SLDPD: Structural logarithm distance to pcRNA of acguD", 
    "SLDRD: Structural logarithm distance acguD ratio", 
    "SLDLN: Structural logarithm distance to lncRNA of acguACGU", 
    "SLDPN: Structural logarithm distance to pcRNA of acguACGU", 
    "SLDRN: Structural logarithm distance acguACGU ratio",
    "SDMFE: Secondary structural minimum free energy", 
    "SFPUS: Secondary structural UP frequency paired-unpaired"
]

# =============================================================================
# 3. 核心处理逻辑
# =============================================================================
def process_fasta_batch(input_fasta, output_excel):
    """批量清洗序列，生成临时大文件，交由 R 多核全速并行处理"""
    
    total_cores = multiprocessing.cpu_count()
    # 为了系统稳定，25核的机器最多分配 20~22 核给 R，避免触发系统的 OOM (内存溢出) 杀手
    use_cores = min(total_cores - 2, 22)  
    
    print(f"\n[{input_fasta}]")
    print(f"🔥 检测到 {total_cores} 个 CPU 核心，已分配 {use_cores} 核投入战斗！")
    
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total_seqs = len(records)
    print(f"共检测到 {total_seqs} 条序列，开始进行 T->U 批量转换...")
    
    cleaned_records = []
    seq_ids = []
    for record in records:
        seq_str = str(record.seq).upper().replace('T', 'U')
        cleaned_records.append(SeqRecord(Seq(seq_str), id=record.id, description=""))
        seq_ids.append(record.id)
        
    temp_fasta = "temp_batch_" + os.path.basename(input_fasta)
    SeqIO.write(cleaned_records, temp_fasta, "fasta")
    
    print(f"转换完成。R 语言计算引擎全速运转中 (并行线程数: {use_cores})，请耐心等待...")
    
    try:
        with localconverter(robjects.default_converter + pandas2ri.converter):
            sstruc_r = robjects.r['extract_batch_SSfeatures'](temp_fasta, use_cores)
            
            sstruc_r.columns = FEATURE_COLUMNS
            sstruc_r.insert(0, 'ID', seq_ids) 
            
            print(f"✅ 特征提取成功！正在保存至 {output_excel} ...")
            sstruc_r.to_excel(output_excel, index=False)
            
    except Exception as e:
        print(f"❌ 批量提取失败! 原因: {e}")
    finally:
        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)

# =============================================================================
# 4. 主执行流程
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print(" 阶段一：处理训练集 (Train Data)")
    print("="*60)
    process_fasta_batch(TRAIN_FASTA, TRAIN_OUT_EXCEL)

    print("="*60)
    print(" 阶段二：处理测试集 (Test Data)")
    print("="*60)
    process_fasta_batch(TEST_FASTA, TEST_OUT_EXCEL)
    
    print("\n🎉 算力释放完毕！所有流程执行成功，可以开始进行 PyTorch 模型训练了。")