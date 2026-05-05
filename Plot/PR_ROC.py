import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# =============================================================================
# 1. 路径与全局配置
# =============================================================================
input_dir = '/root/autodl-tmp/Visualization/Test/'
output_dir = '/root/autodl-tmp/Visualization/Test'
os.makedirs(output_dir, exist_ok=True)

# 自动获取输入目录下所有的 .xlsx 文件
excel_files = [f for f in glob.glob(os.path.join(input_dir, "*.xlsx")) if not os.path.basename(f).startswith('~$')]

# 用户指定的 DNA & RNA 模型标准名称列表
dna_model_list = [
    'DNABERT-2', 'DNABERT', 'Agro-NT', 'Gena-LM', 
    'Nucleotide-Transformer', 'Genos-1.2B', 'HyenaDNA', 'Caduceus'
]

rna_model_list = [
    'RiNALMo-mega', 'RNABERT', 'UTR-LM', 'RNA-FM', 'RNA-MSM', 
    'SpliceBERT', 'RNA-Ernie', 'RESM', 'MP-RNA'
]

# 沿用的主题与色系：从浅色渐变到深色
base_colors = ['#F5DDB5','#F5CCBC','#F5B5BF', '#CEA2B5', '#9A9AB9', '#728AB9', '#56669E']
sns.set(style="ticks", context="talk")

# =============================================================================
# 2. 辅助函数
# =============================================================================
def identify_model(filename):
    """判断文件属于哪个阵营，并返回标准的模型名称"""
    fn_lower = str(filename).lower()
    for std_name in dna_model_list:
        if std_name.lower() in fn_lower:
            return 'DNA', std_name
    for std_name in rna_model_list:
        if std_name.lower() in fn_lower:
            return 'RNA', std_name
    return None, None

def extract_true_and_prob(df):
    """
    智能提取 DataFrame 中的真实标签列和预测概率列
    完全兼容以下三种情况:
    1. label, ensemble_prob_pos, ensemble_pred_label
    2. label, Prob_Class_0, Prob_Class_1, Final_Prediction
    3. label, ensemble_prob, final_prediction
    """
    label_col = None
    prob_col = None
    
    # 获取清理后的列名映射 (全部转小写并去除两端空格，方便精确匹配)
    col_map = {str(c).lower().strip(): c for c in df.columns}
    
    # 1. 找真实标签列
    for target in ['label', 'true_label', 'y_true', 'target']:
        if target in col_map:
            label_col = col_map[target]
            break
            
    # 2. 找预测概率列 (按优先级严格匹配正类概率)
    target_prob_names = [
        'ensemble_prob_pos',   # 情况 1
        'prob_class_1',        # 情况 2
        'ensemble_prob',       # 情况 3
        'prob', 'probability', 'pred_prob', 'score' # 备用选项
    ]
    
    for target in target_prob_names:
        if target in col_map:
            prob_col = col_map[target]
            break
            
    # 3. 终极退火机制：如果都没精确匹配到，找包含 'prob' 且不是 'class_0' 的列
    if prob_col is None:
        for c_lower, c_raw in col_map.items():
            if 'prob' in c_lower and 'class_0' not in c_lower:
                prob_col = c_raw
                break

    if label_col is not None and prob_col is not None:
        return df[label_col].values, df[prob_col].values
        
    return None, None

# =============================================================================
# 3. 数据读取与整理 (加入加密/损坏文件容错机制)
# =============================================================================
model_data = {'DNA': {}, 'RNA': {}}

if not excel_files:
    print(f"在 {input_dir} 目录下没有找到任何 .xlsx 文件，请检查路径。")
else:
    print(f"共找到 {len(excel_files)} 个 .xlsx 文件，开始匹配提取数据...\n")
    for f in excel_files:
        raw_name = os.path.basename(f).replace('.xlsx', '')
        m_type, std_name = identify_model(raw_name)
        
        if m_type is None:
            continue
            
        # 容错读取机制
        try:
            # 优先尝试用 openpyxl 读取（针对标准 .xlsx）
            df = pd.read_excel(f, engine='openpyxl')
        except Exception:
            try:
                # 如果报错，退化使用 xlrd 读取（针对伪装成 .xlsx 的老旧 .xls 文件）
                df = pd.read_excel(f, engine='xlrd')
            except Exception:
                # 如果依然报错（大概率是被加密或文件已损坏），直接跳过
                print(f"  [跳过] 无法读取文件 '{raw_name}' (可能被加密或格式损坏)。")
                continue
                
        y_true, y_prob = extract_true_and_prob(df)
        
        if y_true is not None and y_prob is not None:
            model_data[m_type][std_name] = {'y_true': y_true, 'y_prob': y_prob}
            print(f"  [成功] 提取 {m_type} 模型: {std_name} (数据量: {len(y_true)})")
        else:
            print(f"  [警告] 文件 {raw_name} 无法找到 label 或合适的 prob 列。")

# =============================================================================
# 4. 绘图核心函数
# =============================================================================
def plot_roc_and_pr(group_name, group_data):
    if not group_data:
        print(f"\n没有提取到任何 {group_name} 模型的数据，跳过绘图。")
        return

    # 先计算所有模型的 AUC 用于排序配色 (AUC 越低颜色越浅，越高颜色越深)
    model_auc = {}
    for model_name, data in group_data.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_prob'])
        model_auc[model_name] = auc(fpr, tpr)
        
    # 按 AUC 升序排序
    sorted_models = sorted(model_auc.keys(), key=lambda k: model_auc[k])
    num_models = len(sorted_models)
    
    # 动态生成颜色板
    custom_palette = sns.blend_palette(base_colors, n_colors=num_models)

    # ---------------------------
    # 1. 绘制并保存 ROC 曲线
    # ---------------------------
    plt.figure(figsize=(8, 8))
    for i, model_name in enumerate(sorted_models):
        y_true, y_prob = group_data[model_name]['y_true'], group_data[model_name]['y_prob']
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        # 图例显示 AUC
        plt.plot(fpr, tpr, color=custom_palette[i], lw=2.5, 
                 label=f'{model_name} (AUC = {model_auc[model_name]:.4f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--') # 随机对角线
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=16)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=16)
    # ROC 图例放右下角，不要标题
    plt.legend(loc="lower right", frameon=False, prop={'size': 13})
    sns.despine()
    plt.tight_layout()
    roc_out_path = os.path.join(output_dir, f'{group_name}_ROC_Curve.png')
    plt.savefig(roc_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"==> 成功生成: {roc_out_path}")

    # ---------------------------
    # 2. 绘制并保存 PR 曲线
    # ---------------------------
    plt.figure(figsize=(8, 8))
    for i, model_name in enumerate(sorted_models):
        y_true, y_prob = group_data[model_name]['y_true'], group_data[model_name]['y_prob']
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # 计算 AP (Average Precision) 作为 PR 的面积衡量
        ap = average_precision_score(y_true, y_prob)
        
        plt.plot(recall, precision, color=custom_palette[i], lw=2.5, 
                 label=f'{model_name} (AP = {ap:.4f})')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('Recall', fontweight='bold', fontsize=16)
    plt.ylabel('Precision', fontweight='bold', fontsize=16)
    # PR 图例放左下角，不要标题
    plt.legend(loc="lower left", frameon=False, prop={'size': 13})
    sns.despine()
    plt.tight_layout()
    pr_out_path = os.path.join(output_dir, f'{group_name}_PR_Curve.png')
    plt.savefig(pr_out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"==> 成功生成: {pr_out_path}")

# =============================================================================
# 5. 执行画图
# =============================================================================
print("\n--- 开始绘制图表 ---")
plot_roc_and_pr("DNA", model_data['DNA'])
plot_roc_and_pr("RNA", model_data['RNA'])
print("\n所有图表绘制完毕！")