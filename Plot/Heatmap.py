import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import re

# ==========================================
# 1. 布局超参数 (Layout Hyperparameters) 
# ==========================================
# 1️⃣ 外侧边缘留白 (单位: 英寸) 
# 作用: 控制最左侧的文字（如 'A'、'B' 和 'S'）距离图片最终物理边缘的空白距离，实现“稍微空出来一点”
EDGE_PADDING = 0.15 

# 2️⃣ A / B 标签的水平对齐偏移量 (基于子图宽度的比例)
# 作用: 负数表示向左偏移。你可以微调这个值（例如 -0.14 到 -0.16），使 A 和 B 刚好与最长的Y轴标签（如 'Sparse-encoding' 的 'S'）最左侧严格垂直对齐(平行)。
LABEL_X_OFFSET = -0.145

# 3️⃣ 上下子图的垂直间距
# 作用: 值越小，上下两图靠得越近 (推荐范围: 0.1 ~ 0.5)
H_SPACE = 0.3 


# ==========================================
# 2. 全局配置与路径设置
# ==========================================
# 字体与字号设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['font.size'] = 14  
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['xtick.labelsize'] = 14 
plt.rcParams['ytick.labelsize'] = 14 

# A/B 标签设置
PANEL_LABEL_FONTSIZE = 32

# 路径配置
base_path = '/root/autodl-tmp/data/Heatmap_data'
acc_file = 'ML_ACC.xlsx'
mcc_file = 'ML_MCC.xlsx'

# ==========================================
# 3. 辅助函数定义
# ==========================================
def parse_val(cell):
    """提取单元格中的数值部分"""
    if isinstance(cell, str):
        match = re.match(r"([0-9.-]+)", cell)
        if match: return float(match.group(1))
    return cell

def format_val(cell):
    """保留4位小数"""
    v = parse_val(cell)
    if isinstance(v, (int, float)): return f"{v:.4f}"
    return str(cell)

# ==========================================
# 4. 颜色设置 (应用学习到的配色)
# ==========================================
# 从浅(低值) -> 深(高值) 过渡的暖冷配色
base_colors = ['#F5DDB5', '#F5CCBC', '#F5B5BF', '#CEA2B5', '#9A9AB9', '#728AB9', '#56669E']

# 创建线性渐变色板
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_user", base_colors)

# ==========================================
# 5. 主程序逻辑
# ==========================================

# 任务列表
tasks = [
    {
        'title': 'ACC Heatmap', 
        'filename': acc_file,
        'sheet': 'ACC',
        'label': 'A'
    },
    {
        'title': 'MCC Heatmap', 
        'filename': mcc_file,
        'sheet': 'MCC',
        'label': 'B'
    }
]

# 创建一个上下两行的总画布 (单图高度为 10，两张就是 20)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 20))

# 循环绘图
for ax, task in zip(axes, tasks):
    title = task['title']
    sheet_name = task['sheet']
    filename = task['filename']
    label = task['label']
    
    fpath = os.path.join(base_path, filename)
    
    if not os.path.exists(fpath):
        print(f"⚠️ 文件未找到: {fpath}")
        continue
        
    try:
        # 读取 Excel
        try:
            df = pd.read_excel(fpath, sheet_name=sheet_name, index_col=0)
            print(f"已读取 {title} 的 Sheet: '{sheet_name}'")
        except ValueError:
            print(f"⚠️ 未找到 Sheet '{sheet_name}'，尝试读取默认 Sheet...")
            df = pd.read_excel(fpath, index_col=0)

        # 数据清洗与格式化
        # 将非数值内容转换为 NaN，确保绘图时不报错
        df_plot = df.applymap(parse_val).apply(pd.to_numeric, errors='coerce')
        # 准备用于显示的文本（保留原格式）
        df_annot = df.applymap(format_val)
        
        # 绘图 (指定在当前的 ax 上绘制)
        sns.heatmap(df_plot, 
                    annot=df_annot, 
                    fmt='', 
                    cmap=custom_cmap,  # 使用新定义的色板
                    linewidths=0.5, 
                    linecolor='white',
                    annot_kws={"size": 12, "weight": "normal"},
                    ax=ax) 
        
        # 设置轴标签样式
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight='bold')
        
        # 添加 A 和 B 标签
        # transform=ax.transAxes 确保坐标系与热力图绑定，LABEL_X_OFFSET 用来向左偏移对齐文字
        ax.text(LABEL_X_OFFSET, 1.02, label, transform=ax.transAxes, 
                fontsize=PANEL_LABEL_FONTSIZE, fontweight='bold', fontfamily='serif')
        
    except Exception as e:
        print(f"❌ 处理 {title} 失败: {e}")

# 自动调整整体布局 (仅控制内部间距)
plt.tight_layout(pad=1.0, h_pad=H_SPACE)

# 保存合并后的图像
# bbox_inches='tight' 会自动包裹包含文字在内的最外层边缘
# pad_inches=EDGE_PADDING 会在最外边缘再额外加上指定的安全留白，确保左边不会被一刀切死
save_name = "Combined_ACC_MCC_Heatmap.png"
plt.savefig(save_name, dpi=600, bbox_inches='tight', pad_inches=EDGE_PADDING)
print(f"✅ 已保存合并热力图: {save_name}")

# 关闭画布释放内存
plt.close()