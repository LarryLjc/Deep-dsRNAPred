import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import logging
import warnings
import gc
from collections import OrderedDict 
from torch.utils.data import Dataset, DataLoader

# --- 引入 UMAP 和归一化 ---
import umap
from sklearn.preprocessing import normalize

# --- 引入 Tokenizer 和原生的 AutoModel ---
from multimolecule import RnaTokenizer
from transformers import AutoModel, logging as hf_logging

# =============================================================================
# 全局配色设置 (方便修改)
# =============================================================================
COLOR_NON_DSRNA = '#F5CCBC'  # non-dsRNAs (Label 0) 的颜色
COLOR_DSRNA = '#728AB9'      # dsRNA (Label 1) 的颜色

# =============================================================================
# 全局路径与超参数配置
# =============================================================================
MODEL_PTH_PATH = '/root/autodl-tmp/Model/stage1model_fold_1_best.pth'
PCA_PKL_PATH = '/root/autodl-tmp/Model/pca_model_fold_1.pkl'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
RINALMO_PATH = r'/root/autodl-tmp/Big_Model/rinalmo-mega'

# --- 数据与采样超参数 ---
# 注意：这里必须与你训练日志里输出的 "Max Sequence Length" 保持绝对一致！
MAX_SEQ_LENGTH = 617      
BATCH_SIZE = 64
MAX_TSNE_SAMPLES = 2000   

# --- 模型架构超参数 ---
INPUT_SIZE = 50           
CNN_LAYERS = 3            
CNN_DIMS = 512            
POOL_SIZE = 2             
CBAM_LAYERS = 2           
LSTM_LAYERS = 2           
LSTM_HIDDEN_SIZE = 128    
FC_LAYERS = 3             
FC_DIMS = 64              
DROPOUT_RATE = 0.2        
NUM_CLASSES = 2           

# --- UMAP 与绘图超参数 ---
UMAP_N_NEIGHBORS = 30     
UMAP_MIN_DIST = 0.3       
UMAP_RANDOM_STATE = 42
PLOT_MARKER_SIZE = 40     
PLOT_ALPHA = 0.8          

# =============================================================================
# 基础环境设置
# =============================================================================
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 核心新增：强行解决大模型权重不匹配 (从你的训练脚本完美移植)
# =============================================================================
def fix_and_load_weights(model, model_path):
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safe_path = os.path.join(model_path, "model.safetensors")

    weight_file = None
    if os.path.exists(bin_path):
        weight_file = bin_path
    elif os.path.exists(safe_path):
        weight_file = safe_path

    if weight_file is None:
        print("[Warning] Directory does not contain pytorch_model.bin or model.safetensors.")
        return

    file_size_mb = os.path.getsize(weight_file) / (1024 * 1024)
    if file_size_mb < 1.0:
        raise ValueError(f"权重文件异常小 ({file_size_mb:.3f} MB)，Git LFS 未生效！")

    print(f"Valid model weight found ({file_size_mb:.1f} MB). Aligning dictionary keys...")
    if weight_file.endswith(".bin"):
        state_dict = torch.load(weight_file, map_location="cpu")
    else:
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["rinalmo_model.", "rinalmo.", "roberta.", "bert.", "model.", "base_model."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    core_missing = [k for k in missing if 'encoder' in k]
    if len(core_missing) == 0:
        print("Success: All core encoder weights successfully aligned and loaded!\n")
    else:
        print(f"Warning: Still missing {len(core_missing)} core weights.")

# =============================================================================
# 分类器架构定义 (保持不变)
# =============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        return self.sigmoid(max_out + avg_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        return self.sigmoid(self.conv(result))

class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 7, 9], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channel, channel, kernel_size=(k, 1), padding=(k // 2, 0), groups=group)),
                ('bn', nn.BatchNorm2d(channel)),
                ('relu', nn.ReLU())
            ])))
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0) 
        U = sum(conv_outs) 
        S = U.mean(-1).mean(-1) 
        Z = self.fc(S) 
        weights = [fc(Z).view(bs, c, 1, 1) for fc in self.fcs] 
        attention_weights = self.softmax(torch.stack(weights, 0))
        return (attention_weights * feats).sum(0)

class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernels=[3], pool_size=2):
        super(CNNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), padding=0, bias=True)
        self.sk_attention = SKAttention(channel=out_planes, kernels=kernels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1))
        nn.init.kaiming_normal_(self.conv1x1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1x1.bias, 0.)
    def forward(self, x):
        x = self.conv1x1(x)      
        x = self.sk_attention(x) 
        x = self.relu(x)        
        return self.pool(x)

class Deep_dsRNAPred(nn.Module):
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=512, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=128, lstm_layers=2, cbam_layers=2): 
        super(Deep_dsRNAPred, self).__init__()
        
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.pool_size = pool_size
        self.cnn_layers = cnn_layers
        self.max_seq_length = max_seq_length
        self.cbam_layers = cbam_layers 
        
        for i in range(cnn_layers):
            if i == 0: kernels = [3]
            elif i == 1: kernels = [7,9]
            else: kernels = [7,9]
            self.cnn_blocks.append(CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size))
            in_planes = cnn_dims

        after_cnn_length = max_seq_length
        for _ in range(cnn_layers):
            after_cnn_length = after_cnn_length // pool_size
        self.after_cnn_length = after_cnn_length
        self.cnn_dims = cnn_dims

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(self.cbam_layers)])

        self.bilstm = nn.LSTM(
            input_size=cnn_dims,            
            hidden_size=lstm_hidden_size,  
            num_layers=lstm_layers,         
            bidirectional=True,             
            batch_first=True,               
            dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        self.lstm_output_dim = lstm_hidden_size * 2

        self.lstm_flatten_dim = self.lstm_output_dim * self.after_cnn_length
        self.fc_blocks = nn.ModuleList()
        in_features = self.lstm_flatten_dim
        for _ in range(num_layers):
            fc = nn.Linear(in_features=in_features, out_features=num_dims)
            dropout = nn.Dropout(p=dropout_rate)
            self.fc_blocks.append(nn.Sequential(fc, nn.ReLU(), dropout))
            in_features = num_dims

        self.mid_fc = nn.Linear(num_dims, 128)
        self.mid_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        features_dict = {}
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        
        for block in self.cnn_blocks:
            x = block(x) 
        
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x) 
            
        if return_features:
            features_dict['cnn_cbam'] = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            
        x = x.squeeze(-1).permute(0, 2, 1) 
        lstm_out, _ = self.bilstm(x) 
        
        if return_features:
            features_dict['lstm'] = lstm_out.mean(dim=1) 
            
        x = lstm_out.flatten(start_dim=1) 
        
        for block in self.fc_blocks:
            x = block(x)
            
        x_mid = F.relu(self.mid_fc(x))
        if return_features:
            features_dict['final'] = x_mid
            
        x = self.mid_dropout(x_mid)
        logits = self.output_layer(x)
        
        if return_features:
            return logits, features_dict
        return logits

# =============================================================================
# 数据提取函数 (修复了关键的大模型权重加载逻辑)
# =============================================================================
def extract_RiNALMo_features(sequences, max_len):
    print(f"Loading RiNALMo from {RINALMO_PATH}...")
    tokenizer = RnaTokenizer.from_pretrained(RINALMO_PATH, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(RINALMO_PATH, trust_remote_code=True, local_files_only=True).to(device)
    
    # [极其重要的一步] 应用你的权重修复函数！
    fix_and_load_weights(model, RINALMO_PATH)
    
    model.eval()
    
    feats = []
    print("Extracting true representations...")
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i:i+BATCH_SIZE]
            inputs = tokenizer(
                batch, return_tensors="pt", padding="max_length", 
                truncation=True, max_length=max_len
            ).to(device)
            
            outputs = model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_len:
                if batch_hidden.size(1) > max_len:
                    batch_hidden = batch_hidden[:, :max_len, :]
                else:
                    pad_length = max_len - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
                    
            feats.append(batch_hidden.cpu())
            del inputs, outputs, batch_hidden
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
                
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return torch.cat(feats, 0).numpy()

class RNADataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# =============================================================================
# 可视化绘图逻辑
# =============================================================================

def plot_probability_kde(all_logits, all_labels):
    print("\nDrawing Probability Density Plot...")
    probs = F.softmax(torch.tensor(all_logits), dim=1)[:, 1].numpy()
    
    probs_dsrna = probs[all_labels == 1]
    probs_non_dsrna = probs[all_labels == 0]
    
    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")
    
    sns.kdeplot(probs_non_dsrna, fill=True, color=COLOR_NON_DSRNA, label='non-dsRNAs (True Label 0)', linewidth=2)
    sns.kdeplot(probs_dsrna, fill=True, color=COLOR_DSRNA, label='dsRNA (True Label 1)', linewidth=2)
    
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Decision Boundary (0.5)')
    
    # --- 移除了此处的标题 ---
    # plt.title('Prediction Probability Distribution (RiNALMo Representation)', fontsize=16, fontweight='bold', pad=15)
    
    plt.xlabel('Predicted Probability of being dsRNA', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(-0.1, 1.1)
    plt.legend(fontsize=12, loc='upper center')
    
    plt.tight_layout()
    filename = 'Probability_Distribution_Fixed.jpg'
    plt.savefig(filename, dpi=300, format='jpg', bbox_inches='tight')
    print(f"    Saved: {filename} (这次你应该能看到两座山峰了！)")
    plt.close()

def plot_supervised_umap(features_data, all_labels):
    sns.set_theme(style="ticks")
    
    plot_configs = [
        ('Input', 'a. Input Features (Raw)', 'UMAP_1_Input_Fixed.jpg'),
        ('CNN_CBAM', 'b. Local Features (CNN+CBAM)', 'UMAP_2_Local_Features_Fixed.jpg'),
        ('LSTM', 'c. Sequence Features (BiLSTM)', 'UMAP_3_Sequence_Features_Fixed.jpg'),
        ('Final', 'd. Final Representations', 'UMAP_4_Final_Representation_Fixed.jpg')
    ]
    
    custom_palette = [COLOR_NON_DSRNA, COLOR_DSRNA]
    label_map = {0: 'non-dsRNAs', 1: 'dsRNA'}
    labels_text = np.array([label_map[l] for l in all_labels])
    
    print("\nStarting Advanced UMAP plotting...")
    
    for key, title, filename in plot_configs:
        print(f"  - Processing: {title}...")
        data = normalize(features_data[key], norm='l2', axis=1)
        
        if key == 'Final':
            print("    Using SUPERVISED UMAP for Final Features...")
            reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, 
                                random_state=UMAP_RANDOM_STATE, target_weight=0.5)
            X_emb = reducer.fit_transform(data, y=all_labels) 
        else:
            reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE)
            X_emb = reducer.fit_transform(data)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sns.scatterplot(
            x=X_emb[:, 0], y=X_emb[:, 1], 
            hue=labels_text, 
            palette=dict(zip(label_map.values(), custom_palette)),
            style=labels_text,
            markers={'non-dsRNAs': 'o', 'dsRNA': 'o'}, 
            s=PLOT_MARKER_SIZE, alpha=PLOT_ALPHA, edgecolor=None, ax=ax
        )
        
        # --- 移除了此处的标题 ---
        # ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        ax.legend(title=None, fontsize=12, loc='upper right', frameon=True)
        
        ax.grid(True, linestyle=':', alpha=0.4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, format='jpg', bbox_inches='tight')
        print(f"    Saved: {filename}")
        plt.close()

# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PTH_PATH):
        raise FileNotFoundError(f"Missing Model: {MODEL_PTH_PATH}")
    
    print("Reading Data...")
    df = pd.read_excel(TEST_EXCEL_PATH)
    sequences = df["Sequence"].tolist()
    labels = df["label"].tolist()

    raw_features = extract_RiNALMo_features(sequences, MAX_SEQ_LENGTH)
    
    print("Applying PCA...")
    with open(PCA_PKL_PATH, 'rb') as f:
        pca = pickle.load(f)
    n, l, c = raw_features.shape
    features_2d = raw_features.reshape(-1, c)
    features_reduced = pca.transform(features_2d).reshape(n, l, INPUT_SIZE)
    
    print("Loading Downstream Classifier...")
    test_loader = DataLoader(RNADataset(features_reduced, labels), batch_size=BATCH_SIZE, shuffle=False)
    
    model = Deep_dsRNAPred(
        max_seq_length=MAX_SEQ_LENGTH, 
        input_size=INPUT_SIZE,
        cnn_layers=CNN_LAYERS,
        cnn_dims=CNN_DIMS, 
        pool_size=POOL_SIZE,
        num_layers=FC_LAYERS,
        num_dims=FC_DIMS,
        dropout_rate=DROPOUT_RATE,
        num_classes=NUM_CLASSES,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        cbam_layers=CBAM_LAYERS
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location=device))
    model.eval()
    
    print("Extracting intermediate representations and logits...")
    features_data = {'Input': [], 'CNN_CBAM': [], 'LSTM': [], 'Final': []}
    all_labels = []
    all_logits = [] 
    
    with torch.no_grad():
        for inputs, batch_labels in test_loader:
            inputs = inputs.to(device)
            logits, feats_dict = model(inputs, return_features=True)
            
            all_logits.append(logits.cpu().numpy())
            features_data['Input'].append(inputs.mean(dim=1).cpu().numpy())
            features_data['CNN_CBAM'].append(feats_dict['cnn_cbam'].cpu().numpy())
            features_data['LSTM'].append(feats_dict['lstm'].cpu().numpy())
            features_data['Final'].append(feats_dict['final'].cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            
    for k in features_data:
        features_data[k] = np.concatenate(features_data[k], axis=0)
    all_labels = np.array(all_labels)
    all_logits = np.concatenate(all_logits, axis=0)
    
    if len(all_labels) > MAX_TSNE_SAMPLES:
        print(f"Sampling {MAX_TSNE_SAMPLES} points for clearer visualization...")
        indices = np.random.choice(len(all_labels), MAX_TSNE_SAMPLES, replace=False)
        for k in features_data: features_data[k] = features_data[k][indices]
        all_labels = all_labels[indices]
        all_logits = all_logits[indices]
    
    plot_probability_kde(all_logits, all_labels)
    plot_supervised_umap(features_data, all_labels)
    
    print("\n恭喜！特征已修复，双子峰和漂亮流形已生成！")