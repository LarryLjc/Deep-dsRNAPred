import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import gc
import os
import random
import pickle
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Transformers & Sklearn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score, roc_curve
from sklearn.decomposition import PCA

# =============================================================================
# 0. 全局配置与环境初始化
# =============================================================================
def set_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 1. 路径与超参数配置 (严格对齐训练集)
# =============================================================================
CADUCEUS_PATH = r'/root/autodl-tmp/Big_Model/CAD/lgq12697/PlantCAD2-Small-l24-d0768'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'

CADUCEUS_HIDDEN_DIM = 50 
NUM_CLASSES = 2
BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2
THRESHOLD = 0.5  # 默认集成阈值

# 【非常重要】：请替换为你训练脚本打印出的 Max Seq Len
MAX_SEQ_LENGTH = 615  # <--- 注意！运行前请检查这是否等于训练集打印出的 MAX_SEQ_LENGTH

# =============================================================================
# 全局预加载模型 (移除了 local_files_only=True 以防报错)
# =============================================================================
print(f"Loading Caduceus from {CADUCEUS_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(CADUCEUS_PATH, trust_remote_code=True)
caduceus_model = AutoModel.from_pretrained(CADUCEUS_PATH, trust_remote_code=True).to(device)
caduceus_model.eval()

# =============================================================================
# 2. 网络架构定义 (Deep_dsRNAPred)
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
        return self.sigmoid(self.se(self.maxpool(x)) + self.se(self.avgpool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_result, avg_result], 1)))

class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        return x + (out * self.sa(out))

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 7, 9], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channel, channel, kernel_size=(k, 1), padding=(k // 2, 0), groups=group)),
                ('bn', nn.BatchNorm2d(channel)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])
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
        x = self.pool(x)        
        return x

class Deep_dsRNAPred(nn.Module):
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=256, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
                 cbam_layers=CBAM_LAYERS): 
        super(Deep_dsRNAPred, self).__init__()
        
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.after_cnn_length = max_seq_length
        
        for i in range(cnn_layers):
            if i == 0: kernels = [1]
            elif i == 1: kernels = [3]
            else: kernels = [7,9]
            cnn_block = CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size)
            self.cnn_blocks.append(cnn_block)
            in_planes = cnn_dims
            self.after_cnn_length //= pool_size

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])

        self.bilstm = nn.LSTM(
            input_size=cnn_dims,            
            hidden_size=lstm_hidden_size,  
            num_layers=lstm_layers,         
            bidirectional=True,             
            batch_first=True,               
            dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        
        self.lstm_flatten_dim = lstm_hidden_size * 2 * self.after_cnn_length
        self.fc_blocks = nn.ModuleList()
        in_features = self.lstm_flatten_dim
        for _ in range(num_layers):
            self.fc_blocks.append(nn.Sequential(nn.Linear(in_features, num_dims), nn.ReLU(), nn.Dropout(p=dropout_rate)))
            in_features = num_dims

        self.mid_fc = nn.Linear(num_dims, 128)
        self.mid_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        for block in self.cnn_blocks: x = block(x) 
        for cbam_block in self.cbam_blocks: x = cbam_block(x) 
        x = x.squeeze(-1).permute(0, 2, 1) 
        
        lstm_out, _ = self.bilstm(x) 
        x = lstm_out.flatten(start_dim=1) 
        
        for block in self.fc_blocks: x = block(x)
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)

# =============================================================================
# 3. 数据处理与辅助函数
# =============================================================================
class RNADataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def extract_Caduceus_features(sequences, max_seq_length):
    # 移除了局部加载的冗余代码，直接使用全局预加载的 caduceus_model
    hidden_states_list = []
    print("Extracting raw features for Test Set...")
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            if 'token_type_ids' in inputs: del inputs['token_type_ids']
            if 'attention_mask' in inputs: del inputs['attention_mask']
                
            outputs = caduceus_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            hidden_states_list.append(batch_hidden.cpu())
            
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(hidden_states_list, dim=0).numpy()

def calculate_metrics(y_true, y_pred, pos_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, pos_prob) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. 主执行程序
# =============================================================================
if __name__ == "__main__":
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    y_test = test_df["label"].values
    N_test = len(y_test)
    print(f"Loaded {N_test} test samples. Pos: {sum(y_test)}, Neg: {N_test - sum(y_test)}")

    # 1. 一次性提取大模型特征
    X_test_raw = extract_Caduceus_features(test_sequences, MAX_SEQ_LENGTH)
    ensemble_probs_accum = np.zeros((N_test, NUM_CLASSES))

    # 2. 5折循环集成预测
    for fold in range(1, 6):
        print(f"\n[{fold}/5] Processing Fold {fold}...")
        
        # A. 加载对应折的 PCA 模型进行降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing {pca_path}. Did you run the training script to save the PCA models?")
        
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        N, L, C = X_test_raw.shape
        test_flat = X_test_raw.reshape(-1, C)
        test_red = pca.transform(test_flat)
        test_features = test_red.reshape(N, L, CADUCEUS_HIDDEN_DIM).astype(np.float32)
        
        del test_flat, test_red
        gc.collect()

        # 注意这里的 256 与你训练脚本中下发的 batch_size 保持一致
        test_loader = DataLoader(RNADataset(test_features, y_test), batch_size=256, shuffle=False)

        # B. 加载网络模型权重
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, 
            input_size=CADUCEUS_HIDDEN_DIM,
            cnn_dims=512,  
            cbam_layers=CBAM_LAYERS
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() 

        # C. 收集预测概率
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                logits = model(features)
                probs = F.softmax(logits, dim=1)
                fold_probs.extend(probs.cpu().numpy())
                
        ensemble_probs_accum += np.array(fold_probs)

        del model, test_features, test_loader
        torch.cuda.empty_cache()

    # =========================================================================
    # 5. 计算最终集成评估
    # =========================================================================
    print("\n" + "=" * 50)
    print("Ensemble Evaluation Finished. Calculating metrics...")
    
    final_avg_probs = ensemble_probs_accum / 5.0
    pos_probs = final_avg_probs[:, 1]
    final_preds_default = (pos_probs >= THRESHOLD).astype(int)

    metrics_default = calculate_metrics(y_true=y_test, y_pred=final_preds_default, pos_prob=pos_probs)

    print(f"Ensemble Performance (Threshold = {THRESHOLD}):")
    print(f"Sn  (Recall): {metrics_default['Sn']:.4f} ({metrics_default['Sn']*100:.2f}%)")
    print(f"Sp  (Spec)  : {metrics_default['Sp']:.4f} ({metrics_default['Sp']*100:.2f}%)")
    print(f"ACC (Accur) : {metrics_default['ACC']:.4f} ({metrics_default['ACC']*100:.2f}%)")
    print(f"MCC         : {metrics_default['MCC']:.4f}")
    print(f"F1 Score    : {metrics_default['F1']:.4f}")
    print(f"AUC         : {metrics_default['AUC']:.4f}")
    print("=" * 50)

    # 导出默认 0.5 阈值的预测结果
    test_df['ensemble_prob_pos'] = pos_probs
    test_df['ensemble_pred_label'] = final_preds_default
    output_excel = "Final_Ensemble_Predictions_Caduceus.xlsx"
    test_df.to_excel(output_excel, index=False)

    # =========================================================================
    # 6. 寻找最佳阈值 (Youden's J statistic)
    # =========================================================================
    print("\nAnalyzing for Optimal Threshold...")
    
    fpr, tpr, thresholds = roc_curve(y_test, pos_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    print(f"Calculated Optimal Threshold: {best_threshold:.4f}")
    
    best_preds = (pos_probs >= best_threshold).astype(int)
    best_metrics = calculate_metrics(y_true=y_test, y_pred=best_preds, pos_prob=pos_probs)
    
    print(f"\n--- Performance with Optimal Threshold ({best_threshold:.4f}) ---")
    print(f"Sn  (Recall): {best_metrics['Sn']:.4f} ({best_metrics['Sn']*100:.2f}%)")
    print(f"Sp  (Spec)  : {best_metrics['Sp']:.4f} ({best_metrics['Sp']*100:.2f}%)")
    print(f"ACC (Accur) : {best_metrics['ACC']:.4f} ({best_metrics['ACC']*100:.2f}%)")
    print(f"MCC         : {best_metrics['MCC']:.4f}")
    print(f"F1 Score    : {best_metrics['F1']:.4f}")
    print(f"AUC         : {best_metrics['AUC']:.4f}")
    print("=" * 50)

    # 导出最优阈值的预测结果
    test_df['best_threshold_pred'] = best_preds
    output_excel_best = "Test_Ensemble_Predictions_Best_Threshold_Caduceus.xlsx"
    test_df.to_excel(output_excel_best, index=False)
    print(f"\nDetailed predictions saved to '{output_excel_best}'")