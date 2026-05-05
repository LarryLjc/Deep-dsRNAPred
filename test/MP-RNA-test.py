import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import gc
import os
import joblib  # 用于加载 PCA 模型
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 路径配置 ---
MP_RNA_PATH = r'/root/autodl-tmp/Big_Model/MP-RNA'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

# 加载模型的前缀（请确保当前目录下有这些文件）
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_PREFIX = 'pca_fold_'
OUTPUT_SAVE_PATH = 'Ensemble_Test_Predictions.xlsx'

# --- 核心超参数 (必须与训练时完全一致) ---
# 🚨🚨🚨 重要：这里必须填入你训练时打印出的 Global Max Sequence Length！ 🚨🚨🚨
MAX_SEQ_LENGTH = 615  # <--- 请修改为你训练时的实际值！

MP_RNA_HIDDEN_DIM = 50  
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
THRESHOLD = 0.5  # 软投票决策阈值

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# =============================================================================
# 2. 模型架构定义 (必须保留以成功加载权重)
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
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

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
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=(k, 1), padding=(k // 2, 0), groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0) 
        U = sum(conv_outs) 
        S = U.mean(-1).mean(-1) 
        Z = self.fc(S) 
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1)) 
        attention_weights = torch.stack(weights, 0) 
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V

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
        self.pool_size = pool_size
        self.cnn_layers = cnn_layers
        self.max_seq_length = max_seq_length
        self.cbam_layers = cbam_layers 
        
        for i in range(cnn_layers):
            if i == 0: kernels = [3] 
            else: kernels = [7, 9]
            cnn_block = CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size)
            self.cnn_blocks.append(cnn_block)
            in_planes = cnn_dims

        after_cnn_length = max_seq_length
        for _ in range(cnn_layers):
            after_cnn_length = after_cnn_length // pool_size
        self.after_cnn_length = after_cnn_length
        self.cnn_dims = cnn_dims

        self.cbam_blocks = nn.ModuleList()
        for _ in range(self.cbam_layers):
            self.cbam_blocks.append(CBAMBlock(channel=cnn_dims)) 

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

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        for block in self.cnn_blocks:
            x = block(x) 
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x) 
        x = x.squeeze(-1).permute(0, 2, 1) 
        lstm_out, _ = self.bilstm(x) 
        x = lstm_out.flatten(start_dim=1) 
        for block in self.fc_blocks:
            x = block(x)
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        x = self.output_layer(x)
        return x

# =============================================================================
# 3. 数据处理与评估工具
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, sequences, features=None):
        self.sequences = sequences
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.features is not None:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            return feature
        return self.sequences[idx]

def build_dataloader(sequences, features=None, batch_size=32, shuffle=False):
    dataset = RNADataset(sequences, features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def extract_MP_RNA_features(sequences, max_seq_length, tokenizer, mp_rna_model):
    hidden_states_list = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = mp_rna_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            hidden_states_list.append(batch_hidden.cpu())
            del inputs, outputs, batch_hidden
            torch.cuda.empty_cache()
    
    all_hidden = torch.cat(hidden_states_list, dim=0)
    return all_hidden.numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

def get_model_predictions(model, dataloader):
    model.eval()
    all_prob = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            preds = model(features)
            probs = F.softmax(preds, dim=1)[:, 1] # Target probability for Class 1
            all_prob.extend(probs.cpu().numpy())
    return np.array(all_prob)

# =============================================================================
# 4. 主干逻辑 (测试执行)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> MP-RNA + Deep_dsRNAPred Standalone Testing <<<")
    print("=" * 50)

    # 1. 读入测试数据
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    assert "label" in test_df.columns and "Sequence" in test_df.columns, "Excel must contain 'Sequence' and 'label' columns."
    
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].values
    
    print(f"Enforcing Maximum Sequence Length to: {MAX_SEQ_LENGTH} (Must match training!)")

    # 2. 提取 MP-RNA 原始高维特征
    print(f"\nStep 1: Loading MP-RNA from {MP_RNA_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True)
    mp_rna_model = AutoModel.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True).to(device)
    mp_rna_model.eval()

    print("Step 2: Extracting Raw Features...")
    X_test_raw = extract_MP_RNA_features(test_sequences, MAX_SEQ_LENGTH, tokenizer, mp_rna_model)

    # 清除提取特征占用的大量显存
    print("Deleting MP-RNA model to free GPU memory...")
    del mp_rna_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 3. 集成打分循环 (5个 Fold)
    print("\nStep 3: Ensemble Scoring with K-Fold Models...")
    N_test, L_test, C_test = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, C_test)
    ensemble_probs = np.zeros((KFOLD, N_test))

    for fold in range(KFOLD):
        print(f" -> Processing Fold {fold + 1}/{KFOLD}")
        
        # 加载对应的 PCA
        pca_path = f"{PCA_PREFIX}{fold + 1}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing PCA model file: {pca_path}")
            
        pca = joblib.load(pca_path)
            
        # PCA 降维并还原形状
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(N_test, L_test, MP_RNA_HIDDEN_DIM).astype(np.float32)
        
        # 构造 DataLoader
        test_loader = build_dataloader(test_sequences, features=test_features_reduced, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型架构并加载权重
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=MP_RNA_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing Deep_dsRNAPred weight file: {model_path}")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 获取预测概率
        ensemble_probs[fold] = get_model_predictions(model, test_loader)
        
        # 内存回收
        del pca, test_red, test_features_reduced, model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 4. 指标计算与最终输出
    print("\nStep 4: Calculating Final Metrics...")
    final_avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = (final_avg_probs >= THRESHOLD).astype(int)

    metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)

    print("\n" + "=" * 50)
    print("【 FINAL TEST PERFORMANCE (MP-RNA Ensemble) 】")
    print(f"Sn        : {metrics['Sn']:.4f}")
    print(f"Sp        : {metrics['Sp']:.4f}")
    print(f"ACC       : {metrics['ACC']:.4f}")
    print(f"MCC       : {metrics['MCC']:.4f}")
    print(f"F1_score  : {metrics['F1']:.4f}")
    print(f"AUC       : {metrics['AUC']:.4f}")
    print("=" * 50)

    # 5. 可选：将最终概率保存到Excel
    test_df['ensemble_prob'] = final_avg_probs
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob'] = ensemble_probs[fold]
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"\nDetailed predictions saved to {OUTPUT_SAVE_PATH}")