import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import pickle
import gc
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", category=FutureWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path Configuration (请修改为你本地的实际路径)
GENA_LM_PATH = r'/root/autodl-tmp/Big_Model/Gena-LM/lgq12697/gena-lm-bert-base-t2t'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'

# Hyperparameters (必须与训练时完全一致)
GENA_LM_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2  # 与训练脚本保持一致

# ⚠️⚠️⚠️ 极其重要: 必须填写你训练时终端打印出的 Max Seq Len
# 否则会导致模型 FC 层维度不匹配！
TRAIN_MAX_SEQ_LENGTH = 617  # <--- 请务必替换为训练时打印出的实际数值

# =============================================================================
# 2. Model Architecture (保持与训练集完全相同的结构)
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
# 3. Data Processing & Utilities
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, sequences, labels, features=None):
        self.sequences = sequences
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.features is not None:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            return feature, label
        return self.sequences[idx], label

def build_dataloader(sequences, labels, features=None, batch_size=32, shuffle=False):
    dataset = RNADataset(sequences, labels, features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def resize_pos_embeddings(model, new_max_length):
    config = model.config
    current_max_pos = getattr(config, 'max_position_embeddings', 512)
    
    if new_max_length > current_max_pos:
        base_model = getattr(model, model.base_model_prefix, model)
        if hasattr(base_model, 'embeddings') and hasattr(base_model.embeddings, 'position_embeddings'):
            old_embeddings = base_model.embeddings.position_embeddings.weight.data
            old_embeddings_t = old_embeddings.t().unsqueeze(0)
            
            new_embeddings_t = F.interpolate(
                old_embeddings_t, 
                size=new_max_length, 
                mode='linear', 
                align_corners=True
            )
            
            new_embeddings = new_embeddings_t.squeeze(0).t()
            new_pos_layer = nn.Embedding(new_max_length, config.hidden_size)
            new_pos_layer.weight.data = new_embeddings
            new_pos_layer.to(model.device)
            
            base_model.embeddings.position_embeddings = new_pos_layer
            new_pos_ids = torch.arange(new_max_length).expand((1, -1)).to(model.device)
            base_model.embeddings.register_buffer("position_ids", new_pos_ids)
            model.config.max_position_embeddings = new_max_length

def extract_GENA_LM_features(sequences, max_seq_length, model, tokenizer):
    resize_pos_embeddings(model, max_seq_length)
    
    num_samples = len(sequences)
    hidden_size = getattr(model.config, 'hidden_size', 1024) 
    all_hidden = np.zeros((num_samples, max_seq_length, hidden_size), dtype=np.float16)
    
    base_model = getattr(model, model.base_model_prefix, model)
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            batch_seq = [seq.replace('U', 'T').replace('u', 't') for seq in batch_seq]
            
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = base_model(**inputs, output_hidden_states=True)
            
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                batch_hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                batch_hidden = outputs.hidden_states[-1]
            else:
                batch_hidden = outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            current_batch_len = len(batch_seq)
            all_hidden[i : i + current_batch_len] = batch_hidden.cpu().numpy().astype(np.float16)
            
    return all_hidden

def transform_pca_inference(X_data, ipca, n_components=50):
    N, L, C = X_data.shape
    pca_chunk_size = 100
    X_reduced_list = []
    for i in range(0, N, pca_chunk_size):
        chunk = X_data[i : i + pca_chunk_size]
        chunk_flat = chunk.reshape(-1, C)
        chunk_reduced = ipca.transform(chunk_flat)
        chunk_3d = chunk_reduced.reshape(chunk.shape[0], L, n_components)
        X_reduced_list.append(chunk_3d.astype(np.float32))
    return np.concatenate(X_reduced_list, axis=0)

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. Main Test Execution
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 Starting Deep_dsRNAPred (Gena-LM + StrictPCA) Evaluation")
    print(f"Loading Test Data from: {TEST_EXCEL_PATH}")
    
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    y_true = test_df["label"].values

    print(f"Number of test sequences: {len(test_sequences)}")
    print(f"Target Sequence Length: {TRAIN_MAX_SEQ_LENGTH}")
    
    # --- Step 1: Extract Raw Features ---
    print("\n[1/3] Loading Gena-LM model for feature extraction...")
    tokenizer = AutoTokenizer.from_pretrained(GENA_LM_PATH, trust_remote_code=True, local_files_only=True)
    gena_lm_model = AutoModel.from_pretrained(GENA_LM_PATH, trust_remote_code=True, local_files_only=True).to(device)
    gena_lm_model.eval()

    print("Extracting Raw Features...")
    X_test_raw = extract_GENA_LM_features(test_sequences, TRAIN_MAX_SEQ_LENGTH, gena_lm_model, tokenizer)
    
    # Free Memory
    del gena_lm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Feature extraction complete. Gena-LM model removed from memory.\n")

    # --- Step 2: Evaluate Each Fold & Collect Probabilities ---
    print("[2/3] Evaluating 5 Folds...")
    
    n_test = X_test_raw.shape[0]
    ensemble_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
    fold_metrics = []

    for fold in range(1, KFOLD + 1):
        print(f"--- Processing Fold {fold} ---")
        
        # 1. 加载对应的 PCA 模型进行降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold}.pkl"
        with open(pca_path, 'rb') as f:
            ipca_model = pickle.load(f)
            
        X_test_pca = transform_pca_inference(X_test_raw, ipca_model, n_components=GENA_LM_HIDDEN_DIM)
        
        # 构建 DataLoader (必须不打乱顺序)
        test_loader = build_dataloader(test_sequences, y_true, X_test_pca, batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. 初始化分类器模型并加载权重 (cnn_dims 必须设为 512，与训练脚本实例化保持一致)
        model = Deep_dsRNAPred(
            max_seq_length=TRAIN_MAX_SEQ_LENGTH,
            input_size=GENA_LM_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=0.2, num_classes=NUM_CLASSES, 
            lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
            cbam_layers=CBAM_LAYERS
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 3. 推理阶段
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        fold_probs = np.array(fold_probs)
        ensemble_probs += fold_probs
        
        # 计算当前折的指标
        fold_preds = fold_probs.argmax(axis=1)
        metrics = calculate_metrics(y_true, fold_preds, fold_probs)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold} Metrics: Sn={metrics['Sn']:.3f}, Sp={metrics['Sp']:.3f}, ACC={metrics['ACC']:.3f}, AUC={metrics['AUC']:.3f}")
        
        del model, ipca_model, X_test_pca, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- Step 3: Ensemble Evaluation (5折概率平均) ---
    print("\n[3/3] Calculating Final Ensemble Performance...")
    
    ensemble_probs /= KFOLD
    ensemble_preds = ensemble_probs.argmax(axis=1)
    
    ensemble_metrics = calculate_metrics(y_true, ensemble_preds, ensemble_probs)
    
    print("="*80)
    print("🏆 FINAL TEST SET PERFORMANCE (Ensemble of 5 Folds)")
    print("="*80)
    print(f"Sensitivity (Sn) : {ensemble_metrics['Sn']:.4f}")
    print(f"Specificity (Sp) : {ensemble_metrics['Sp']:.4f}")
    print(f"Accuracy (ACC)   : {ensemble_metrics['ACC']:.4f}")
    print(f"MCC              : {ensemble_metrics['MCC']:.4f}")
    print(f"F1 Score         : {ensemble_metrics['F1']:.4f}")
    print(f"ROC AUC          : {ensemble_metrics['AUC']:.4f}")
    print("="*80)

    # 保存预测结果到 Excel
    test_df['Prob_Class_0'] = ensemble_probs[:, 0]
    test_df['Prob_Class_1'] = ensemble_probs[:, 1]
    test_df['Predicted_Label'] = ensemble_preds
    save_pred_path = "Test_Predictions_Ensemble_GenaLM.xlsx"
    test_df.to_excel(save_pred_path, index=False)
    print(f"Detailed predictions saved to: {save_pred_path}")