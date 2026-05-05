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

# --- 引入 ModelScope 的 DNABert ---
from modelscope import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path Configuration (请修改为你本地的实际路径)
DNA_BERT_PATH = r'/root/autodl-tmp/Big_Model/DNABERT/zhangtaolab/plant-dnabert-BPE'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_MODEL_PREFIX = 'pca_model_fold_'

# Hyperparameters (必须与训练时完全一致)
DNA_BERT_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# ⚠️⚠️⚠️ 极其重要: 必须填写你训练时终端打印出的 Max Sequence Length
# 否则会导致模型 FC 层维度不匹配！
TRAIN_MAX_SEQ_LENGTH = 615  # <--- 请替换为实际的数值

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
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1).mean(-1)
        Z = self.fc(S)
        weights = [fc(Z).view(bs, c, 1, 1) for fc in self.fcs]
        attention_weights = self.softmax(torch.stack(weights, 0))
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
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=512, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=128, lstm_layers=2,
                 cbam_layers=2):
        super(Deep_dsRNAPred, self).__init__()
        
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.after_cnn_length = max_seq_length
        
        for i in range(cnn_layers):
            kernels = [3] if i == 0 else [7, 9]
            cnn_block = CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size)
            self.cnn_blocks.append(cnn_block)
            in_planes = cnn_dims
            self.after_cnn_length //= pool_size

        self.cbam_blocks = nn.ModuleList()
        for _ in range(cbam_layers):
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
            self.fc_blocks.append(nn.Sequential(fc, nn.ReLU(), nn.Dropout(p=dropout_rate)))
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
        return self.output_layer(x)

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

def resize_pos_embeddings(model, tokenizer, new_max_length):
    config = model.config
    current_max_pos = config.max_position_embeddings
    if new_max_length <= current_max_pos:
        return 
    if hasattr(model, 'bert'):
        embed_layer = model.bert.embeddings
    elif hasattr(model, 'roberta'):
        embed_layer = model.roberta.embeddings
    elif hasattr(model, 'embeddings'):
        embed_layer = model.embeddings
    else:
        raise ValueError("Could not find the embeddings layer in the model.")
        
    old_embeddings = embed_layer.position_embeddings.weight.data
    old_embeddings_t = old_embeddings.t().unsqueeze(0) 
    new_embeddings_t = F.interpolate(old_embeddings_t, size=new_max_length, mode='linear', align_corners=True)
    new_embeddings = new_embeddings_t.squeeze(0).t()
    
    new_pos_layer = nn.Embedding(new_max_length, config.hidden_size)
    new_pos_layer.weight.data = new_embeddings
    new_pos_layer.to(device)
    embed_layer.position_embeddings = new_pos_layer
    
    if hasattr(embed_layer, 'position_ids'):
        new_pos_ids = torch.arange(new_max_length).expand((1, -1)).to(device)
        embed_layer.register_buffer("position_ids", new_pos_ids, persistent=False)
        
    model.config.max_position_embeddings = new_max_length
    tokenizer.model_max_length = new_max_length 

def extract_DNA_BERT_features(sequences, max_seq_length, model, tokenizer):
    resize_pos_embeddings(model, tokenizer, max_seq_length)
    hidden_states_list = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            batch_hidden = outputs.hidden_states[-1]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            hidden_states_list.append(batch_hidden.cpu())
            
    return torch.cat(hidden_states_list, dim=0).numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. Main Test Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 Starting Deep_dsRNAPred Evaluation")
    print(f"Loading Test Data from: {TEST_EXCEL_PATH}")
    
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    y_true = test_df["label"].values

    print(f"Number of test sequences: {len(test_sequences)}")
    print(f"Target Sequence Length: {TRAIN_MAX_SEQ_LENGTH}")
    
    # --- Step 1: Extract Raw Features ---
    print("\n[1/3] Loading DNA-Bert model for feature extraction...")
    tokenizer = AutoTokenizer.from_pretrained(DNA_BERT_PATH)
    dna_bert_model = AutoModelForMaskedLM.from_pretrained(DNA_BERT_PATH).to(device)
    dna_bert_model.eval()

    print("Extracting Raw Features...")
    X_test_raw = extract_DNA_BERT_features(test_sequences, TRAIN_MAX_SEQ_LENGTH, dna_bert_model, tokenizer)
    n_test, seq_len, feat_dim = X_test_raw.shape
    X_test_flat = X_test_raw.reshape(-1, feat_dim)
    
    # Free Memory
    del dna_bert_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Feature extraction complete. DNA-Bert model removed from memory.\n")

    # --- Step 2: Evaluate Each Fold & Collect Probabilities ---
    print("[2/3] Evaluating 5 Folds...")
    
    all_folds_probs = [] # 用于保存所有折的预测概率以进行 Ensemble
    fold_metrics = []

    for fold in range(1, KFOLD + 1):
        print(f"--- Processing Fold {fold} ---")
        
        # 1. 加载对应的 PCA 模型进行降维
        pca_path = f"{PCA_MODEL_PREFIX}{fold}.pkl"
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        X_test_red = pca.transform(X_test_flat)
        X_test_fold = X_test_red.reshape(n_test, seq_len, DNA_BERT_HIDDEN_DIM)
        
        # 构建 DataLoader
        test_loader = build_dataloader(test_sequences, y_true, X_test_fold, batch_size=BATCH_SIZE, shuffle=False)
        
        # 2. 初始化分类器模型并加载权重
        model = Deep_dsRNAPred(
            max_seq_length=TRAIN_MAX_SEQ_LENGTH,
            input_size=DNA_BERT_HIDDEN_DIM,
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
        all_folds_probs.append(fold_probs)
        
        # 计算当前折的指标
        fold_preds = fold_probs.argmax(axis=1)
        metrics = calculate_metrics(y_true, fold_preds, fold_probs)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold} Metrics: Sn={metrics['Sn']:.3f}, Sp={metrics['Sp']:.3f}, ACC={metrics['ACC']:.3f}, AUC={metrics['AUC']:.3f}")
        
        del model, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- Step 3: Ensemble Evaluation (5折概率平均) ---
    print("\n[3/3] Calculating Final Ensemble Performance...")
    # 将5折输出的概率做平均
    ensemble_probs = np.mean(all_folds_probs, axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    
    ensemble_metrics = calculate_metrics(y_true, ensemble_preds, ensemble_probs)
    
    print("="*60)
    print("🏆 FINAL TEST SET PERFORMANCE (Ensemble of 5 Folds)")
    print("="*60)
    print(f"Sensitivity (Sn) : {ensemble_metrics['Sn']:.4f}")
    print(f"Specificity (Sp) : {ensemble_metrics['Sp']:.4f}")
    print(f"Accuracy (ACC)   : {ensemble_metrics['ACC']:.4f}")
    print(f"MCC              : {ensemble_metrics['MCC']:.4f}")
    print(f"F1 Score         : {ensemble_metrics['F1']:.4f}")
    print(f"ROC AUC          : {ensemble_metrics['AUC']:.4f}")
    print("="*60)

    # 保存预测结果到 Excel（可选）
    test_df['Prob_Class_0'] = ensemble_probs[:, 0]
    test_df['Prob_Class_1'] = ensemble_probs[:, 1]
    test_df['Predicted_Label'] = ensemble_preds
    save_pred_path = "Test_Predictions_Ensemble.xlsx"
    test_df.to_excel(save_pred_path, index=False)
    print(f"Detailed predictions saved to: {save_pred_path}")