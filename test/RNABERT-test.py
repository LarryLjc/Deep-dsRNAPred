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

# Model Imports
from multimolecule import RnaTokenizer, RnaBertModel
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. Global Configuration
# =============================================================================
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 【必填】路径与超参数配置 ---
RNA_BERT_PATH = r'/root/autodl-tmp/Big_Model/RNABert/ZhejiangLab-LifeScience/rnabert'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

# ⚠️ 这里的长度必须和你之前跑训练代码时，终端打印出来的 "Max Sequence Length" 完全一致！
TRAINED_MAX_SEQ_LENGTH = 615  # <--- 请务必修改为真实的训练长度

# 模型和PCA文件的前缀 (需要和训练脚本保存的路径在同级目录)
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'
OUTPUT_SAVE_PATH = 'RNABert_Final_Ensemble_Predictions.xlsx'

# 模型结构超参数 (必须与训练时一致)
RNA_BERT_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# =============================================================================
# 2. Attention Modules & Core Architecture
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
        max_out = self.se(self.maxpool(x))
        avg_out = self.se(self.avgpool(x))
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
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x

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
        V = (attention_weights * feats).sum(0)
        return V

class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernels=[3], pool_size=2):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), padding=0, bias=True)
        self.sk_attention = SKAttention(channel=out_planes, kernels=kernels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1))
        
    def forward(self, x):
        return self.pool(self.relu(self.sk_attention(self.conv1x1(x))))

class Deep_dsRNAPred(nn.Module):
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=256, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
                 cbam_layers=CBAM_LAYERS):
        super().__init__()
        
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.after_cnn_length = max_seq_length
        
        for i in range(cnn_layers):
            kernels = [3] if i == 0 else [7, 9]
            cnn_block = CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size)
            self.cnn_blocks.append(cnn_block)
            in_planes = cnn_dims
            self.after_cnn_length //= pool_size

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])

        self.bilstm = nn.LSTM(
            input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        self.lstm_flatten_dim = lstm_hidden_size * 2 * self.after_cnn_length
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
        for block in self.cnn_blocks: x = block(x)
        for block in self.cbam_blocks: x = block(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        for block in self.fc_blocks: x = block(x)
        x = self.mid_dropout(F.relu(self.mid_fc(x)))
        return self.output_layer(x)

# =============================================================================
# 3. Data Processing & Feature Extraction
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)

def resize_pos_embeddings(model, new_max_length):
    config = model.config
    current_max_pos = config.max_position_embeddings
    
    if new_max_length > current_max_pos:
        old_embeddings = model.embeddings.position_embeddings.weight.data
        old_embeddings_t = old_embeddings.t().unsqueeze(0)
        
        new_embeddings_t = F.interpolate(
            old_embeddings_t, size=new_max_length, mode='linear', align_corners=True
        )
        
        new_embeddings = new_embeddings_t.squeeze(0).t()
        new_pos_layer = nn.Embedding(new_max_length, config.hidden_size)
        new_pos_layer.weight.data = new_embeddings
        new_pos_layer.to(model.device)
        
        model.embeddings.position_embeddings = new_pos_layer
        new_pos_ids = torch.arange(new_max_length).expand((1, -1)).to(model.device)
        model.embeddings.register_buffer("position_ids", new_pos_ids)
        model.config.max_position_embeddings = new_max_length

def extract_RNA_BERT_features(sequences, max_seq_length):
    print(f"Loading RNA-Bert model from {RNA_BERT_PATH} ...")
    tokenizer = RnaTokenizer.from_pretrained(RNA_BERT_PATH)
    rna_bert_model = RnaBertModel.from_pretrained(RNA_BERT_PATH).to(device)
    rna_bert_model.eval()

    resize_pos_embeddings(rna_bert_model, max_seq_length)

    hidden_states_list = []
    
    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seq = sequences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_seq, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_seq_length
        ).to(device)
        
        with torch.no_grad():
            outputs = rna_bert_model(**inputs)
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

    # Free memory immediately
    del rna_bert_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
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
# 4. Main Inference Execution
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*50}\nStarting Independent Ensemble Inference (RNA-Bert)\n{'='*50}")
    
    # 1. Load Data
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    print(f"Loaded {len(test_sequences)} sequences for testing.")
    print(f"Using fixed Max Sequence Length: {TRAINED_MAX_SEQ_LENGTH}")

    # 2. Extract Raw Features
    print("\nExtracting RNA-Bert Embeddings...")
    X_test_raw = extract_RNA_BERT_features(test_sequences, TRAINED_MAX_SEQ_LENGTH)
    n_test, seq_len, feat_dim = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, feat_dim)

    # 3. Ensemble Processing
    ensemble_probs = np.zeros((KFOLD, n_test, NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"\n--- Loading and Predicting with Fold {fold + 1}/{KFOLD} ---")
        
        # A. Load PCA and Transform
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Cannot find PCA file: {pca_path}. Ensure it is in the correct directory.")
            
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(n_test, seq_len, RNA_BERT_HIDDEN_DIM).astype(np.float32)
        
        # B. Setup DataLoader
        test_dataset = RNADataset(test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # C. Load Deep_dsRNAPred Model
        model = Deep_dsRNAPred(
            max_seq_length=TRAINED_MAX_SEQ_LENGTH, input_size=RNA_BERT_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64, 
            dropout_rate=0.2, num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find Model weights: {model_path}.")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # D. Predict
        all_prob = []
        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                preds = model(features)
                probs = F.softmax(preds, dim=1) 
                all_prob.extend(probs.cpu().numpy())
                
        ensemble_probs[fold] = np.array(all_prob)
        
        # Clean up memory per fold
        del pca, test_red, test_features_reduced, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Compute Final Ensemble & Metrics
    print("\n" + "=" * 50)
    print("Aggregating Soft-Voting Ensemble Results...")
    
    # Calculate average probabilities across 5 folds
    final_avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = np.argmax(final_avg_probs, axis=1)

    # Save Results
    test_df['ensemble_prob_class_1'] = final_avg_probs[:, 1]
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
        
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"Predictions detailed saved to: {OUTPUT_SAVE_PATH}")

    # Print Metrics if possible
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n【Independent Ensemble Test Final Metrics】")
        print(f"ACC: {metrics['ACC']:.4f} | MCC: {metrics['MCC']:.4f} | "
              f"F1: {metrics['F1']:.4f} | AUC: {metrics['AUC']:.4f}")
        print(f"Sn:  {metrics['Sn']:.4f} | Sp:  {metrics['Sp']:.4f}")
        print("=" * 50)