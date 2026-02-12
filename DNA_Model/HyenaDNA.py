import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc   # 用于显式内存回收
import sys  # 用于添加路径
import pickle # 用于保存pkl文件
import os     # 用于创建目录
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# Transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import KFold
# Metrics (Added f1_score)
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
# PCA
from sklearn.decomposition import PCA

# =============================================================================
# 1. 全局配置与环境初始化 (Global Configuration & Setup)
# =============================================================================

# 过滤特定警告
warnings.filterwarnings("ignore", message="Unable to import Triton")


# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
HYENADNA_PATH = r'/root/autodl-tmp/Big_Model/HyenaDNA/ZhejiangLab-LifeScience/hyenadna-large-1m-seqlen-hf'
TRAIN_CSV_PATH = "/root/autodl-tmp/data/train_combined.csv"
TEST_CSV_PATH = "/root/autodl-tmp/data/test_combined.csv"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
SAVE_PKL_DIR = './pca_features_pkl' 

# 维度与超参数配置
PCA_OUTPUT_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 256
EPOCHS = 40
DROPOUT_RATE = 0.2

# LSTM 与 CBAM 超参数
LSTM_HIDDEN_SIZE = 128  # LSTM 隐藏层维度
LSTM_LAYERS = 2         # 固定为2层 BiLSTM
CBAM_LAYERS = 2         # 默认2层 CBAM

# --- 模型加载 ---
print(f"Loading HyenaDNA from {HYENADNA_PATH}...")

try:
    config = AutoConfig.from_pretrained(HYENADNA_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(HYENADNA_PATH, trust_remote_code=True)
    hyenadna_model = AutoModel.from_pretrained(
        HYENADNA_PATH, 
        config=config, 
        trust_remote_code=True
    ).to(device)
    print("Success: HyenaDNA model loaded.")

except Exception as e:
    print(f"\n[Error] Model loading failed: {e}")
    raise e

hyenadna_model.eval()   # 特征提取模式，不训练


# =============================================================================
# 2. 辅助配置类 (Config Class)
# =============================================================================

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='stage1model_fold_'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix


# =============================================================================
# 3. 注意力机制模块 (Attention Modules)
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
        b, c, _, _ = x.size()
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


# =============================================================================
# 4. 核心网络架构 (Deep_dsRNAPred)
# =============================================================================

class CNNBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernels=[3], pool_size=2):
        super(CNNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=(1, 1), 
            padding=0,
            bias=True
        )
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
        
        # --- CNN Blocks ---
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.pool_size = pool_size
        self.cnn_layers = cnn_layers
        self.max_seq_length = max_seq_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.cbam_layers = cbam_layers 
        
        for i in range(cnn_layers):
            if i == 0:
                kernels = [3] 
            else:
                kernels = [7, 9]
            cnn_block = CNNBlock(
                in_planes=in_planes,
                out_planes=cnn_dims,
                kernels=kernels, 
                pool_size=pool_size
            )
            self.cnn_blocks.append(cnn_block)
            in_planes = cnn_dims

        after_cnn_length = max_seq_length
        for _ in range(cnn_layers):
            after_cnn_length = after_cnn_length // pool_size
        self.after_cnn_length = after_cnn_length
        self.cnn_dims = cnn_dims

        # --- CBAM Modules ---
        self.cbam_blocks = nn.ModuleList()
        for _ in range(self.cbam_layers):
            self.cbam_blocks.append(CBAMBlock(channel=cnn_dims)) 

        # --- BiLSTM ---
        self.bilstm = nn.LSTM(
            input_size=cnn_dims,           
            hidden_size=lstm_hidden_size,  
            num_layers=lstm_layers,         
            bidirectional=True,             
            batch_first=True,               
            dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        self.lstm_output_dim = lstm_hidden_size * 2

        # --- Fully Connected Blocks ---
        self.lstm_flatten_dim = self.lstm_output_dim * self.after_cnn_length
        self.fc_blocks = nn.ModuleList()
        in_features = self.lstm_flatten_dim
        for _ in range(num_layers):
            fc = nn.Linear(in_features=in_features, out_features=num_dims)
            dropout = nn.Dropout(p=dropout_rate)
            self.fc_blocks.append(nn.Sequential(fc, nn.ReLU(), dropout))
            in_features = num_dims

        # --- Output Layers ---
        self.mid_fc = nn.Linear(num_dims, 128)
        self.mid_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        # Adjust for CNN: [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        
        # SKAttention-CNN
        for block in self.cnn_blocks:
            x = block(x) 
        
        # CBAM
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x) 
        
        # Adjust for LSTM
        x = x.squeeze(-1)       
        x = x.permute(0, 2, 1) 
        
        # BiLSTM
        lstm_out, (h_n, c_n) = self.bilstm(x) 
        
        # Flatten
        x = lstm_out.flatten(start_dim=1) 
        
        # FC Layers
        for block in self.fc_blocks:
            x = block(x)
        
        # Output
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        x = self.output_layer(x)
        
        return x


# =============================================================================
# 5. Data Processing & Utils
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


def build_dataloader(sequences, labels, features=None, batch_size=32, shuffle=True):
    dataset = RNADataset(sequences, labels, features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def extract_HyenaDNA_features(sequences, max_seq_length):
    """
    Extract features using HyenaDNA.
    """
    num_samples = len(sequences)
    hidden_size = config.d_model 
    
    print(f"Allocating memory for features: {num_samples} samples, shape=({num_samples}, {max_seq_length}, {hidden_size})")
    all_hidden = np.zeros((num_samples, max_seq_length, hidden_size), dtype=np.float16)
    
    for i in range(0, num_samples, BATCH_SIZE):
        batch_seq = sequences[i : i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_seq_length
        ).to(device)
        
        with torch.no_grad():
            outputs = hyenadna_model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                batch_hidden = outputs.last_hidden_state
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
            
        del inputs, outputs, batch_hidden
        torch.cuda.empty_cache()
        
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Processed batch {i // BATCH_SIZE + 1}/{(num_samples + BATCH_SIZE - 1) // BATCH_SIZE}", end='\r')

    print("\nFeature extraction complete.")
    return all_hidden


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0) # TPR
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1, 'FPR': fpr, 'TPR': sn}


# =============================================================================
# 6. Training & Validation Loop
# =============================================================================

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_acc, total_count = 0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(features)
        
        # 计算 Loss 仅用于梯度反向传播，不返回
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        
        preds_cls = preds.argmax(1)
        total_acc += (preds_cls == labels).sum().item()
        total_count += labels.size(0)
        
    avg_acc = total_acc / total_count
    return avg_acc


def validate_one_epoch(dataloader, model, device):
    # 完全移除 loss_fn 参数和 Loss 计算
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            preds = model(features)
            
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.argmax(1).cpu().numpy())
            all_prob.extend(F.softmax(preds, dim=1).cpu().numpy())
            
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)
    
    metrics = calculate_metrics(all_true, all_pred, all_prob)
    return metrics


def kfold_train(fold, train_features, train_labels, val_features, val_labels, 
                train_sequences, val_sequences, config, epochs, train_loader, val_loader, test_loader):
    # Initialize Model
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size,
        cnn_layers=3,         
        cnn_dims=512,         
        pool_size=2,
        num_layers=3,
        num_dims=64,
        dropout_rate=0.2,
        num_classes=config.num_classes,
        cbam_layers=CBAM_LAYERS 
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    best_val_metrics = None
    print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM Layers: {CBAM_LAYERS}) =====")
    
    for epoch in range(epochs):
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best_Nye_aug.pth")
        
        # 按照要求顺序打印
        print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # Test Evaluation
    model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best_Nye_aug.pth", map_location=config.device))
    test_metrics = validate_one_epoch(test_loader, model, config.device)
    
    # 按照要求顺序打印
    print(f"Fold {fold + 1} Best Result | "
          f"Test Sn: {test_metrics['Sn']:.1%} | Test Sp: {test_metrics['Sp']:.1%} | "
          f"Test ACC: {test_metrics['ACC']:.1%} | Test MCC: {test_metrics['MCC']:.3f} | "
          f"Test F1: {test_metrics['F1']:.3f} | Test AUC: {test_metrics['AUC']:.3f}")
    
    return {
        'fold': fold + 1,
        'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
        'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
        'best_val_AUC': best_val_metrics['AUC'], 'best_val_F1': best_val_metrics['F1'],
        'test_Sn': test_metrics['Sn'],
        'test_Sp': test_metrics['Sp'], 
        'test_ACC': test_metrics['ACC'],
        'test_MCC': test_metrics['MCC'], 
        'test_AUC': test_metrics['AUC'],
        'test_F1': test_metrics['F1'],
        'test_FPR': test_metrics['FPR'], 
        'test_TPR': test_metrics['TPR']
    }


# =============================================================================
# 7. Main Execution
# =============================================================================

if __name__ == "__main__":
    if not os.path.exists(SAVE_PKL_DIR):
        os.makedirs(SAVE_PKL_DIR)
        print(f"Created directory for PCA features: {SAVE_PKL_DIR}")

    # --- Step 1: Read Data ---
    print(f"Reading training data from: {TRAIN_CSV_PATH}")
    cv_df = pd.read_csv(TRAIN_CSV_PATH)
    print(f"Reading testing data from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)

    assert "label" in cv_df.columns and "sequence" in cv_df.columns
    assert "label" in test_df.columns and "sequence" in test_df.columns

    print(f"CV Data Size: {len(cv_df)} | Pos: {sum(cv_df['label'])} | Neg: {len(cv_df)-sum(cv_df['label'])}")
    print(f"Test Data Size: {len(test_df)} | Pos: {sum(test_df['label'])} | Neg: {len(test_df)-sum(test_df['label'])}")

    cv_sequences = cv_df["sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["sequence"].tolist()
    test_labels = test_df["label"].tolist()

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Auto-detected Max Sequence Length: {MAX_SEQ_LENGTH}")

    # --- Step 2: Feature Extraction ONLY (PCA Moved to inside Loop) ---
    print("Starting Feature Extraction for CV set using HyenaDNA...")
    X_cv_raw = extract_HyenaDNA_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values
    gc.collect()

    print("Starting Feature Extraction for Test set using HyenaDNA...")
    X_test_raw = extract_HyenaDNA_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values
    gc.collect()

    # --- Step 3: 5-Fold Cross Validation (With Proper PCA inside loop) ---
    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=PCA_OUTPUT_DIM, 
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreprocessing Fold {fold + 1}...")
        
        # 1. Split Raw Data
        train_features_raw = X_cv_raw[train_idx]
        train_labels = y_cv[train_idx]
        train_sequences = [cv_sequences[i] for i in train_idx]
        
        val_features_raw = X_cv_raw[val_idx]
        val_labels = y_cv[val_idx]
        val_sequences = [cv_sequences[i] for i in val_idx]
        
        # 2. PCA Fitting (Only on Training Data of this Fold!)
        N_train, L_train, C_train = train_features_raw.shape
        train_flat = train_features_raw.reshape(-1, C_train)
        
        pca = PCA(n_components=PCA_OUTPUT_DIM)
        pca.fit(train_flat) 
        
        # 3. Transform Train
        train_features_red_flat = pca.transform(train_flat)
        train_features = train_features_red_flat.reshape(N_train, L_train, PCA_OUTPUT_DIM).astype(np.float32)
        
        # 4. Transform Validation (Using Train's PCA)
        N_val, L_val, C_val = val_features_raw.shape
        val_flat = val_features_raw.reshape(-1, C_val)
        val_features_red_flat = pca.transform(val_flat)
        val_features = val_features_red_flat.reshape(N_val, L_val, PCA_OUTPUT_DIM).astype(np.float32)
        
        # 5. Transform Test (Using Train's PCA)
        N_test, L_test, C_test = X_test_raw.shape
        test_flat = X_test_raw.reshape(-1, C_test)
        test_features_red_flat = pca.transform(test_flat)
        test_features = test_features_red_flat.reshape(N_test, L_test, PCA_OUTPUT_DIM).astype(np.float32)

        # 保存 PCA 特征
        fold_data_to_save = {
            'fold_id': fold + 1,
            'train_features': train_features,
            'train_labels': train_labels,
            'val_features': val_features,
            'val_labels': val_labels,
            'test_features': test_features,
            'test_labels': y_test,
            'pca_model': pca 
        }
        
        pkl_filename = os.path.join(SAVE_PKL_DIR, f"fold_{fold + 1}_pca{PCA_OUTPUT_DIM}_features.pkl")
        try:
            with open(pkl_filename, 'wb') as f:
                pickle.dump(fold_data_to_save, f)
            print(f"Successfully saved PCA features to: {pkl_filename}")
        except Exception as e:
            print(f"Failed to save pickle file for fold {fold+1}: {e}")

        # Clean up
        del train_flat, train_features_red_flat, val_flat, val_features_red_flat, test_flat, test_features_red_flat
        gc.collect()

        # 6. Build DataLoaders
        train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = build_dataloader(test_sequences, test_labels, test_features, batch_size=BATCH_SIZE, shuffle=False)
        
        # 7. Train
        fold_result = kfold_train(fold, train_features, train_labels, val_features, val_labels, 
                                  train_sequences, val_sequences, config, EPOCHS, 
                                  train_loader, val_loader, test_loader)
        all_fold_results.append(fold_result)
    
    # --- Step 4: Summary ---
    get_avg = lambda key: np.mean([res[key] for res in all_fold_results])
    get_std = lambda key: np.std([res[key] for res in all_fold_results])
    
    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred, Feature: HyenaDNA+StrictPCA(50))")
    print(f"Max Seq Len: {MAX_SEQ_LENGTH} | CBAM Layers: {CBAM_LAYERS}")
    
    print("\n【Validation Avg Metrics】")
    print(f"Sn:  {get_avg('best_val_Sn'):.3f} ± {get_std('best_val_Sn'):.3f}")
    print(f"Sp:  {get_avg('best_val_Sp'):.3f} ± {get_std('best_val_Sp'):.3f}")
    print(f"ACC: {get_avg('best_val_ACC'):.3f} ± {get_std('best_val_ACC'):.3f}")
    print(f"MCC: {get_avg('best_val_MCC'):.3f} ± {get_std('best_val_MCC'):.3f}")
    print(f"F1:  {get_avg('best_val_F1'):.3f} ± {get_std('best_val_F1'):.3f}")
    print(f"AUC: {get_avg('best_val_AUC'):.3f} ± {get_std('best_val_AUC'):.3f}")
    
    print("\n【Test Avg Metrics】")
    print(f"Sn:  {get_avg('test_Sn'):.3f} ± {get_std('test_Sn'):.3f}")
    print(f"Sp:  {get_avg('test_Sp'):.3f} ± {get_std('test_Sp'):.3f}")
    print(f"ACC: {get_avg('test_ACC'):.3f} ± {get_std('test_ACC'):.3f}")
    print(f"MCC: {get_avg('test_MCC'):.3f} ± {get_std('test_MCC'):.3f}")
    print(f"F1:  {get_avg('test_F1'):.3f} ± {get_std('test_F1'):.3f}")
    print(f"AUC: {get_avg('test_AUC'):.3f} ± {get_std('test_AUC'):.3f}")
    print("=" * 80)

    # --- Step 5: Save Results (No Loss) ---
    result_df = pd.DataFrame(all_fold_results)
    result_save_path = "Deep_dsRNAPred_Performance.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nPerformance metrics saved to: {result_save_path}")