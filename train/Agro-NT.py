import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import pickle
import gc
import sys
import random
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# --- 引入 ModelScope 的 Agro-NT 模型 ---
from modelscope import AutoModelForMaskedLM, AutoTokenizer

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.decomposition import PCA  # 恢复为标准 PCA

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

warnings.filterwarnings("ignore")

# --- Random Seed Setup ---
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 3407
setup_seed(SEED)
print(f"Global Random Seed set to: {SEED}")
# -------------------------------

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path Configuration (Agro-NT 绝对路径)
AGRO_NT_PATH = r'/root/autodl-tmp/Big_Model/Agro-NT/ZhejiangLab-LifeScience/agro-nucleotide-transformer-1b'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'
OUTPUT_SAVE_PATH = 'Final_Ensemble_Predictions_AgroNT.xlsx' # 保存预测结果

# Hyperparameters
AGRO_NT_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# --- 【修改】优化器与学习率调度超参数 ---
LEARNING_RATE = 1e-4      # Adam 学习率
WEIGHT_DECAY = 1e-5       # Adam 权重衰减
STEP_SIZE = 20            # StepLR 每训练 20 个 epoch 调整一次
GAMMA = 0.5               # StepLR 每次调整将学习率减半
# ---------------------------------------------

# Initialize Tokenizer & Model
print(f"Loading Agro-NT from {AGRO_NT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(AGRO_NT_PATH)
agro_nt_model = AutoModelForMaskedLM.from_pretrained(AGRO_NT_PATH).to(device)
agro_nt_model.eval()

# =============================================================================
# 2. Config Class
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
# 3. Attention Modules
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


# =============================================================================
# 4. Core Model Architecture: Deep_dsRNAPred
# =============================================================================

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
        # Input: [B, L, C] -> CNN format: [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        for block in self.cnn_blocks:
            x = block(x)
        
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x)
        
        # To LSTM format: [B, L_new, C_new]
        x = x.squeeze(-1).permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        
        for block in self.fc_blocks:
            x = block(x)
        
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)


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


def extract_AgroNT_features_to_disk(sequences, max_seq_length, mmap_filename):
    """提取 Agro-NT 原始高维特征并直接落盘 (Memmap)，彻底防 OOM"""
    print(f"Extracting features directly to disk: {mmap_filename}...")
    fp = None
    shape = None
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            
            # 关闭截断，仅做 Padding 即可（因为 Agro-NT 原生长度足够大）
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=False, max_length=max_seq_length
            ).to(device)
            
            # 使用 output_hidden_states=True 获取隐藏层特征
            outputs = agro_nt_model(**inputs, output_hidden_states=True)
            batch_hidden = outputs.hidden_states[-1]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            # 转为半精度以大幅降低硬盘写入和读取负荷
            batch_hidden_np = batch_hidden.cpu().half().numpy()
            
            # 初始化硬盘映射文件
            if fp is None:
                hidden_dim = batch_hidden_np.shape[2]
                shape = (len(sequences), max_seq_length, hidden_dim)
                fp = np.memmap(mmap_filename, dtype='float16', mode='w+', shape=shape)
            
            # 边提取边落盘
            fp[i : i + batch_hidden_np.shape[0]] = batch_hidden_np[:]
            fp.flush()
            
            del inputs, outputs, batch_hidden, batch_hidden_np
            torch.cuda.empty_cache()
            gc.collect()
            
    return np.memmap(mmap_filename, dtype='float16', mode='r', shape=shape)


def chunked_pca_transform_for_memmap(pca_model, X_mmap, indices=None, chunk_size=1000):
    """流式读取 memmap 进行 PCA 降维，极大节约内存"""
    if indices is None:
        indices = np.arange(X_mmap.shape[0])
        
    res = []
    for i in range(0, len(indices), chunk_size):
        chunk_idx = indices[i : i + chunk_size]
        chunk_data = X_mmap[chunk_idx].astype(np.float32) 
        chunk_flat = chunk_data.reshape(-1, chunk_data.shape[-1])
        
        transformed_flat = pca_model.transform(chunk_flat)
        transformed = transformed_flat.reshape(len(chunk_idx), chunk_data.shape[1], -1)
        res.append(transformed)
        
    return np.vstack(res)


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc, 'FPR': fpr, 'TPR': sn}


# =============================================================================
# 6. Training & Validation Functions
# =============================================================================

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_acc, total_count = 0, 0
    
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(features)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        
        preds_cls = preds.argmax(1)
        total_acc += (preds_cls == labels).sum().item()
        total_count += labels.size(0)
        
    avg_acc = total_acc / total_count
    return avg_acc


def validate_one_epoch(dataloader, model, device):
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
                train_sequences, val_sequences, config, epochs, train_loader, val_loader):
    
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size,
        cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
        dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    
    # --- 【修改】使用 Adam 和 StepLR ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    # ----------------------------------------
    
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
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
        
        print(f"Epoch {epoch + 1:3d} | Train ACC: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # 【修改】移除了内部对 Test set 的评估
    return {
        'fold': fold + 1,
        'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
        'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
        'best_val_F1': best_val_metrics['F1'], 'best_val_AUC': best_val_metrics['AUC']
    }


# =============================================================================
# 7. Main Execution
# =============================================================================

if __name__ == "__main__":
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = np.array(cv_df["label"].tolist())
    test_sequences = test_df["Sequence"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")

    cv_mmap_path = '/root/autodl-tmp/cv_features.dat'
    test_mmap_path = '/root/autodl-tmp/test_features.dat'
    
    print("Extracting Raw Features (Train)...")
    X_cv_raw = extract_AgroNT_features_to_disk(cv_sequences, MAX_SEQ_LENGTH, cv_mmap_path)
    
    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_AgroNT_features_to_disk(test_sequences, MAX_SEQ_LENGTH, test_mmap_path)

    print("Feature extraction complete. Deleting Agro-NT model to free memory...")
    del agro_nt_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared. Starting Cross Validation.")

    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=AGRO_NT_HIDDEN_DIM, 
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreparing Fold {fold + 1}...")
        
        # 强制排序以提升 memmap 硬盘连续读取速度
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        
        # 使用标准 PCA 并配合随机采样防止 OOM
        print("Fitting standard PCA over a randomly sampled subset of training data to prevent OOM...")
        sample_size = min(1000, len(train_idx))
        sampled_indices = np.random.choice(train_idx, size=sample_size, replace=False)
        sampled_indices.sort()
        
        sampled_data = X_cv_raw[sampled_indices].astype(np.float32)
        train_flat_sampled = sampled_data.reshape(-1, sampled_data.shape[-1])
        
        pca = PCA(n_components=AGRO_NT_HIDDEN_DIM, random_state=SEED)
        pca.fit(train_flat_sampled)
        
        # 拟合完立即释放采样数据
        del sampled_data, train_flat_sampled
        gc.collect()

        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to {pca_save_path}")
        
        print("Transforming training features...")
        train_features = chunked_pca_transform_for_memmap(pca, X_cv_raw, train_idx)
        
        print("Transforming validation features...")
        val_features = chunked_pca_transform_for_memmap(pca, X_cv_raw, val_idx)
        
        train_labels_fold = cv_labels[train_idx]
        val_labels_fold = cv_labels[val_idx]
        
        train_seqs_fold = [cv_sequences[i] for i in train_idx]
        val_seqs_fold = [cv_sequences[i] for i in val_idx]
        
        train_loader = build_dataloader(train_seqs_fold, train_labels_fold, train_features, batch_size=256, shuffle=True) 
        val_loader = build_dataloader(val_seqs_fold, val_labels_fold, val_features, batch_size=256, shuffle=False)
        
        fold_result = kfold_train(fold, train_features, train_labels_fold, val_features, val_labels_fold, 
                                  train_seqs_fold, val_seqs_fold, config, EPOCHS, 
                                  train_loader, val_loader)
        all_fold_results.append(fold_result)

        print(f"Cleaning up Fold {fold + 1} memory...")
        del train_features, val_features
        del train_loader, val_loader
        del pca
        torch.cuda.empty_cache()
        gc.collect()

    # --- 步骤 4: 集成学习 (Soft Voting) 测试集评估 ---
    print("\n" + "=" * 80)
    print(">>> Starting Ensemble Evaluation on Test Set <<<")
    
    ensemble_probs = np.zeros((KFOLD, len(test_labels), NUM_CLASSES))
    test_probs_sum = np.zeros((len(test_labels), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"-> Processing Fold {fold + 1}/{KFOLD} for Inference...")
        
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_features = chunked_pca_transform_for_memmap(pca, X_test_raw, None)
        test_loader = build_dataloader(test_sequences, test_labels, test_features, batch_size=256, shuffle=False)
        
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=AGRO_NT_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(config.device)
        
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
        model.eval()
        
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(config.device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        ensemble_probs[fold] = np.array(fold_probs)
        test_probs_sum += np.array(fold_probs)
        
        del pca, test_features, model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    test_probs_avg = test_probs_sum / KFOLD
    test_preds = np.argmax(test_probs_avg, axis=1) 
    
    ensemble_test_metrics = calculate_metrics(test_labels, test_preds, test_probs_avg)

    test_df['ensemble_prob_class_1'] = test_probs_avg[:, 1]
    test_df['final_prediction'] = test_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
        
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"==> Test predictions saved to: {OUTPUT_SAVE_PATH}")

    # --- 步骤 5: 结果汇总输出 ---
    def get_avg_std(key):
        vals = [res[key] for res in all_fold_results]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred | Feature: Agro-NT + Standard PCA)")
    print(f"Parameters: SeqLen={MAX_SEQ_LENGTH}, LSTM_Dim={LSTM_HIDDEN_SIZE}, CBAM_Layers={CBAM_LAYERS}")
    
    print("\n【Validation Average】")
    print(f"Sn : {get_avg_std('best_val_Sn')[0]:.3f} ± {get_avg_std('best_val_Sn')[1]:.3f}")
    print(f"Sp : {get_avg_std('best_val_Sp')[0]:.3f} ± {get_avg_std('best_val_Sp')[1]:.3f}")
    print(f"ACC: {get_avg_std('best_val_ACC')[0]:.3f} ± {get_avg_std('best_val_ACC')[1]:.3f}")
    print(f"MCC: {get_avg_std('best_val_MCC')[0]:.3f} ± {get_avg_std('best_val_MCC')[1]:.3f}")
    print(f"F1 : {get_avg_std('best_val_F1')[0]:.3f} ± {get_avg_std('best_val_F1')[1]:.3f}")
    print(f"AUC: {get_avg_std('best_val_AUC')[0]:.3f} ± {get_avg_std('best_val_AUC')[1]:.3f}")

    print("\n【Ensemble Test Metrics (Average Softmax Voting)】")
    print(f"Sn:  {ensemble_test_metrics['Sn']:.3f}")
    print(f"Sp:  {ensemble_test_metrics['Sp']:.3f}")
    print(f"ACC: {ensemble_test_metrics['ACC']:.3f}")
    print(f"MCC: {ensemble_test_metrics['MCC']:.3f}")
    print(f"F1:  {ensemble_test_metrics['F1']:.3f}")
    print(f"AUC: {ensemble_test_metrics['AUC']:.3f}")
    print("=" * 80)

    result_df = pd.DataFrame(all_fold_results)
    ensemble_test_metrics['fold'] = 'Ensemble_Test'
    result_df = pd.concat([result_df, pd.DataFrame([ensemble_test_metrics])], ignore_index=True)
    
    result_save_path = "Model_Performance_Deep_dsRNAPred_AgroNT.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nMetrics saved to: {result_save_path}")