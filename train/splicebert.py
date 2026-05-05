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

# --- 引入 SpliceBert 模型 ---
from multimolecule import RnaTokenizer, SpliceBertModel

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.decomposition import PCA, IncrementalPCA

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

# Path Configuration (请修改为你本地 Splice 模型的实际绝对路径)
SPLICE_BERT_PATH = r'/root/autodl-tmp/Big_Model/SpliceBert/ZhejiangLab-LifeScience/splicebert'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'

# Hyperparameters
SPLICE_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# --- 优化器与学习率调度超参数 ---
LEARNING_RATE = 1e-4       
WEIGHT_DECAY = 1e-5        
STEP_SIZE = 20
GAMMA = 0.5

# Initialize Tokenizer & Model
print(f"Loading SpliceBert from {SPLICE_BERT_PATH}...")
tokenizer = RnaTokenizer.from_pretrained(SPLICE_BERT_PATH)
splice_model = SpliceBertModel.from_pretrained(SPLICE_BERT_PATH).to(device)

# 【重要】仅提取特征，不更新权重，设为评估模式
splice_model.eval() 

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


def extract_Splice_features_to_disk(sequences, max_seq_length, mmap_filename):
    """提取 SpliceBert 原始高维特征并直接落盘，防 OOM"""
    print(f"Extracting features directly to disk: {mmap_filename}...")
    fp = None
    shape = None
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = splice_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            batch_hidden_np = batch_hidden.cpu().half().numpy()
            
            if fp is None:
                hidden_dim = batch_hidden_np.shape[2]
                shape = (len(sequences), max_seq_length, hidden_dim)
                fp = np.memmap(mmap_filename, dtype='float16', mode='w+', shape=shape)
            
            fp[i : i + batch_hidden_np.shape[0]] = batch_hidden_np[:]
            fp.flush()
            
            del inputs, outputs, batch_hidden, batch_hidden_np
            torch.cuda.empty_cache()
            gc.collect()
            
    return np.memmap(mmap_filename, dtype='float16', mode='r', shape=shape)


def chunked_pca_transform_for_memmap(pca_model, X_mmap, indices=None, chunk_size=1000):
    """流式读取 memmap 进行 PCA 降维"""
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


def predict_test_probs(dataloader, model, device):
    """用于测试集推理，仅返回模型预测的概率矩阵"""
    model.eval()
    all_prob = []
    
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            preds = model(features)
            # 获取 softmax 后的概率分布
            all_prob.extend(F.softmax(preds, dim=1).cpu().numpy())
            
    return np.array(all_prob)


def kfold_train(fold, train_features, train_labels, val_features, val_labels, 
                train_sequences, val_sequences, config, epochs, train_loader, val_loader):
    """单折训练过程：剥离了 Test，仅保存最优验证集模型"""
    
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size,
        cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
        dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    
    # --- 修改处 1: 替换为 Adam 和 StepLR ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
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
    X_cv_raw = extract_Splice_features_to_disk(cv_sequences, MAX_SEQ_LENGTH, cv_mmap_path)
    
    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_Splice_features_to_disk(test_sequences, MAX_SEQ_LENGTH, test_mmap_path)

    print("Feature extraction complete. Deleting SpliceBert model to free memory...")
    del splice_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared. Starting Cross Validation.")

    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=SPLICE_HIDDEN_DIM, 
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    all_fold_results = []
    
    # ---------------------------------------------------------
    # 阶段 1：交叉验证训练与保存 (不再此处处理 Test)
    # ---------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreparing Fold {fold + 1}...")
        
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        
        print("Fitting IncrementalPCA over all training data in batches...")
        pca = IncrementalPCA(n_components=SPLICE_HIDDEN_DIM)
        pca_chunk_size = 1000 
        
        for i in range(0, len(train_idx), pca_chunk_size):
            chunk_indices = train_idx[i : i + pca_chunk_size]
            chunk_data = X_cv_raw[chunk_indices].astype(np.float32)
            chunk_flat = chunk_data.reshape(-1, chunk_data.shape[-1])
            pca.partial_fit(chunk_flat)
            
            del chunk_data, chunk_flat
            gc.collect()

        os.makedirs('./pca_features_pkl', exist_ok=True)
        pca_save_path = f"./pca_features_pkl/{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved IncrementalPCA model to {pca_save_path}")
        
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
        del train_features, val_features, train_loader, val_loader, pca
        torch.cuda.empty_cache()
        gc.collect()
    
    # 打印验证集平均结果
    def get_avg_std(key):
        vals = [res[key] for res in all_fold_results]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred | Feature: SpliceBert + IncrementalPCA)")
    print("\n【Validation Average】")
    print(f"Sn : {get_avg_std('best_val_Sn')[0]:.3f} ± {get_avg_std('best_val_Sn')[1]:.3f}")
    print(f"Sp : {get_avg_std('best_val_Sp')[0]:.3f} ± {get_avg_std('best_val_Sp')[1]:.3f}")
    print(f"ACC: {get_avg_std('best_val_ACC')[0]:.3f} ± {get_avg_std('best_val_ACC')[1]:.3f}")
    print(f"MCC: {get_avg_std('best_val_MCC')[0]:.3f} ± {get_avg_std('best_val_MCC')[1]:.3f}")
    print(f"F1 : {get_avg_std('best_val_F1')[0]:.3f} ± {get_avg_std('best_val_F1')[1]:.3f}")
    print(f"AUC: {get_avg_std('best_val_AUC')[0]:.3f} ± {get_avg_std('best_val_AUC')[1]:.3f}")
    print("=" * 80)

    # ---------------------------------------------------------
    # --- 修改处 2: 阶段 2 - 集成测试 (Ensemble Testing) ---
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("Starting Ensemble Testing Phase (Soft Voting)...")
    
    # 初始化一个形状为 (测试样本数, 类别数) 的全零矩阵，用于累加每一折的概率预测
    ensemble_test_probs = np.zeros((len(test_labels), NUM_CLASSES))
    
    for fold in range(KFOLD):
        print(f"Evaluating Fold {fold + 1} on Test Set...")
        
        # 1. 加载当前折对应的 PCA 模型
        pca_save_path = f"./pca_features_pkl/{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'rb') as f:
            pca = pickle.load(f)
            
        # 2. 对 Test 原始特征进行降维 (每次降维都基于当前折训练集的分布)
        test_features_fold = chunked_pca_transform_for_memmap(pca, X_test_raw, None)
        test_loader_fold = build_dataloader(test_sequences, test_labels, test_features_fold, batch_size=256, shuffle=False)
        
        # 3. 初始化并加载当前折对应的最佳网络权重
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
        ).to(config.device)
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
        
        # 4. 预测 Test 概率并累加
        fold_probs = predict_test_probs(test_loader_fold, model, config.device)
        ensemble_test_probs += fold_probs
        
        # 5. 内存清理
        del pca, test_features_fold, test_loader_fold, model
        torch.cuda.empty_cache()
        gc.collect()

    # 将累加的概率除以折数，得到平均概率
    ensemble_test_probs /= KFOLD
    # 取平均概率最大的一项作为最终类别预测
    ensemble_test_preds = np.argmax(ensemble_test_probs, axis=1)
    
    # 计算最终的集成测试指标
    final_test_metrics = calculate_metrics(test_labels, ensemble_test_preds, ensemble_test_probs)

    print("\n【Final Ensemble Test Results】")
    print(f"Test Sn : {final_test_metrics['Sn']:.1%}")
    print(f"Test Sp : {final_test_metrics['Sp']:.1%}")
    print(f"Test ACC: {final_test_metrics['ACC']:.1%}")
    print(f"Test MCC: {final_test_metrics['MCC']:.3f}")
    print(f"Test F1 : {final_test_metrics['F1']:.3f}")
    print(f"Test AUC: {final_test_metrics['AUC']:.3f}")
    print("=" * 80)

    # --- Step 5: Save Results ---
    result_df = pd.DataFrame(all_fold_results)
    result_save_path = "Model_Performance_Deep_dsRNAPred_Splice_Val_Only.xlsx"
    result_df.to_excel(result_save_path, index=False)
    
    # 将最终的 test metrics 也保存为一个独立的文件
    test_result_df = pd.DataFrame([final_test_metrics])
    test_result_save_path = "Ensemble_Test_Performance.xlsx"
    test_result_df.to_excel(test_result_save_path, index=False)
    
    print(f"\nValidation metrics saved to: {result_save_path}")
    print(f"Ensemble Test metrics saved to: {test_result_save_path}")