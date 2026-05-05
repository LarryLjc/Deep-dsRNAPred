import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc   # 核心：用于显式内存回收
import os
import random
import pickle  # 新增：用于保存 PCA 模型
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.model_selection import KFold
# Metrics
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
# PCA
from sklearn.decomposition import PCA

# =============================================================================
# 0. 设置全局随机种子
# =============================================================================
def set_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
CADUCEUS_PATH = r'/root/autodl-tmp/Big_Model/CAD/lgq12697/PlantCAD2-Small-l24-d0768'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_combined.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'
OUTPUT_SAVE_PATH = 'Final_Ensemble_Predictions_Caduceus.xlsx' # 保存集成预测结果的路径

# 维度与超参数配置
CADUCEUS_HIDDEN_DIM = 50  # PCA降维目标维度
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64  
EPOCHS = 240
DROPOUT_RATE = 0.5

# --- 【修改】优化器与学习率调度超参数 ---
LEARNING_RATE = 1e-4      # Adam 学习率
WEIGHT_DECAY = 1e-5       # Adam 权重衰减
STEP_SIZE = 20            # StepLR 每训练 20 个 epoch 调整一次
GAMMA = 0.5               # StepLR 每次调整将学习率减半
# ---------------------------------------------

# 模型结构参数
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# 模型加载
print(f"Loading Caduceus from {CADUCEUS_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(CADUCEUS_PATH, trust_remote_code=True, local_files_only=True)
caduceus_model = AutoModel.from_pretrained(CADUCEUS_PATH, trust_remote_code=True, local_files_only=True).to(device)
caduceus_model.eval()

# =============================================================================
# 2. 辅助配置类
# =============================================================================

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='model_'):
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
        
        # --- CNN Blocks ---
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.pool_size = pool_size
        self.cnn_layers = cnn_layers
        self.max_seq_length = max_seq_length
        self.cbam_layers = cbam_layers 
        
        for i in range(cnn_layers):
            if i == 0: kernels = [1]
            elif i == 1: kernels = [3]
            else: kernels = [7,9]
            cnn_block = CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size)
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

        # --- Fully Connected ---
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
        # [B, L, C] -> [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        
        for block in self.cnn_blocks:
            x = block(x) 
        
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x) 
        
        # [B, C, L, 1] -> [B, L, C]
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
# 5. 数据处理与工具函数
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

# 【核心防爆升级 1】: 将提取的特征直接保存到硬盘 (memmap)，不占用系统内存
def extract_Caduceus_features_to_disk(sequences, max_seq_length, mmap_filename):
    """将特征分批提取并直接写入硬盘 (memmap)，彻底消除系统内存爆炸(Killed)"""
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
            
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            if 'attention_mask' in inputs:
                del inputs['attention_mask']
                
            outputs = caduceus_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            # 转为半精度释放 50% 的占用
            batch_hidden_np = batch_hidden.cpu().half().numpy()
            
            # 在提取出第一个 Batch 时，初始化硬盘映射文件
            if fp is None:
                hidden_dim = batch_hidden_np.shape[2]
                shape = (len(sequences), max_seq_length, hidden_dim)
                fp = np.memmap(mmap_filename, dtype='float16', mode='w+', shape=shape)
            
            # 边提取边落盘
            fp[i : i + batch_hidden_np.shape[0]] = batch_hidden_np[:]
            fp.flush() # 强制写入硬盘
            
            # 清理显存与垃圾
            del inputs, outputs, batch_hidden, batch_hidden_np
            torch.cuda.empty_cache()
            gc.collect()
            
    # 返回一个只读的硬盘映射数组，完全不占系统内存
    return np.memmap(mmap_filename, dtype='float16', mode='r', shape=shape)

# 【核心防爆升级 3】：流式读取硬盘进行分块降维
def chunked_pca_transform_for_memmap(pca_model, X_mmap, indices=None, chunk_size=1000):
    """流式读取 memmap 进行 PCA 降维，把巨大的硬盘数据压成超小的内存数组"""
    if indices is None:
        indices = np.arange(X_mmap.shape[0])
        
    res = []
    # 每次只从硬盘加载 1000 条序列，降维后体积缩小十几倍，再放入内存
    for i in range(0, len(indices), chunk_size):
        chunk_idx = indices[i : i + chunk_size]
        chunk_data = X_mmap[chunk_idx].astype(np.float32) # (chunk, L, C)
        chunk_flat = chunk_data.reshape(-1, chunk_data.shape[-1])
        
        transformed_flat = pca_model.transform(chunk_flat)
        transformed = transformed_flat.reshape(len(chunk_idx), chunk_data.shape[1], -1)
        res.append(transformed)
        
    return np.vstack(res)

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0) # TPR
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1, 'FPR': fpr, 'TPR': sn}

# =============================================================================
# 6. 训练与验证流程
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
    # 初始化模型
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
    
    # === 【修改】使用 Adam 和 StepLR ===
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    # ========================================
    
    best_val_acc = 0.0
    best_val_metrics = None
    print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM: {CBAM_LAYERS}) =====")
    
    for epoch in range(epochs):
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        
        # 更新学习率
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
        
        print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # --- 【修改】移除训练阶段内对测试集的评估 ---
    return {
        'fold': fold + 1,
        'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
        'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
        'best_val_AUC': best_val_metrics['AUC'], 'best_val_F1': best_val_metrics['F1']
    }

# =============================================================================
# 7. 主执行程序
# =============================================================================

if __name__ == "__main__":
    # --- 步骤 1: 读取数据 ---
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    assert "label" in cv_df.columns and "Sequence" in cv_df.columns
    assert "label" in test_df.columns and "Sequence" in test_df.columns

    print(f"CV Data: {len(cv_df)} (Pos: {sum(cv_df['label'])}, Neg: {len(cv_df)-sum(cv_df['label'])})")
    print(f"Test Data: {len(test_df)} (Pos: {sum(test_df['label'])}, Neg: {len(test_df)-sum(test_df['label'])})")

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = np.array(cv_df["label"].tolist())
    test_sequences = test_df["Sequence"].tolist()
    test_labels = np.array(test_df["label"].tolist())

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Max Seq Len: {MAX_SEQ_LENGTH}")

    # --- 步骤 2: 将特征提取直接绑定到硬盘 ---
    cv_mmap_path = '/root/autodl-tmp/cv_features.dat'
    test_mmap_path = '/root/autodl-tmp/test_features.dat'
    
    X_cv_raw = extract_Caduceus_features_to_disk(cv_sequences, MAX_SEQ_LENGTH, cv_mmap_path)
    X_test_raw = extract_Caduceus_features_to_disk(test_sequences, MAX_SEQ_LENGTH, test_mmap_path)

    # 释放大模型占用显存
    del caduceus_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # --- 步骤 3: 5折交叉验证 ---
    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=CADUCEUS_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=3407) 
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreparing Fold {fold + 1}...")
        
        # 仅随机抽取 1000 条数据用来拟合 PCA 模型，足以捕获特征，防止内存崩溃
        sample_size = min(1000, len(train_idx))
        sampled_indices = np.random.choice(train_idx, size=sample_size, replace=False)
        sampled_indices.sort() # 为了硬盘读取更高效，排个序
        
        sampled_data = X_cv_raw[sampled_indices].astype(np.float32)
        train_flat_sampled = sampled_data.reshape(-1, sampled_data.shape[-1])

        pca = PCA(n_components=CADUCEUS_HIDDEN_DIM, random_state=3407)
        pca.fit(train_flat_sampled)
        del sampled_data, train_flat_sampled  # 拟合完立即丢弃采样集

        # 保存 PCA
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to {pca_save_path}")
        
        # 依次流式分块降维，获取降维后的轻量级特征矩阵
        train_features = chunked_pca_transform_for_memmap(pca, X_cv_raw, train_idx)
        val_features = chunked_pca_transform_for_memmap(pca, X_cv_raw, val_idx)
        # 注意：此处不再对 test_features 进行降维和 DataLoader 的创建，移至最后的集成评估阶段
        
        train_labels_fold = cv_labels[train_idx]
        val_labels_fold = cv_labels[val_idx]
        
        train_sequences_fold = [cv_sequences[i] for i in train_idx]
        val_sequences_fold = [cv_sequences[i] for i in val_idx]
        
        # 建立 DataLoader
        train_loader = build_dataloader(train_sequences_fold, train_labels_fold, train_features, batch_size=256, shuffle=True) 
        val_loader = build_dataloader(val_sequences_fold, val_labels_fold, val_features, batch_size=256, shuffle=False)
        
        # 训练 (不再传入 test_loader)
        fold_result = kfold_train(fold, train_features, train_labels_fold, val_features, val_labels_fold, 
                                  train_sequences_fold, val_sequences_fold, config, EPOCHS, 
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
        
        # 1. 加载当前折对应的 PCA 模型
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'rb') as f:
            pca = pickle.load(f)
            
        # 2. 对硬盘上的 test memmap 进行分块降维提取
        test_features = chunked_pca_transform_for_memmap(pca, X_test_raw, None)
        
        # 3. 准备测试集 DataLoader
        test_loader = build_dataloader(test_sequences, test_labels, test_features, batch_size=256, shuffle=False)
        
        # 4. 加载当前折保存的最佳 NN 模型参数
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=CADUCEUS_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(config.device)
        
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
        model.eval()
        
        # 5. 预测概率打分
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(config.device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        ensemble_probs[fold] = np.array(fold_probs)
        test_probs_sum += np.array(fold_probs)
        
        # 内存回收
        del pca, test_features, model, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # 对 5 折预测概率求平均
    test_probs_avg = test_probs_sum / KFOLD
    
    # 按照阈值 (argmax等效于 0.5 阈值) 判定最终类别
    test_preds = np.argmax(test_probs_avg, axis=1) 
    
    # 计算集成测试集的 Metrics
    ensemble_test_metrics = calculate_metrics(test_labels, test_preds, test_probs_avg)

    # 将预测结果写入 test_df 并保存
    test_df['ensemble_prob_class_1'] = test_probs_avg[:, 1]
    test_df['final_prediction'] = test_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
        
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"==> Test predictions saved to: {OUTPUT_SAVE_PATH}")


    # --- 步骤 5: 结果汇总输出 ---
    get_avg = lambda key: np.mean([res[key] for res in all_fold_results])
    get_std = lambda key: np.std([res[key] for res in all_fold_results])
    
    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred, Feature: Caduceus+StrictPCA({CADUCEUS_HIDDEN_DIM}))")
    print(f"Max Seq Len: {MAX_SEQ_LENGTH} | CBAM Layers: {CBAM_LAYERS}")
    
    print("\n【Validation Avg Metrics】")
    print(f"Sn:  {get_avg('best_val_Sn'):.3f} ± {get_std('best_val_Sn'):.3f}")
    print(f"Sp:  {get_avg('best_val_Sp'):.3f} ± {get_std('best_val_Sp'):.3f}")
    print(f"ACC: {get_avg('best_val_ACC'):.3f} ± {get_std('best_val_ACC'):.3f}")
    print(f"MCC: {get_avg('best_val_MCC'):.3f} ± {get_std('best_val_MCC'):.3f}")
    print(f"F1:  {get_avg('best_val_F1'):.3f} ± {get_std('best_val_F1'):.3f}")
    print(f"AUC: {get_avg('best_val_AUC'):.3f} ± {get_std('best_val_AUC'):.3f}")
    
    print("\n【Ensemble Test Metrics (Average Softmax Voting)】")
    print(f"Sn:  {ensemble_test_metrics['Sn']:.3f}")
    print(f"Sp:  {ensemble_test_metrics['Sp']:.3f}")
    print(f"ACC: {ensemble_test_metrics['ACC']:.3f}")
    print(f"MCC: {ensemble_test_metrics['MCC']:.3f}")
    print(f"F1:  {ensemble_test_metrics['F1']:.3f}")
    print(f"AUC: {ensemble_test_metrics['AUC']:.3f}")
    print("=" * 80)

    # --- 步骤 6: 保存性能结果 ---
    result_df = pd.DataFrame(all_fold_results)
    
    # 增加一行记录集成的 Test 结果
    ensemble_test_metrics['fold'] = 'Ensemble_Test'
    result_df = pd.concat([result_df, pd.DataFrame([ensemble_test_metrics])], ignore_index=True)
    
    result_save_path = "Deep_dsRNAPred_Performance.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nPerformance metrics saved to: {result_save_path}")