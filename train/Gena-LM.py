import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc   
import os
import random
import pickle  
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
from sklearn.decomposition import IncrementalPCA

# =============================================================================
# 0. 设置全局随机种子
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

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", category=FutureWarning) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
GENA_LM_PATH = r'/root/autodl-tmp/Big_Model/Gena-LM/lgq12697/gena-lm-bert-base-t2t'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_combined.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'  

# 维度与超参数配置
GENA_LM_HIDDEN_DIM = 50  
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5

# --- 修改：优化器超参数配置 ---
LEARNING_RATE = 1e-4      
WEIGHT_DECAY = 1e-5       
STEP_SIZE = 20
GAMMA = 0.5

# 模型结构参数
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# 模型加载
print(f"Loading Gena-LM from {GENA_LM_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(GENA_LM_PATH, trust_remote_code=True, local_files_only=True)
gena_lm_model = AutoModel.from_pretrained(GENA_LM_PATH, trust_remote_code=True, local_files_only=True).to(device)
gena_lm_model.eval()

# =============================================================================
# --- 位置编码线性插值函数 ---
# =============================================================================
def resize_pos_embeddings(model, new_max_length):
    """Interpolate position embeddings if sequence length exceeds model limit."""
    config = model.config
    current_max_pos = getattr(config, 'max_position_embeddings', 512)
    
    if new_max_length > current_max_pos:
        print(f"\n[Warning] Input length {new_max_length} > model limit {current_max_pos}.")
        print(f"Doing position embedding interpolation to support {new_max_length} tokens...")
        
        base_model = getattr(model, model.base_model_prefix, model)
        
        if hasattr(base_model, 'embeddings') and hasattr(base_model.embeddings, 'position_embeddings'):
            old_embeddings = base_model.embeddings.position_embeddings.weight.data
            old_embeddings_t = old_embeddings.t().unsqueeze(0)
            
            # 使用线性插值拉伸特征维度
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
            print("Position embeddings resized successfully.\n")
        else:
            print("[Warning] Standard position_embeddings not found. Make sure the model uses absolute position embeddings.")


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


def extract_GENA_LM_features(sequences, max_seq_length):
    """提取 Gena-LM 原始高维特征"""
    
    resize_pos_embeddings(gena_lm_model, max_seq_length)
    
    num_samples = len(sequences)
    hidden_size = getattr(gena_lm_model.config, 'hidden_size', 1024) 
    
    print(f"Allocating memory for features: {num_samples} samples, shape=({num_samples}, {max_seq_length}, {hidden_size})")
    all_hidden = np.zeros((num_samples, max_seq_length, hidden_size), dtype=np.float16)
    
    base_model = getattr(gena_lm_model, gena_lm_model.base_model_prefix, gena_lm_model)
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            
            # Gena-LM 是 DNA 模型，替换 U 为 T
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
            
            del inputs, outputs, batch_hidden
            torch.cuda.empty_cache()
            
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"Processed batch {i // BATCH_SIZE + 1}/{(num_samples + BATCH_SIZE - 1) // BATCH_SIZE}", end='\r')
                
    print("\nFeature extraction complete.")
    return all_hidden


def fit_transform_pca_strict(X_train, X_val, n_components=50, batch_size=4096):
    """【修改】仅对 Train 和 Val 进行 PCA 处理，Test 集移至最后集成阶段"""
    print(f"Fitting IncrementalPCA on TRAINING data only (Dim: {X_train.shape[-1]} -> {n_components})...")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    N_train, L, C = X_train.shape
    pca_chunk_size = 100 
    
    for i in range(0, N_train, pca_chunk_size):
        chunk = X_train[i : i + pca_chunk_size]
        chunk_flat = chunk.reshape(-1, C) 
        ipca.partial_fit(chunk_flat)
    
    print("PCA Fit complete. Transforming Train, Val sets...")

    def transform_data(X_data):
        if X_data is None: return None
        N, L, C = X_data.shape
        X_reduced_list = []
        for i in range(0, N, pca_chunk_size):
            chunk = X_data[i : i + pca_chunk_size]
            chunk_flat = chunk.reshape(-1, C)
            chunk_reduced = ipca.transform(chunk_flat)
            chunk_3d = chunk_reduced.reshape(chunk.shape[0], L, n_components)
            X_reduced_list.append(chunk_3d.astype(np.float32))
        return np.concatenate(X_reduced_list, axis=0)

    X_train_pca = transform_data(X_train)
    X_val_pca = transform_data(X_val)
    
    return X_train_pca, X_val_pca, ipca


def transform_pca_inference(X_data, ipca, n_components=50):
    """【新增】专用于测试集推理的独立 PCA 转换函数"""
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
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
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
                train_sequences, val_sequences, config, epochs):
    """【修改】剥离测试集推理过程，只保留训练和验证"""
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
    
    # --- 修改：使用 Adam 和 StepLR ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
    
    best_val_acc = 0.0
    best_val_metrics = None
    print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM: {CBAM_LAYERS}) =====")
    
    for epoch in range(epochs):
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            # 只保存验证集效果最好的一次权重
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
        
        print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    print(f"Fold {fold + 1} Best Validation Result | "
          f"Val ACC: {best_val_metrics['ACC']:.1%} | Val MCC: {best_val_metrics['MCC']:.3f} | "
          f"Val AUC: {best_val_metrics['AUC']:.3f}")
    
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
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
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    assert "label" in cv_df.columns and "Sequence" in cv_df.columns
    assert "label" in test_df.columns and "Sequence" in test_df.columns

    print(f"CV Data: {len(cv_df)} (Pos: {sum(cv_df['label'])}, Neg: {len(cv_df)-sum(cv_df['label'])})")
    print(f"Test Data: {len(test_df)} (Pos: {sum(test_df['label'])}, Neg: {len(test_df)-sum(test_df['label'])})")

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].tolist()

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences]) + 2
    print(f"Max Seq Len (including special tokens): {MAX_SEQ_LENGTH}")

    print("Extracting Raw Features (CV)...")
    X_cv_raw = extract_GENA_LM_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values

    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_GENA_LM_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values

    del gena_lm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("\nGena-LM model deleted. Memory cleared for K-Fold training.")

    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=GENA_LM_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=3407) 
    all_fold_results = []
    
    # ---------------------------------------------------------
    # 第一阶段：K-Fold 交叉验证训练 (不含测试集评估)
    # ---------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreparing Fold {fold + 1}...")
        
        X_train_raw = X_cv_raw[train_idx]
        train_labels = y_cv[train_idx]
        train_sequences = [cv_sequences[i] for i in train_idx]
        
        X_val_raw = X_cv_raw[val_idx]
        val_labels = y_cv[val_idx]
        val_sequences = [cv_sequences[i] for i in val_idx]
        
        # 仅拟合/转换 Train 和 Val
        X_train_fold, X_val_fold, ipca_model = fit_transform_pca_strict(
            X_train_raw, X_val_raw, n_components=GENA_LM_HIDDEN_DIM
        )

        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(ipca_model, f)
        print(f"Saved PCA model to {pca_save_path}")
        
        fold_result = kfold_train(fold, X_train_fold, train_labels, X_val_fold, val_labels, 
                                  train_sequences, val_sequences, config, EPOCHS)
        all_fold_results.append(fold_result)

        print(f"Cleaning up Fold {fold + 1} memory...")
        del X_train_raw, X_val_raw
        del X_train_fold, X_val_fold, ipca_model
        gc.collect()
        
    # ---------------------------------------------------------
    # 第二阶段：独立集成测试 (Soft Voting)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(">>> Starting Ensemble Testing on Independent Test Set <<<")
    
    n_test = X_test_raw.shape[0]
    ensemble_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)
    
    for fold in range(KFOLD):
        print(f"Loading Fold {fold + 1} PCA and Model for Inference...")
        
        # 1. 加载本折保存的 PCA 模型，转换 Test 数据
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'rb') as f:
            ipca_model = pickle.load(f)
        X_test_pca = transform_pca_inference(X_test_raw, ipca_model, n_components=GENA_LM_HIDDEN_DIM)
        
        # 2. 构建数据加载器 (千万不能 shuffle，保证顺序对齐)
        test_loader = build_dataloader(test_sequences, test_labels, X_test_pca, batch_size=BATCH_SIZE, shuffle=False)
        
        # 3. 加载本折最好的神经网络模型
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps, input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
        ).to(device)
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=device))
        model.eval()
        
        # 4. 预测概率并累加
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        ensemble_probs += np.array(fold_probs)
        
        # 释放资源
        del model, ipca_model, X_test_pca, test_loader
        gc.collect()
        
    # 5. 计算均值，并以 0.5 为阈值通过 argmax 得出最终标签
    ensemble_probs /= KFOLD
    final_preds = np.argmax(ensemble_probs, axis=1)
    
    # 6. 计算最终集成测试指标
    test_metrics = calculate_metrics(y_test, final_preds, ensemble_probs)
    
    
    # ---------------------------------------------------------
    # 结果打印与保存
    # ---------------------------------------------------------
    get_avg = lambda key: np.mean([res[key] for res in all_fold_results])
    get_std = lambda key: np.std([res[key] for res in all_fold_results])
    
    print("\n" + "=" * 80)
    print(f"Summary (Model: Deep_dsRNAPred, Feature: Gena-LM+StrictPCA({GENA_LM_HIDDEN_DIM}))")
    print(f"Max Seq Len: {MAX_SEQ_LENGTH} | CBAM Layers: {CBAM_LAYERS}")
    print(f"Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}) | StepLR: ({STEP_SIZE}, {GAMMA})")
    
    print("\n【Validation Avg Metrics (5 Folds)】")
    print(f"Sn:  {get_avg('best_val_Sn'):.3f} ± {get_std('best_val_Sn'):.3f}")
    print(f"Sp:  {get_avg('best_val_Sp'):.3f} ± {get_std('best_val_Sp'):.3f}")
    print(f"ACC: {get_avg('best_val_ACC'):.3f} ± {get_std('best_val_ACC'):.3f}")
    print(f"MCC: {get_avg('best_val_MCC'):.3f} ± {get_std('best_val_MCC'):.3f}")
    print(f"F1:  {get_avg('best_val_F1'):.3f} ± {get_std('best_val_F1'):.3f}")
    print(f"AUC: {get_avg('best_val_AUC'):.3f} ± {get_std('best_val_AUC'):.3f}")
    
    print("\n【Final Ensemble Test Metrics (Soft Voting)】")
    print(f"Sn:  {test_metrics['Sn']: .3f}")
    print(f"Sp:  {test_metrics['Sp']: .3f}")
    print(f"ACC: {test_metrics['ACC']:.3f}")
    print(f"MCC: {test_metrics['MCC']:.3f}")
    print(f"F1:  {test_metrics['F1']: .3f}")
    print(f"AUC: {test_metrics['AUC']:.3f}")
    print("=" * 80)

    # 结果保存：将每一折的信息与最后集成测试结果统一
    result_df = pd.DataFrame(all_fold_results)
    ensemble_row = {
        'fold': 'Ensemble_Test',
        'best_val_Sn': np.nan, 'best_val_Sp': np.nan, 'best_val_ACC': np.nan, 
        'best_val_MCC': np.nan, 'best_val_AUC': np.nan, 'best_val_F1': np.nan,
        'test_Sn': test_metrics['Sn'], 'test_Sp': test_metrics['Sp'],
        'test_ACC': test_metrics['ACC'], 'test_MCC': test_metrics['MCC'],
        'test_AUC': test_metrics['AUC'], 'test_F1': test_metrics['F1'],
        'test_FPR': test_metrics['FPR'], 'test_TPR': test_metrics['TPR']
    }
    result_df = pd.concat([result_df, pd.DataFrame([ensemble_row])], ignore_index=True)
    
    result_save_path = "Deep_dsRNAPred_GenaLM_Performance.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nPerformance metrics saved to: {result_save_path}")