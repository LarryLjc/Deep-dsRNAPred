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

# --- 引入 Tokenizer 和原生的 AutoModel ---
from multimolecule import RnaTokenizer
from transformers import AutoModel

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.decomposition import PCA

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

# Path Configuration (请修改为你本地 UtrLm 的实际路径)
UTR_LM_PATH = r'/root/autodl-tmp/Big_Model/utrlm'   
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
SAVE_MODEL_PREFIX = 'stage1model_fold_'

# Hyperparameters
UTR_LM_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# --- 【修改为 Adam 与 StepLR 相关的参数】 ---
LEARNING_RATE = 1e-4      # Adam 初始学习率
WEIGHT_DECAY = 1e-5       # Adam 权重衰减
STEP_SIZE = 20            # StepLR: 每 20 个 epoch 调整一次
GAMMA = 0.5               # StepLR: 每次衰减系数为 0.5

# Initialize Tokenizer & Model
print(f"Loading UtrLm from {UTR_LM_PATH}...")
tokenizer = RnaTokenizer.from_pretrained(UTR_LM_PATH)
utr_lm_model = AutoModel.from_pretrained(UTR_LM_PATH, trust_remote_code=True).to(device)

# =============================================================================
# --- 核心新增：强行解决权重前缀不匹配 & 拦截 LFS 假文件 ---
# =============================================================================
def fix_and_load_weights(model, model_path):
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safe_path = os.path.join(model_path, "model.safetensors")

    weight_file = None
    if os.path.exists(bin_path):
        weight_file = bin_path
    elif os.path.exists(safe_path):
        weight_file = safe_path

    if weight_file is None:
        print("[Warning] Directory does not contain pytorch_model.bin or model.safetensors.")
        return

    # 1. 检查是否为 Git LFS 指针假文件 (小于 1 MB 绝对不可能是大模型权重)
    file_size_mb = os.path.getsize(weight_file) / (1024 * 1024)
    if file_size_mb < 1.0:
        raise ValueError(
            f"\n{'='*70}\n"
            f"[致命错误] 你的权重文件太小了！(仅 {file_size_mb:.3f} MB)\n"
            f"这意味着你用 git clone 下载时没有成功触发 Git LFS，下载到的只是一个文本指针文件。\n"
            f"解决办法：请手动去 Hugging Face 网页端下载 {os.path.basename(weight_file)} 大文件，"
            f"然后上传替换掉你服务器上的同名小文件！\n"
            f"{'='*70}"
        )

    # 2. 手动读取并对齐键名
    print(f"Valid model weight found ({file_size_mb:.1f} MB). Aligning dictionary keys...")
    if weight_file.endswith(".bin"):
        state_dict = torch.load(weight_file, map_location="cpu")
    else:
        try:
            from safetensors.torch import load_file
            state_dict = load_file(weight_file)
        except ImportError:
            print("[Warning] safetensors library missing. Cannot load .safetensors file.")
            return

    new_state_dict = {}
    for k, v in state_dict.items():
        # 剥离所有可能导致报错的前缀外壳
        new_k = k
        for prefix in ["utrlm_model.", "utrlm.", "roberta.", "bert.", "model.", "base_model."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        new_state_dict[new_k] = v

    # 强行灌入模型
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    core_missing = [k for k in missing if 'encoder' in k]
    if len(core_missing) == 0:
        print("Success: All core encoder weights successfully aligned and loaded!\n")
    else:
        print(f"Warning: Still missing {len(core_missing)} core weights. You might be using a completely incompatible checkpoint.")

# 强行对齐权重并设置为推理模式
fix_and_load_weights(utr_lm_model, UTR_LM_PATH)
utr_lm_model.eval()
# =============================================================================

# --- 位置编码线性插值函数 ---
def resize_pos_embeddings(model, new_max_length):
    """Interpolate position embeddings if sequence length exceeds model limit."""
    config = model.config
    current_max_pos = config.max_position_embeddings
    
    if new_max_length > current_max_pos:
        print(f"\n[Warning] Input length {new_max_length} > model limit {current_max_pos}.")
        print(f"Doing position embedding interpolation to support {new_max_length} tokens...")
        
        # 智能获取底层模型，定位 position_embeddings
        base_model = getattr(model, model.base_model_prefix, model)
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
        print("Position embeddings resized successfully.\n")


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


def extract_UTR_LM_features(sequences, max_seq_length):
    # 调用线性插值
    resize_pos_embeddings(utr_lm_model, max_seq_length)

    hidden_states_list = []
    # Use torch.no_grad to save memory during extraction
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            
            # 引入 truncation=True 以防特殊情况，因为已经插值支持了 max_seq_length
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = utr_lm_model(**inputs)
            # 提取最后一层隐藏状态作为序列特征
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            # 双重保险：确保张量对齐
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
                train_sequences, val_sequences, config, epochs):
    
    # 1. Initialize Model
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size,
        cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
        dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
    ).to(config.device)
    
    # 2. Optimization
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    
    # --- 【修改】使用 Adam 优化器 和 StepLR ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # 3. Data Loaders
    train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
    
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
        
        # Formatted Output
        print(f"Epoch {epoch + 1:3d} | Train ACC: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # --- MEMORY CLEANUP ---
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
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
    # --- Step 1: Data Loading ---
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].tolist()

    # Determine Max Seq Length. 加 2 是为了预留 [CLS] 和 [SEP] 等特殊 Token 的位置
    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences]) + 2
    print(f"\nMax Sequence Length (including special tokens): {MAX_SEQ_LENGTH}\n")

    # --- Step 2: Feature Extraction (Raw) ---
    print("Extracting Raw Features (Train)...")
    X_cv_raw = extract_UTR_LM_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values

    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_UTR_LM_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values

    # --- CRITICAL STEP: DELETE MODEL TO PREVENT OOM KILL ---
    print("Feature extraction complete. Deleting UtrLm model to free memory...")
    del utr_lm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    print("Memory cleared. Starting Cross Validation.")

    # --- Step 3: Cross Validation with Strict PCA ---
    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=UTR_LM_HIDDEN_DIM, 
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    all_fold_results = []
    
    # Original shape for PCA reshaping
    n_cv, seq_len, feat_dim = X_cv_raw.shape
    n_test = X_test_raw.shape[0]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        # Split Raw Data
        X_train_raw = X_cv_raw[train_idx]
        X_val_raw = X_cv_raw[val_idx]
        
        # Flatten for PCA: (N * L, C)
        X_train_flat = X_train_raw.reshape(-1, feat_dim)
        X_val_flat = X_val_raw.reshape(-1, feat_dim)
        
        # PCA Fit on Train ONLY
        pca = PCA(n_components=UTR_LM_HIDDEN_DIM, random_state=SEED)
        X_train_red = pca.fit_transform(X_train_flat)
        
        # --- 确认并保存每一折的 PCA 模型 ---
        pca_save_path = f'pca_model_fold_{fold+1}.pkl'
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        
        if os.path.exists(pca_save_path):
            print(f"Successfully saved {pca_save_path}")
        else:
            print(f"ERROR: Failed to save {pca_save_path}")
        # -----------------------------------------------
            
        # Transform Val using Train's PCA
        X_val_red = pca.transform(X_val_flat)
        
        # Reshape Back: (N, L, 50)
        X_train_fold = X_train_red.reshape(len(train_idx), seq_len, UTR_LM_HIDDEN_DIM)
        X_val_fold = X_val_red.reshape(len(val_idx), seq_len, UTR_LM_HIDDEN_DIM)
        
        # Prepare Lists
        train_labels_fold = y_cv[train_idx]
        val_labels_fold = y_cv[val_idx]
        train_seqs_fold = [cv_sequences[i] for i in train_idx]
        val_seqs_fold = [cv_sequences[i] for i in val_idx]
        
        # Run Training (去除了 test 数据传递)
        fold_result = kfold_train(
            fold, X_train_fold, train_labels_fold, X_val_fold, val_labels_fold,
            train_seqs_fold, val_seqs_fold, config, EPOCHS
        )
        all_fold_results.append(fold_result)
        
        # Clean up Fold Specific Variables
        del X_train_raw, X_val_raw, X_train_flat, X_val_flat
        del X_train_red, X_val_red
        del X_train_fold, X_val_fold
        del pca
        gc.collect()

    
    # --- Step 4: Ensemble Testing on Independent Test Set ---
    print("\n" + "=" * 80)
    print("Starting Ensemble Testing on Test Set (Soft Voting)...")
    
    # 用于累加 5 折模型对 Test 数据的预测概率
    ensemble_test_probs = np.zeros((n_test, NUM_CLASSES))
    
    for fold in range(KFOLD):
        print(f"Loading fold {fold + 1} PCA and Model for ensemble...")
        
        # 1. 加载此折的 PCA
        pca_save_path = f'pca_model_fold_{fold+1}.pkl'
        with open(pca_save_path, 'rb') as f:
            pca = pickle.load(f)
            
        # 2. 用此折的 PCA 对原始 Test 特征进行转换
        X_test_flat = X_test_raw.reshape(-1, feat_dim)
        X_test_red = pca.transform(X_test_flat)
        X_test_fold = X_test_red.reshape(n_test, seq_len, UTR_LM_HIDDEN_DIM)
        
        # 3. 初始化并加载此折的最优模型权重
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=config.dropout, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
        ).to(config.device)
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
        model.eval()
        
        # 4. 构建 DataLoader 并推理获取概率
        test_loader = build_dataloader(test_sequences, y_test, X_test_fold, batch_size=BATCH_SIZE, shuffle=False)
        
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(config.device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        # 累加这折的概率
        ensemble_test_probs += np.array(fold_probs)
        
        # 内存清理
        del model, pca, test_loader, X_test_fold, X_test_red
        torch.cuda.empty_cache()
        gc.collect()

    # 将累加的概率除以折数，求得平均预测概率
    ensemble_test_probs /= KFOLD
    
    # 根据平均概率打分（argmax 实际上就是按照 0.5 阈值比较）
    final_test_preds = np.argmax(ensemble_test_probs, axis=1)
    
    # 计算最终的集成 Test 指标
    test_metrics = calculate_metrics(y_test, final_test_preds, ensemble_test_probs)

    # --- Step 5: Summary Print & Save ---
    def get_avg_std(key):
        vals = [res[key] for res in all_fold_results]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 80)
    print(f"5-Fold CV & Ensemble Summary (Model: Deep_dsRNAPred | Feature: UtrLm + Local PCA)")
    print(f"Parameters: SeqLen={MAX_SEQ_LENGTH}, LSTM_Dim={LSTM_HIDDEN_SIZE}, CBAM_Layers={CBAM_LAYERS}")
    
    print("\n【Validation Average】 (From each fold)")
    print(f"Sn : {get_avg_std('best_val_Sn')[0]:.3f} ± {get_avg_std('best_val_Sn')[1]:.3f}")
    print(f"Sp : {get_avg_std('best_val_Sp')[0]:.3f} ± {get_avg_std('best_val_Sp')[1]:.3f}")
    print(f"ACC: {get_avg_std('best_val_ACC')[0]:.3f} ± {get_avg_std('best_val_ACC')[1]:.3f}")
    print(f"MCC: {get_avg_std('best_val_MCC')[0]:.3f} ± {get_avg_std('best_val_MCC')[1]:.3f}")
    print(f"F1 : {get_avg_std('best_val_F1')[0]:.3f} ± {get_avg_std('best_val_F1')[1]:.3f}")
    print(f"AUC: {get_avg_std('best_val_AUC')[0]:.3f} ± {get_avg_std('best_val_AUC')[1]:.3f}")

    print("\n【Ensemble Test Result】 (Average Probability Soft Voting)")
    print(f"Sn : {test_metrics['Sn']:.3f}")
    print(f"Sp : {test_metrics['Sp']:.3f}")
    print(f"ACC: {test_metrics['ACC']:.3f}")
    print(f"MCC: {test_metrics['MCC']:.3f}")
    print(f"F1 : {test_metrics['F1']:.3f}")
    print(f"AUC: {test_metrics['AUC']:.3f}")
    print("=" * 80)

    # 保存验证集的详细信息以及总的 Test 结果
    result_df = pd.DataFrame(all_fold_results)
    result_save_path = "Model_Performance_Deep_dsRNAPred_UtrLm.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nFold Metrics saved to: {result_save_path}")