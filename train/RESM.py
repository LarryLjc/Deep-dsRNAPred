import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import os
import sys
import gc  # 用于显式内存回收
import pickle  # 用于保存和加载 PCA 模型
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# Import transformers components
from transformers import EsmModel, EsmTokenizer, EsmConfig
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.decomposition import PCA

# =============================================================================
# 0. [Ultimate Patch] Define missing config classes
# =============================================================================
class Config: pass
class OptimizerConfig: pass
class DataConfig: pass
class TrainConfig: pass
class TransformerConfig: pass
class LoggingConfig: pass
class ModelConfig: pass
class LossConfig: pass
class SchedulerConfig: pass
class TrainerConfig: pass
class LogConfig: pass
class CallbackConfig: pass
class CheckpointConfig: pass
class ExperimentConfig: pass
class RunnerConfig: pass

# Register these classes to the current module
current_module = sys.modules[__name__]
missing_classes = [
    'Config', 'OptimizerConfig', 'DataConfig', 'TrainConfig', 'TransformerConfig',
    'LoggingConfig', 'ModelConfig', 'LossConfig', 'SchedulerConfig', 'TrainerConfig',
    'LogConfig', 'CallbackConfig', 'CheckpointConfig', 'ExperimentConfig', 'RunnerConfig'
]
for class_name in missing_classes:
    if not hasattr(current_module, class_name):
        setattr(current_module, class_name, Config)

# =============================================================================
# 1. Global Configuration & Setup
# =============================================================================

warnings.filterwarnings("ignore")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path Configuration
RESM_CKPT_PATH = r'/root/autodl-tmp/Big_Model/RESM/RESM-150M-KDNY.ckpt' 
ESM_BASE_MODEL_LOCAL_PATH = '/root/autodl-tmp/Big_Model/esm2_t6_150'

TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

# Saving Configuration
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'
OUTPUT_SAVE_PATH = 'Ensemble_Test_Predictions.xlsx'
RESULT_SAVE_PATH = "Deep_dsRNAPred_Performance.xlsx"

# Hyperparameters
PCA_OUTPUT_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 60
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

# =============================================================================
# 2. RESM Model Loading & Mapping Logic
# =============================================================================

def rna_to_pseudo_protein(seq):
    seq = seq.replace('T', 'U')
    mapping = str.maketrans({'A': 'K', 'U': 'D', 'C': 'N', 'G': 'Y'})
    return seq.translate(mapping)

def get_esm_150m_config():
    """Force 150M Parameter Config"""
    print("Forcing EsmConfig to 150M Architecture (30 Layers, 640 Hidden)...")
    return EsmConfig(
        vocab_size=33,
        hidden_size=640,           
        num_hidden_layers=30,      
        num_attention_heads=20,    
        intermediate_size=2560,    
        max_position_embeddings=1024,
        pad_token_id=1,
        mask_token_id=32,
        token_dropout=True,
        position_embedding_type="rotary",
        use_cache=False
    )

def load_resm_model_offline(ckpt_path, local_model_path, device):
    print(f"Loading Tokenizer from local path: {local_model_path}")
    try:
        tokenizer = EsmTokenizer.from_pretrained(local_model_path, local_files_only=True)
    except Exception as e:
        print(f"Error loading local tokenizer: {e}")
        raise e

    config = get_esm_150m_config()
    model = EsmModel(config)
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = 1
    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'padding_idx'):
        model.embeddings.padding_idx = 1

    print(f"Loading RESM weights from {ckpt_path}...")
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint directly: {e}")
            raise e
        
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k
            if name.startswith('model.'): name = name[6:]
            if name.startswith('esm_model.'): name = name[10:]
            
            if name.startswith('embed_tokens'):
                name = name.replace('embed_tokens', 'embeddings.word_embeddings')
            
            name = name.replace('layers.', 'encoder.layer.')
            name = name.replace('self_attn.k_proj', 'attention.self.key')
            name = name.replace('self_attn.v_proj', 'attention.self.value')
            name = name.replace('self_attn.q_proj', 'attention.self.query')
            name = name.replace('self_attn.out_proj', 'attention.output.dense')
            name = name.replace('self_attn_layer_norm', 'attention.LayerNorm')
            name = name.replace('final_layer_norm', 'LayerNorm')
            name = name.replace('fc1', 'intermediate.dense')
            name = name.replace('fc2', 'output.dense')
            
            if name == 'emb_layer_norm_after.weight': name = 'encoder.emb_layer_norm_after.weight'
            if name == 'emb_layer_norm_after.bias': name = 'encoder.emb_layer_norm_after.bias'
            
            if 'embed_positions' in name or "contact_head" in name or "lm_head" in name or "regression" in name:
                continue

            new_state_dict[name] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(missing)}")
            
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found! Model uses random initialization.")
    
    model.to(device)
    model.eval()
    return tokenizer, model

# =============================================================================
# 3. Aux Config Class
# =============================================================================

class CurrentModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='stage1model_fold_'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix

# =============================================================================
# 4. Attention Modules & Core Network
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
# 5. Data Processing & Utils
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, sequences, labels=None, features=None):
        self.sequences = sequences
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.features is not None:
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return feature, label
            return feature
        return self.sequences[idx]

def build_dataloader(sequences, labels=None, features=None, batch_size=32, shuffle=True):
    dataset = RNADataset(sequences, labels, features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def extract_RESM_features(sequences, max_seq_length, tokenizer, resm_model):
    mapped_sequences = [rna_to_pseudo_protein(seq) for seq in sequences]
    hidden_states_list = []
    
    resm_model.to(device)
    
    for i in range(0, len(mapped_sequences), BATCH_SIZE):
        batch_seq = mapped_sequences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_seq, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_seq_length
        )
        
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = resm_model(**inputs)
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
            
    return torch.cat(hidden_states_list, dim=0).numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc, 'FPR': fpr, 'TPR': sn}

# =============================================================================
# 6. Training & Validation Loop (CV Only)
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

def run_training_cv(cv_df, X_cv_raw, MAX_SEQ_LENGTH):
    cv_sequences = cv_df["Sequence"].tolist()
    y_cv = cv_df["label"].values

    config = CurrentModelConfig(
        max_time_steps=MAX_SEQ_LENGTH, input_size=PCA_OUTPUT_DIM, num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE, save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_fold_results = []
    
    n_cv, seq_len, feat_dim = X_cv_raw.shape

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\n===== Preparing Fold {fold + 1}/{KFOLD} =====")
        
        X_train_raw = X_cv_raw[train_idx]
        X_val_raw = X_cv_raw[val_idx]
        
        train_labels_fold = y_cv[train_idx]
        val_labels_fold = y_cv[val_idx]
        train_seqs_fold = [cv_sequences[i] for i in train_idx]
        val_seqs_fold = [cv_sequences[i] for i in val_idx]
        
        # PCA Fit on Train ONLY
        X_train_flat = X_train_raw.reshape(-1, feat_dim)
        pca = PCA(n_components=PCA_OUTPUT_DIM, random_state=42)
        pca.fit(X_train_flat)
        
        # --- 保存 PCA 模型 ---
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to {pca_save_path}")
        
        # Transform Train and Val
        X_train_red = pca.transform(X_train_flat)
        X_train_fold = X_train_red.reshape(len(train_idx), seq_len, PCA_OUTPUT_DIM).astype(np.float32)
        
        X_val_flat = X_val_raw.reshape(-1, feat_dim)
        X_val_red = pca.transform(X_val_flat)
        X_val_fold = X_val_red.reshape(len(val_idx), seq_len, PCA_OUTPUT_DIM).astype(np.float32)

        del X_train_raw, X_val_raw, X_train_flat, X_val_flat, X_train_red, X_val_red, pca
        gc.collect()
        
        train_loader = build_dataloader(train_seqs_fold, train_labels_fold, X_train_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_seqs_fold, val_labels_fold, X_val_fold, batch_size=BATCH_SIZE, shuffle=False)
        
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps, input_size=config.input_size, cnn_layers=3, cnn_dims=512,
            pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, num_classes=config.num_classes,
            cbam_layers=CBAM_LAYERS
        ).to(config.device)
        
        loss_fn = nn.CrossEntropyLoss().to(config.device)
        
        # --- 【修改】优化器与调度器标准化 ---
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        
        best_val_acc = 0.0
        best_val_metrics = None
        
        for epoch in range(EPOCHS):
            train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
            val_metrics = validate_one_epoch(val_loader, model, config.device)
            scheduler.step()
            
            if val_metrics['ACC'] > best_val_acc:
                best_val_acc = val_metrics['ACC']
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
            
            print(f"Epoch {epoch + 1:3d} | Train ACC: {train_acc:.1%} | "
                  f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | Val F1: {val_metrics['F1']:.3f}")
        
        # 记录内部纯验证集结果
        all_fold_results.append({
            'fold': fold + 1,
            'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
            'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
            'best_val_F1': best_val_metrics['F1'], 'best_val_AUC': best_val_metrics['AUC']
        })

        del model, optimizer, scheduler, train_loader, val_loader
        del X_train_fold, X_val_fold
        torch.cuda.empty_cache()
        gc.collect()
    
    # 汇总并保存 CV 结果
    result_df = pd.DataFrame(all_fold_results)
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\nTraining complete. CV Performance metrics saved to: {RESULT_SAVE_PATH}")


# =============================================================================
# 7. 推理流程 (加载多模型集成与软投票机制)
# =============================================================================

def get_model_predictions(model, dataloader):
    """返回全尺寸 Softmax 概率矩阵 [N, 2]"""
    model.eval()
    all_prob = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            preds = model(features)
            probs = F.softmax(preds, dim=1) 
            all_prob.extend(probs.cpu().numpy())
    return np.array(all_prob)

def run_ensemble_inference(test_df, X_test_raw, MAX_SEQ_LENGTH):
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    n_test, seq_len, feat_dim = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, feat_dim)
    
    # 储存每折产生的二维概率矩阵
    ensemble_probs = np.zeros((KFOLD, len(test_sequences), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"\n--- Inference: Processing Fold {fold + 1}/{KFOLD} ---")
        
        # 1. 加载对应折数的 PCA 模型进行降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(n_test, seq_len, PCA_OUTPUT_DIM).astype(np.float32)
        
        # 2. 构建无 Label 的 Dataloader
        test_dataset = RNADataset(test_sequences, features=test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 3. 初始化并加载最佳 Model
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=PCA_OUTPUT_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 4. 预测概率并存储 (shape: [N, 2])
        fold_probs = get_model_predictions(model, test_loader)
        ensemble_probs[fold] = fold_probs
        
        del pca, test_red, test_features_reduced, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 5. 集成计算：5折预测概率求均值，通过 argmax 判定类别
    print("\n" + "=" * 50)
    print("Calculating Ensemble Predictions...")
    final_avg_probs = np.mean(ensemble_probs, axis=0) # shape: [N, 2]
    final_preds = np.argmax(final_avg_probs, axis=1)

    # 保存预测文件
    test_df['ensemble_prob_class_1'] = final_avg_probs[:, 1]
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # 若含有真实标签，计算性能并将结果追加
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n【Ensemble Test Final Metrics】")
        print(f"ACC: {metrics['ACC']:.3f} | MCC: {metrics['MCC']:.3f} | F1: {metrics['F1']:.3f} | AUC: {metrics['AUC']:.3f}")
        print("=" * 50)
        
        try:
            res_df = pd.read_excel(RESULT_SAVE_PATH)
            metrics['fold'] = 'Ensemble_Test'
            res_df = pd.concat([res_df, pd.DataFrame([metrics])], ignore_index=True)
            res_df.to_excel(RESULT_SAVE_PATH, index=False)
            print(f"Ensemble metrics appended to {RESULT_SAVE_PATH}")
        except Exception as e:
            print(f"Failed to append to {RESULT_SAVE_PATH}: {e}")


# =============================================================================
# 8. Main Execution
# =============================================================================

if __name__ == "__main__":
    
    # --- 控制开关 ---
    DO_TRAIN = True         # 执行模型训练和 PCA 保存
    DO_INFERENCE = True     # 执行多模型加载、集成打分与预测
    
    cv_df_global = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df_global = pd.read_excel(TEST_EXCEL_PATH)

    assert "label" in cv_df_global.columns and "Sequence" in cv_df_global.columns
    assert "label" in test_df_global.columns and "Sequence" in test_df_global.columns

    print(f"CV Dataset Size: {len(cv_df_global)}")
    print(f"Test Dataset Size: {len(test_df_global)}")

    cv_sequences = cv_df_global["Sequence"].tolist()
    test_sequences = test_df_global["Sequence"].tolist()

    GLOBAL_MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Auto-detected Max Sequence Length: {GLOBAL_MAX_SEQ_LENGTH}")

    # --- 统一前置：提取所有所需的 Raw Feature ---
    # 【极致显存优化策略】：统一利用 RESM 提取所需特征，完成后删除大模型释放 VRAM 给训练使用。
    print("\n--- Phase 1: Global Feature Extraction ---")
    
    tokenizer, resm_model = load_resm_model_offline(RESM_CKPT_PATH, ESM_BASE_MODEL_LOCAL_PATH, device)
    
    X_cv_raw = None
    if DO_TRAIN:
        print("Extracting CV Set Features...")
        X_cv_raw = extract_RESM_features(cv_sequences, GLOBAL_MAX_SEQ_LENGTH, tokenizer, resm_model)
    
    X_test_raw = None
    if DO_INFERENCE:
        print("Extracting Test Set Features...")
        X_test_raw = extract_RESM_features(test_sequences, GLOBAL_MAX_SEQ_LENGTH, tokenizer, resm_model)
    
    # !!! 关键：释放大模型占用的极高显存 !!!
    print("\nDeleting RESM model to free GPU memory for Deep_dsRNAPred training...")
    del resm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # --- 运行主管道 ---
    if DO_TRAIN:
        print("\n" + "#" * 50)
        print(">>> STARTING TRAINING PIPELINE <<<")
        print("#" * 50)
        run_training_cv(cv_df_global, X_cv_raw, GLOBAL_MAX_SEQ_LENGTH)
        
    if DO_INFERENCE:
        print("\n" + "#" * 50)
        print(">>> STARTING ENSEMBLE INFERENCE PIPELINE <<<")
        print("#" * 50)
        run_ensemble_inference(test_df_global, X_test_raw, GLOBAL_MAX_SEQ_LENGTH)