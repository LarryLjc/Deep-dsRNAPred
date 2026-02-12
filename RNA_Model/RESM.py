import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import os
import sys
import gc  # Added for memory management
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

TRAIN_EXCEL_PATH = "/root/autodl-tmp/Run/Run/yuchuli/RNAi_train616_New_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/Run/Run/yuchuli/RNAi_test616_New_RNA.xlsx"
SAVE_MODEL_PREFIX = 'stage1model_fold_'

# Hyperparameters
RESM_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 256
EPOCHS = 40
DROPOUT_RATE = 0.2

LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

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

# Load global model initially
tokenizer, resm_model = load_resm_model_offline(RESM_CKPT_PATH, ESM_BASE_MODEL_LOCAL_PATH, device)

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
# 4. Attention Modules
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
# 5. Core Network Architecture: Deep_dsRNAPred
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
        # Input: [B, L, C] -> CNN: [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        for block in self.cnn_blocks:
            x = block(x)
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x)
        
        # To LSTM: [B, L_new, C_new]
        x = x.squeeze(-1).permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        
        for block in self.fc_blocks:
            x = block(x)
        
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)

# =============================================================================
# 6. Data Processing & Utils
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

def extract_RESM_features(sequences, max_seq_length):
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
# 7. Training & Validation Loop
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
                test_raw_features, test_labels, test_sequences,
                train_sequences, val_sequences, config, epochs):
    
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps, input_size=config.input_size, cnn_layers=3, cnn_dims=512,
        pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, num_classes=config.num_classes,
        cbam_layers=CBAM_LAYERS
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    best_val_metrics = None
    
    # Test loader uses features transformed by THIS fold's PCA
    test_loader = build_dataloader(test_sequences, test_labels, test_raw_features, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n===== Fold {fold + 1}/{KFOLD} Training (CBAM Layers: {CBAM_LAYERS}) =====")
    
    for epoch in range(epochs):
        train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
        
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best_Nye_aug.pth")
        
        print(f"Epoch {epoch + 1:3d} | Train ACC: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # Testing
    model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best_Nye_aug.pth", map_location=config.device))
    test_metrics = validate_one_epoch(test_loader, model, config.device)
    
    print(f"Fold {fold + 1} Best Result | Test Sn: {test_metrics['Sn']:.1%} | "
          f"Test Sp: {test_metrics['Sp']:.1%} | Test ACC: {test_metrics['ACC']:.1%} | "
          f"Test MCC: {test_metrics['MCC']:.3f} | Test F1: {test_metrics['F1']:.3f} | Test AUC: {test_metrics['AUC']:.3f}")
    
    # Cleanup memory
    del model, optimizer, scheduler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'fold': fold + 1,
        'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
        'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
        'best_val_F1': best_val_metrics['F1'], 'best_val_AUC': best_val_metrics['AUC'], 
        'test_Sn': test_metrics['Sn'], 'test_Sp': test_metrics['Sp'], 
        'test_ACC': test_metrics['ACC'], 'test_MCC': test_metrics['MCC'], 
        'test_F1': test_metrics['F1'], 'test_AUC': test_metrics['AUC'],
        'test_FPR': test_metrics['FPR'], 'test_TPR': test_metrics['TPR']
    }

# =============================================================================
# 8. Main Execution
# =============================================================================

if __name__ == "__main__":
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    assert "label" in cv_df.columns and "Sequence" in cv_df.columns, "Training Excel missing key columns"
    assert "label" in test_df.columns and "Sequence" in test_df.columns, "Test Excel missing key columns"

    print(f"CV Dataset Size: {len(cv_df)} | Pos: {sum(cv_df['label'])} | Neg: {len(cv_df)-sum(cv_df['label'])}")
    print(f"Test Dataset Size: {len(test_df)} | Pos: {sum(test_df['label'])} | Neg: {len(test_df)-sum(test_df['label'])}")

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].tolist()

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Auto-detected Max Sequence Length: {MAX_SEQ_LENGTH}")

    # --- Step 2: Feature Extraction (Raw) ---
    print("Starting Feature Extraction (Using RESM-150M-KDNY)...")
    X_cv_raw = extract_RESM_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values

    print("Extracting Test Set Features...")
    X_test_raw = extract_RESM_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values
    
    # !!! CRITICAL: Free up Feature Extractor Model Memory !!!
    print("Deleting RESM model to free GPU memory for training...")
    del resm_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # --- Step 3: Cross Validation with Strict PCA ---
    config = CurrentModelConfig(
        max_time_steps=MAX_SEQ_LENGTH, input_size=RESM_HIDDEN_DIM, num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE, save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_fold_results = []
    
    # Original shape info
    n_cv, seq_len, feat_dim = X_cv_raw.shape
    n_test = X_test_raw.shape[0]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        # 1. Split Raw Data
        X_train_raw = X_cv_raw[train_idx]
        X_val_raw = X_cv_raw[val_idx]
        
        # 2. Flatten for PCA: (N * L, 640)
        X_train_flat = X_train_raw.reshape(-1, feat_dim)
        X_val_flat = X_val_raw.reshape(-1, feat_dim)
        X_test_flat = X_test_raw.reshape(-1, feat_dim)
        
        # 3. PCA Fit on Train ONLY
        pca = PCA(n_components=RESM_HIDDEN_DIM, random_state=42)
        X_train_red = pca.fit_transform(X_train_flat)
        X_val_red = pca.transform(X_val_flat)
        X_test_red = pca.transform(X_test_flat) # Transform test using this fold's scaler
        
        # 4. Reshape Back: (N, L, 50)
        X_train_fold = X_train_red.reshape(len(train_idx), seq_len, RESM_HIDDEN_DIM)
        X_val_fold = X_val_red.reshape(len(val_idx), seq_len, RESM_HIDDEN_DIM)
        X_test_fold = X_test_red.reshape(n_test, seq_len, RESM_HIDDEN_DIM)
        
        # 5. Prepare Lists
        train_labels_fold = y_cv[train_idx]
        val_labels_fold = y_cv[val_idx]
        train_seqs_fold = [cv_sequences[i] for i in train_idx]
        val_seqs_fold = [cv_sequences[i] for i in val_idx]
        
        # 6. Run Training
        fold_result = kfold_train(
            fold, X_train_fold, train_labels_fold, X_val_fold, val_labels_fold, 
            X_test_fold, y_test, test_sequences,
            train_seqs_fold, val_seqs_fold, config, EPOCHS
        )
        all_fold_results.append(fold_result)
        
        # 7. cleanup fold memory
        del X_train_raw, X_val_raw, X_train_flat, X_val_flat, X_test_flat
        del X_train_red, X_val_red, X_test_red
        del X_train_fold, X_val_fold, X_test_fold
        del pca
        gc.collect()
    
    # --- Step 4: Summary ---
    def get_avg_std(key):
        vals = [res[key] for res in all_fold_results]
        return np.mean(vals), np.std(vals)

    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred | Feat: RESM + Local PCA)")
    print(f"Max Seq Len: {MAX_SEQ_LENGTH} | LSTM Layers: {LSTM_LAYERS} | CBAM Layers: {CBAM_LAYERS}")
    
    print("\n[Validation Metrics]")
    print(f"Sn : {get_avg_std('best_val_Sn')[0]:.3f} ± {get_avg_std('best_val_Sn')[1]:.3f}")
    print(f"Sp : {get_avg_std('best_val_Sp')[0]:.3f} ± {get_avg_std('best_val_Sp')[1]:.3f}")
    print(f"ACC: {get_avg_std('best_val_ACC')[0]:.3f} ± {get_avg_std('best_val_ACC')[1]:.3f}")
    print(f"MCC: {get_avg_std('best_val_MCC')[0]:.3f} ± {get_avg_std('best_val_MCC')[1]:.3f}")
    print(f"F1 : {get_avg_std('best_val_F1')[0]:.3f} ± {get_avg_std('best_val_F1')[1]:.3f}")
    print(f"AUC: {get_avg_std('best_val_AUC')[0]:.3f} ± {get_avg_std('best_val_AUC')[1]:.3f}")
    
    print("\n[Test Metrics]")
    print(f"Sn : {get_avg_std('test_Sn')[0]:.3f} ± {get_avg_std('test_Sn')[1]:.3f}")
    print(f"Sp : {get_avg_std('test_Sp')[0]:.3f} ± {get_avg_std('test_Sp')[1]:.3f}")
    print(f"ACC: {get_avg_std('test_ACC')[0]:.3f} ± {get_avg_std('test_ACC')[1]:.3f}")
    print(f"MCC: {get_avg_std('test_MCC')[0]:.3f} ± {get_avg_std('test_MCC')[1]:.3f}")
    print(f"F1 : {get_avg_std('test_F1')[0]:.3f} ± {get_avg_std('test_F1')[1]:.3f}")
    print(f"AUC: {get_avg_std('test_AUC')[0]:.3f} ± {get_avg_std('test_AUC')[1]:.3f}")
    print("=" * 80)

    result_df = pd.DataFrame(all_fold_results)
    result_save_path = "Model_Performance_Deep_dsRNAPred.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nPerformance metrics saved to: {result_save_path}")