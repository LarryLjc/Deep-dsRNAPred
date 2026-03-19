import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import random
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. GLOBAL CONFIGURATION & SEED SETUP
# =============================================================================

warnings.filterwarnings("ignore")

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
TRAIN_CSV_PATH = "/root/autodl-tmp/data/train_combined.csv"
TEST_CSV_PATH = "/root/autodl-tmp/data/test_combined.csv"
SAVE_MODEL_PREFIX = 'stage1model_fold_'

# Hyperparameters
FEATURE_DIM = 30  # CTD produces 30 features
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 256
EPOCHS = 60
DROPOUT_RATE = 0.2
LSTM_HIDDEN_SIZE = 128 
LSTM_LAYERS = 2 
CBAM_LAYERS = 2 
RANDOM_SEED = 3407

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(RANDOM_SEED)
print(f"Global Seed set to: {RANDOM_SEED}")
print(f"Using device: {device}")

# =============================================================================
# 2. CTD FEATURE EXTRACTION LOGIC
# =============================================================================

class CTDcoder:
    """Generates CTD counts for a sequence"""
    def __init__(self, infasta=None):
        self.infasta = infasta

    def CTD(self, seq):
        n = len(seq)
        n = float(n)
        num_A, num_T, num_G, num_C = 0, 0, 0, 0
        AT_trans, AG_trans, AC_trans, TG_trans, TC_trans, GC_trans = 0, 0, 0, 0, 0, 0
        
        # Calculate Composition and Transition
        for i in range(len(seq) - 1):
            if seq[i] == "A": num_A += 1
            if seq[i] == "T": num_T += 1
            if seq[i] == "G": num_G += 1
            if seq[i] == "C": num_C += 1
            
            if (seq[i] == "A" and seq[i+1] == "T") or (seq[i] == "T" and seq[i+1] == "A"): AT_trans += 1
            if (seq[i] == "A" and seq[i+1] == "G") or (seq[i] == "G" and seq[i+1] == "A"): AG_trans += 1
            if (seq[i] == "A" and seq[i+1] == "C") or (seq[i] == "C" and seq[i+1] == "A"): AC_trans += 1
            if (seq[i] == "T" and seq[i+1] == "G") or (seq[i] == "G" and seq[i+1] == "T"): TG_trans += 1
            if (seq[i] == "T" and seq[i+1] == "C") or (seq[i] == "C" and seq[i+1] == "T"): TC_trans += 1
            if (seq[i] == "G" and seq[i+1] == "C") or (seq[i] == "C" and seq[i+1] == "G"): GC_trans += 1

        if seq[-1] == "A": num_A += 1
        elif seq[-1] == "T": num_T += 1
        elif seq[-1] == "G": num_G += 1
        elif seq[-1] == "C": num_C += 1

        # Calculate Distribution
        a, t, g, c = 0, 0, 0, 0
        A0_dis, A1_dis, A2_dis, A3_dis, A4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        T0_dis, T1_dis, T2_dis, T3_dis, T4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        G0_dis, G1_dis, G2_dis, G3_dis, G4_dis = 0.0, 0.0, 0.0, 0.0, 0.0
        C0_dis, C1_dis, C2_dis, C3_dis, C4_dis = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(len(seq)):
            # A Distribution
            if seq[i] == "A":
                a += 1
                if a == 1: A0_dis = ((i * 1.0) + 1) / n
                if a == int(round(num_A / 4.0)): A1_dis = ((i * 1.0) + 1) / n
                if a == int(round(num_A / 2.0)): A2_dis = ((i * 1.0) + 1) / n
                if a == int(round((num_A * 3 / 4.0))): A3_dis = ((i * 1.0) + 1) / n
                if a == num_A: A4_dis = ((i * 1.0) + 1) / n
            
            # T Distribution
            if seq[i] == "T":
                t += 1
                if t == 1: T0_dis = ((i * 1.0) + 1) / n
                if t == int(round(num_T / 4.0)): T1_dis = ((i * 1.0) + 1) / n
                if t == int(round((num_T / 2.0))): T2_dis = ((i * 1.0) + 1) / n
                if t == int(round((num_T * 3 / 4.0))): T3_dis = ((i * 1.0) + 1) / n
                if t == num_T: T4_dis = ((i * 1.0) + 1) / n

            # G Distribution
            if seq[i] == "G":
                g += 1
                if g == 1: G0_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G / 4.0)): G1_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G / 2.0)): G2_dis = ((i * 1.0) + 1) / n
                if g == int(round(num_G * 3 / 4.0)): G3_dis = ((i * 1.0) + 1) / n
                if g == num_G: G4_dis = ((i * 1.0) + 1) / n

            # C Distribution
            if seq[i] == "C":
                c += 1
                if c == 1: C0_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C / 4.0)): C1_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C / 2.0)): C2_dis = ((i * 1.0) + 1) / n
                if c == int(round(num_C * 3 / 4.0)): C3_dis = ((i * 1.0) + 1) / n
                if c == num_C: C4_dis = ((i * 1.0) + 1) / n
        
        divisor = (n - 1) if n > 1 else 1
        
        return (
            num_A/n, num_T/n, num_G/n, num_C/n,
            AT_trans/divisor, AG_trans/divisor, AC_trans/divisor, TG_trans/divisor, TC_trans/divisor, GC_trans/divisor,
            A0_dis, A1_dis, A2_dis, A3_dis, A4_dis,
            T0_dis, T1_dis, T2_dis, T3_dis, T4_dis,
            G0_dis, G1_dis, G2_dis, G3_dis, G4_dis,
            C0_dis, C1_dis, C2_dis, C3_dis, C4_dis
        )

def extract_CTD_features(sequences, max_seq_length):
    ctd_coder = CTDcoder()
    features_list = []
    print(f"Extracting CTD features (Total: {len(sequences)})...")
    
    for idx, seq in enumerate(sequences):
        # 1. Extract 30 scalar features
        ctd_features = ctd_coder.CTD(seq.upper().replace('U', 'T'))
        
        # 2. Convert to numpy float array
        feat_vec = np.array(ctd_features, dtype=np.float32)
        
        # 3. Broadcast to [Max_Seq_Length, 30] for model input
        seq_feat = np.tile(feat_vec, (max_seq_length, 1))
        features_list.append(seq_feat)
        
    return np.array(features_list)

# =============================================================================
# 3. ATTENTION MODULES
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
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in range(len(kernels))])
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
# 4. MODEL ARCHITECTURE: Deep-dsRNAPred
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
        return self.pool(x)

class Deep_dsRNAPred(nn.Module):
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=256, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=LSTM_HIDDEN_SIZE, lstm_layers=LSTM_LAYERS,
                 cbam_layers=CBAM_LAYERS):
        super(Deep_dsRNAPred, self).__init__()
        
        # 1. CNN Blocks
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        self.after_cnn_length = max_seq_length
        
        for i in range(cnn_layers):
            kernels = [3] if i == 0 else [7, 9]
            self.cnn_blocks.append(CNNBlock(in_planes, cnn_dims, kernels, pool_size))
            in_planes = cnn_dims
            self.after_cnn_length //= pool_size
            
        # 2. CBAM Blocks
        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])

        # 3. BiLSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        # 4. Fully Connected
        lstm_flatten_dim = lstm_hidden_size * 2 * self.after_cnn_length
        self.fc_blocks = nn.ModuleList()
        in_features = lstm_flatten_dim
        
        for _ in range(num_layers):
            self.fc_blocks.append(nn.Sequential(
                nn.Linear(in_features, num_dims),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ))
            in_features = num_dims

        # 5. Output
        self.mid_fc = nn.Linear(num_dims, 128)
        self.mid_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        # [B, L, C] -> [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        for block in self.cnn_blocks:
            x = block(x)
        
        for cbam in self.cbam_blocks:
            x = cbam(x)
        
        # [B, C, L, 1] -> [B, L, C]
        x = x.squeeze(-1).permute(0, 2, 1)
        
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        
        for block in self.fc_blocks:
            x = block(x)
        
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)

# =============================================================================
# 5. UTILITIES & METRICS
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

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='stage1model_fold_'):
        self.device = device
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix

# =============================================================================
# 6. TRAINING & VALIDATION LOOPS
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
        
    return total_acc / total_count

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
            
    metrics = calculate_metrics(np.array(all_true), np.array(all_pred), np.array(all_prob))
    return metrics

def kfold_train(fold, train_features, train_labels, val_features, val_labels, 
                config, epochs, train_loader, val_loader, test_loader):
    
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size, 
        cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
        dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    best_val_metrics = None
    print(f"\n===== Fold {fold + 1}/{KFOLD} Training (Feature: CTD) =====")
    
    for epoch in range(epochs):
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best_CTD.pth")
        
        print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
              f"Sn: {val_metrics['Sn']:.1%} | Sp: {val_metrics['Sp']:.1%} | "
              f"ACC: {val_metrics['ACC']:.1%} | MCC: {val_metrics['MCC']:.3f} | "
              f"F1: {val_metrics['F1']:.3f} | AUC: {val_metrics['AUC']:.3f}")
    
    # Test Evaluation
    model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best_CTD.pth", map_location=config.device))
    test_metrics = validate_one_epoch(test_loader, model, config.device)
    
    print(f"Fold {fold + 1} Best Model | "
          f"Sn: {test_metrics['Sn']:.1%} | Sp: {test_metrics['Sp']:.1%} | "
          f"ACC: {test_metrics['ACC']:.1%} | MCC: {test_metrics['MCC']:.3f} | "
          f"F1: {test_metrics['F1']:.3f} | AUC: {test_metrics['AUC']:.3f}")
    
    return {
        'fold': fold + 1,
        'val_Sn': best_val_metrics['Sn'], 'val_Sp': best_val_metrics['Sp'],
        'val_ACC': best_val_metrics['ACC'], 'val_MCC': best_val_metrics['MCC'],
        'val_F1': best_val_metrics['F1'], 'val_AUC': best_val_metrics['AUC'],
        'test_Sn': test_metrics['Sn'], 'test_Sp': test_metrics['Sp'],
        'test_ACC': test_metrics['ACC'], 'test_MCC': test_metrics['MCC'],
        'test_F1': test_metrics['F1'], 'test_AUC': test_metrics['AUC']
    }

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Load Data
    cv_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    assert "label" in cv_df.columns and "sequence" in cv_df.columns, "Columns missing in Train CSV"
    assert "label" in test_df.columns and "sequence" in test_df.columns, "Columns missing in Test CSV"

    print(f"CV Size: {len(cv_df)} | Test Size: {len(test_df)}")

    cv_sequences = cv_df["sequence"].astype(str).str.lower().tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["sequence"].astype(str).str.lower().tolist()
    test_labels = test_df["label"].tolist()

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")

    # 2. Extract Features (CTD)
    X_cv = extract_CTD_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = np.array(cv_labels)
    X_test = extract_CTD_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = np.array(test_labels)

    print(f"Feature Shape | CV: {X_cv.shape} | Test: {X_test.shape}")
    
    # 3. K-Fold Cross Validation
    test_loader = build_dataloader(test_sequences, test_labels, X_test, batch_size=BATCH_SIZE, shuffle=False)
    config = ModelConfig(max_time_steps=MAX_SEQ_LENGTH, input_size=FEATURE_DIM, num_classes=NUM_CLASSES, 
                         dropout=DROPOUT_RATE, save_model_prefix=SAVE_MODEL_PREFIX)
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv)):
        train_loader = build_dataloader([cv_sequences[i] for i in train_idx], y_cv[train_idx], X_cv[train_idx], 
                                      batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader([cv_sequences[i] for i in val_idx], y_cv[val_idx], X_cv[val_idx], 
                                    batch_size=BATCH_SIZE, shuffle=False)
        
        result = kfold_train(fold, None, None, None, None, config, EPOCHS, train_loader, val_loader, test_loader)
        all_fold_results.append(result)
    
    # 4. Summary & Save
    print("\n" + "=" * 100)
    print(f"5-Fold CV Summary (Model: Deep-dsRNAPred | Feature: CTD | Dim: {FEATURE_DIM})")
    
    def print_stats(prefix):
        metrics = ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']
        res_str = f"【{prefix} Metrics】\n"
        for m in metrics:
            vals = [r[f'{prefix.lower()}_{m}'] for r in all_fold_results]
            res_str += f"{m}: {np.mean(vals):.3f} ± {np.std(vals):.3f} | "
        print(res_str.strip(' | '))

    print_stats("Val")
    print("-" * 100)
    print_stats("Test")
    
    print("=" * 100)

    pd.DataFrame(all_fold_results).to_excel("Model_Performance_CTD.xlsx", index=False)
    print(f"Results saved to Model_Performance_CTD.xlsx")