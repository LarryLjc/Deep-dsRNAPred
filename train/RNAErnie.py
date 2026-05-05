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

# Model Imports
from multimolecule import RnaTokenizer, RnaErnieModel
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

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path Configuration
RNA_ERNIE_PATH = r'/root/autodl-tmp/Big_Model/RNAErnie_Original/ZhejiangLab-LifeScience/rnaernie'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_PREFIX = 'pca_model_fold_'
OUTPUT_SAVE_PATH = 'Ensemble_Test_Predictions.xlsx'
RESULT_SAVE_PATH = "Model_Performance_Deep_dsRNAPred.xlsx"

# Hyperparameters
RNA_ERNIE_HIDDEN_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2
THRESHOLD = 0.5  # Ensemble prediction threshold

# Optimizer/Scheduler settings
LEARNING_RATE = 2e-4      
WEIGHT_DECAY = 1e-4       
MIN_LR = 1e-6             

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


def resize_pos_embeddings(model, new_max_length):
    config = model.config
    current_max_pos = config.max_position_embeddings
    if new_max_length > current_max_pos:
        print(f"\n[Warning] Input length {new_max_length} > model limit {current_max_pos}.")
        print(f"Doing position embedding interpolation to support {new_max_length} tokens...")
        
        old_embeddings = model.embeddings.position_embeddings.weight.data
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
        
        model.embeddings.position_embeddings = new_pos_layer
        new_pos_ids = torch.arange(new_max_length).expand((1, -1)).to(model.device)
        model.embeddings.register_buffer("position_ids", new_pos_ids)
        model.config.max_position_embeddings = new_max_length
        print("Position embeddings resized successfully.\n")


def extract_RNA_ERNIE_features(sequences, max_seq_length, model, tokenizer, device):
    resize_pos_embeddings(model, max_seq_length)
    hidden_states_list = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = model(**inputs)
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
    
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc, 'FPR': fpr, 'TPR': sn}

# =============================================================================
# 6. Training Pipeline Functions
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

def run_training_cv(X_cv_raw, y_cv, cv_sequences, X_test_raw, y_test, test_sequences, MAX_SEQ_LENGTH):
    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH, input_size=RNA_ERNIE_HIDDEN_DIM, 
        num_classes=NUM_CLASSES, dropout=DROPOUT_RATE, save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    all_fold_results = []
    n_cv, seq_len, feat_dim = X_cv_raw.shape
    n_test = X_test_raw.shape[0]

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        X_train_raw = X_cv_raw[train_idx]
        X_val_raw = X_cv_raw[val_idx]
        
        X_train_flat = X_train_raw.reshape(-1, feat_dim)
        X_val_flat = X_val_raw.reshape(-1, feat_dim)
        X_test_flat = X_test_raw.reshape(-1, feat_dim)
        
        # PCA Fitting and Saving
        pca = PCA(n_components=RNA_ERNIE_HIDDEN_DIM, random_state=SEED)
        X_train_red = pca.fit_transform(X_train_flat)
        
        with open(f'{PCA_PREFIX}{fold+1}.pkl', 'wb') as f:
            pickle.dump(pca, f)
            
        X_val_red = pca.transform(X_val_flat)
        X_test_red = pca.transform(X_test_flat) 
        
        X_train_fold = X_train_red.reshape(len(train_idx), seq_len, RNA_ERNIE_HIDDEN_DIM)
        X_val_fold = X_val_red.reshape(len(val_idx), seq_len, RNA_ERNIE_HIDDEN_DIM)
        X_test_fold = X_test_red.reshape(n_test, seq_len, RNA_ERNIE_HIDDEN_DIM)
        
        train_labels_fold = y_cv[train_idx]
        val_labels_fold = y_cv[val_idx]
        train_seqs_fold = [cv_sequences[i] for i in train_idx]
        val_seqs_fold = [cv_sequences[i] for i in val_idx]
        
        # Model & Optimizer Setup
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps, input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=0.2, num_classes=config.num_classes, cbam_layers=CBAM_LAYERS
        ).to(config.device)
        
        loss_fn = nn.CrossEntropyLoss().to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
        
        train_loader = build_dataloader(train_seqs_fold, train_labels_fold, X_train_fold, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_seqs_fold, val_labels_fold, X_val_fold, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = build_dataloader(test_sequences, y_test, X_test_fold, batch_size=BATCH_SIZE, shuffle=False)
        
        best_val_acc = 0.0
        best_val_metrics = None
        
        print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM Layers: {CBAM_LAYERS}) =====")
        for epoch in range(EPOCHS):
            train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
            val_metrics = validate_one_epoch(val_loader, model, config.device)
            scheduler.step()
            
            if val_metrics['ACC'] > best_val_acc:
                best_val_acc = val_metrics['ACC']
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
            
            print(f"Epoch {epoch + 1:3d} | Train ACC: {train_acc:.1%} | Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | Val F1: {val_metrics['F1']:.3f}")
        
        # Testing
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
        test_metrics = validate_one_epoch(test_loader, model, config.device)
        print(f"Fold {fold + 1} Best Result | Test ACC: {test_metrics['ACC']:.1%} | Test F1: {test_metrics['F1']:.3f}")
        
        all_fold_results.append({
            'fold': fold + 1,
            'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
            'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
            'best_val_F1': best_val_metrics['F1'], 'best_val_AUC': best_val_metrics['AUC'],
            'test_Sn': test_metrics['Sn'], 'test_Sp': test_metrics['Sp'], 
            'test_ACC': test_metrics['ACC'], 'test_MCC': test_metrics['MCC'], 
            'test_F1': test_metrics['F1'], 'test_AUC': test_metrics['AUC'],
            'test_FPR': test_metrics['FPR'], 'test_TPR': test_metrics['TPR']
        })
        
        # Memory Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        del X_train_raw, X_val_raw, X_train_flat, X_val_flat, X_test_flat
        del X_train_red, X_val_red, X_test_red
        del X_train_fold, X_val_fold, X_test_fold
        del pca
        torch.cuda.empty_cache()
        gc.collect()

    result_df = pd.DataFrame(all_fold_results)
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\nTraining Complete! Metrics saved to: {RESULT_SAVE_PATH}")


# =============================================================================
# 7. Inference Pipeline Functions
# =============================================================================

def get_model_predictions(model, dataloader):
    model.eval()
    all_prob = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            preds = model(features)
            probs = F.softmax(preds, dim=1)[:, 1] # Target probability for Class 1
            all_prob.extend(probs.cpu().numpy())
    return np.array(all_prob)

def run_ensemble_inference(test_df, X_test_raw, MAX_SEQ_LENGTH):
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    n_test, seq_len, feat_dim = X_test_raw.shape
    X_test_flat = X_test_raw.reshape(-1, feat_dim)
    ensemble_probs = np.zeros((KFOLD, n_test))

    for fold in range(KFOLD):
        print(f"\n--- Inference: Processing Fold {fold + 1}/{KFOLD} ---")
        
        # 1. Load PCA Model (.pkl)
        pca_path = f"{PCA_PREFIX}{fold + 1}.pkl"
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        # 2. Apply PCA Transformation
        X_test_red = pca.transform(X_test_flat)
        X_test_fold = X_test_red.reshape(n_test, seq_len, RNA_ERNIE_HIDDEN_DIM).astype(np.float32)
        
        # 3. Dataloader Build
        test_dataset = RNADataset(test_sequences, features=X_test_fold)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 4. Load Neural Network (.pth)
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=RNA_ERNIE_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2, num_layers=3, num_dims=64,
            dropout_rate=0.2, num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 5. Extract Probabilities
        fold_probs = get_model_predictions(model, test_loader)
        ensemble_probs[fold] = fold_probs
        
        # Clean Memory
        del pca, X_test_red, X_test_fold, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 6. Compute Ensembles
    print("\n" + "=" * 50)
    print("Calculating Ensemble Predictions...")
    final_avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = (final_avg_probs >= THRESHOLD).astype(int)

    # 7. Save to Dataframe
    test_df['ensemble_prob'] = final_avg_probs
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob'] = ensemble_probs[fold]
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"Ensemble Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # Evaluate if ground truth exists
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n【Ensemble Test Final Metrics】")
        print(f"Threshold: {THRESHOLD} | ACC: {metrics['ACC']:.3f} | MCC: {metrics['MCC']:.3f} | F1: {metrics['F1']:.3f} | AUC: {metrics['AUC']:.3f}")
        print("=" * 50)


# =============================================================================
# 8. Main Execution Controller
# =============================================================================

if __name__ == "__main__":
    
    # ---------------------------------------------
    # Execution Switches
    # ---------------------------------------------
    DO_TRAIN = True         # Run cross validation and train new models
    DO_INFERENCE = True     # Load saved .pkl and .pth models to score test data
    # ---------------------------------------------

    print("Step 1: Reading Excel files...")
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    cv_sequences = cv_df["Sequence"].tolist()
    y_cv = cv_df["label"].values
    test_sequences = test_df["Sequence"].tolist()
    y_test = test_df["label"].values if "label" in test_df.columns else None

    # Calculate global MAX_SEQ_LENGTH
    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Max Sequence Length identified: {MAX_SEQ_LENGTH}")

    # Load Initial Feature Extractor
    print(f"\nStep 2: Loading RNA-Ernie from {RNA_ERNIE_PATH}...")
    tokenizer = RnaTokenizer.from_pretrained(RNA_ERNIE_PATH, model_max_length=1024)
    rna_ernie_model = RnaErnieModel.from_pretrained(RNA_ERNIE_PATH).to(device)
    rna_ernie_model.eval()

    # Feature Extraction
    X_cv_raw = None
    if DO_TRAIN:
        print("Extracting Raw Features (Train)...")
        X_cv_raw = extract_RNA_ERNIE_features(cv_sequences, MAX_SEQ_LENGTH, rna_ernie_model, tokenizer, device)
        
    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_RNA_ERNIE_features(test_sequences, MAX_SEQ_LENGTH, rna_ernie_model, tokenizer, device)

    # CRITICAL: Delete model and clear memory to prevent OOM
    print("\nFeature extraction complete. Deleting RNA-Ernie model to free memory...")
    del rna_ernie_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    if DO_TRAIN:
        print("\n" + "#" * 60)
        print(">>> STARTING CROSS-VALIDATION TRAINING PIPELINE <<<")
        print("#" * 60)
        run_training_cv(X_cv_raw, y_cv, cv_sequences, X_test_raw, y_test, test_sequences, MAX_SEQ_LENGTH)
        
    if DO_INFERENCE:
        print("\n" + "#" * 60)
        print(">>> STARTING ENSEMBLE INFERENCE PIPELINE <<<")
        print("#" * 60)
        run_ensemble_inference(test_df, X_test_raw, MAX_SEQ_LENGTH)