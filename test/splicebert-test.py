import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import pickle
import gc
import os
import random
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# --- 引入 SpliceBert 模型 ---
from multimolecule import RnaTokenizer, SpliceBertModel

# --- Metrics ---
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

warnings.filterwarnings("ignore")

def setup_seed(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 2. Path & Hyperparameter Configuration
# =============================================================================

SPLICE_BERT_PATH = r'/root/autodl-tmp/Big_Model/SpliceBert/ZhejiangLab-LifeScience/splicebert'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
TEST_MMAP_PATH = '/root/autodl-tmp/test_features_inference.dat'  # 独立于训练的 mmap 文件
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = './pca_features_pkl/pca_model_fold_'

# 【极其重要】: 请将这里替换为你训练脚本跑出来时打印的 Max Sequence Length！
MAX_SEQ_LENGTH = 615  # <--- 修改这里！

SPLICE_HIDDEN_DIM = 50 
NUM_CLASSES = 2
BATCH_SIZE = 64        # 提取特征时的 Batch Size
INFER_BATCH_SIZE = 256 # 推理时的 Batch Size
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2
THRESHOLD = 0.5

# =============================================================================
# 3. Model Architecture (完全一致)
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
        return self.sigmoid(self.se(self.maxpool(x)) + self.se(self.avgpool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([max_result, avg_result], 1)))

class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        out = x * self.ca(x)
        return x + (out * self.sa(out))

class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 7, 9], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(channel, channel, kernel_size=(k, 1), padding=(k // 2, 0), groups=group)),
                ('bn', nn.BatchNorm2d(channel)),
                ('relu', nn.ReLU())
            ])) for k in kernels
        ])
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
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
        return (attention_weights * feats).sum(0)

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
        return self.pool(self.relu(self.sk_attention(self.conv1x1(x))))

class Deep_dsRNAPred(nn.Module):
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=256, 
                 pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
                 num_classes=2, lstm_hidden_size=128, lstm_layers=2, cbam_layers=2):
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

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])
        self.bilstm = nn.LSTM(input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers, bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)
        self.lstm_output_dim = lstm_hidden_size * 2
        
        self.lstm_flatten_dim = self.lstm_output_dim * self.after_cnn_length
        self.fc_blocks = nn.ModuleList()
        in_features = self.lstm_flatten_dim
        for _ in range(num_layers):
            self.fc_blocks.append(nn.Sequential(nn.Linear(in_features, num_dims), nn.ReLU(), nn.Dropout(p=dropout_rate)))
            in_features = num_dims

        self.mid_fc = nn.Linear(num_dims, 128)
        self.mid_dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        for block in self.cnn_blocks: x = block(x)
        for cbam_block in self.cbam_blocks: x = cbam_block(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        for block in self.fc_blocks: x = block(x)
        return self.output_layer(self.mid_dropout(F.relu(self.mid_fc(x))))


# =============================================================================
# 4. Data Processing & Utilities
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def extract_Splice_features_to_disk(sequences, max_seq_length, mmap_filename):
    print(f"Loading SpliceBert from {SPLICE_BERT_PATH}...")
    tokenizer = RnaTokenizer.from_pretrained(SPLICE_BERT_PATH)
    splice_model = SpliceBertModel.from_pretrained(SPLICE_BERT_PATH).to(device)
    splice_model.eval() 

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
            
    del tokenizer, splice_model
    gc.collect()
    torch.cuda.empty_cache()
    return np.memmap(mmap_filename, dtype='float16', mode='r', shape=shape)

def chunked_pca_transform_for_memmap(pca_model, X_mmap, indices=None, chunk_size=1000):
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

def calculate_metrics(y_true, y_pred, pos_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, pos_prob) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 5. Main Execution
# =============================================================================

if __name__ == "__main__":
    print(f"Loading Test Data from: {TEST_EXCEL_PATH}")
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    y_test = np.array(test_df["label"].tolist())
    N_test = len(y_test)
    
    print(f"Loaded {N_test} test samples. Positives: {sum(y_test)}, Negatives: {N_test - sum(y_test)}")
    print(f"Assumed MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")

    # 1. Extract Raw Features (Mmap)
    X_test_raw = extract_Splice_features_to_disk(test_sequences, MAX_SEQ_LENGTH, TEST_MMAP_PATH)
    
    ensemble_probs_accum = np.zeros((N_test, NUM_CLASSES))

    # 2. Cross-Validation Inference Loop
    for fold in range(1, 6):
        print(f"\n[{fold}/5] Processing Fold {fold}...")
        
        # A. Load IncrementalPCA model for this fold
        pca_path = f"{PCA_SAVE_PREFIX}{fold}.pkl"
        print(f"Loading PCA Model: {pca_path}")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        # Transform Test Features
        test_features = chunked_pca_transform_for_memmap(pca, X_test_raw, None)
        test_loader = DataLoader(RNADataset(test_features, y_test), batch_size=INFER_BATCH_SIZE, shuffle=False)

        # B. Load Deep_dsRNAPred model
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH,
            input_size=SPLICE_HIDDEN_DIM,
            cnn_dims=512, 
            lstm_hidden_size=LSTM_HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            cbam_layers=CBAM_LAYERS
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold}_best.pth"
        print(f"Loading Network Weights: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # C. Inference
        fold_probs = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(device)
                logits = model(features)
                probs = F.softmax(logits, dim=1)
                fold_probs.extend(probs.cpu().numpy())
                
        ensemble_probs_accum += np.array(fold_probs)

        # Cleanup Memory
        del model, test_features, test_loader, pca
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # 6. Final Evaluation
    # =========================================================================
    print("\n" + "=" * 50)
    print("Ensemble Evaluation Finished. Calculating metrics...")
    
    # Average across 5 folds
    final_avg_probs = ensemble_probs_accum / 5.0
    pos_probs = final_avg_probs[:, 1]
    final_preds = (pos_probs >= THRESHOLD).astype(int)

    metrics = calculate_metrics(y_true=y_test, y_pred=final_preds, pos_prob=pos_probs)

    print(f"Ensemble Performance (Model: SpliceBert, Threshold = {THRESHOLD}):")
    print(f"Sn  (Recall): {metrics['Sn']:.4f}")
    print(f"Sp  (Spec)  : {metrics['Sp']:.4f}")
    print(f"ACC (Accur) : {metrics['ACC']:.4f}")
    print(f"MCC         : {metrics['MCC']:.4f}")
    print(f"F1 Score    : {metrics['F1']:.4f}")
    print(f"AUC         : {metrics['AUC']:.4f}")
    print("=" * 50)

    # Save to Excel
    test_df['ensemble_prob_pos'] = pos_probs
    test_df['ensemble_pred_label'] = final_preds
    output_excel = "Test_Ensemble_Predictions_Splice.xlsx"
    test_df.to_excel(output_excel, index=False)
    print(f"\nDetailed predictions saved to '{output_excel}'")