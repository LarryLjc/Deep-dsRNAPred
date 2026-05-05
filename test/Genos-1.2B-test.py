import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import gc
import joblib
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Transformers & Genos_1.2B
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

warnings.filterwarnings("ignore", message="Unable to import Triton")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for testing: {device}")

# --- 路径配置 (请根据实际情况修改) ---
GENOS_PATH = r'/root/autodl-tmp/Big_Model/Genos/BGI-HangzhouAI/Genos-1.2B'
TEST_CSV_PATH = "/root/autodl-tmp/data/test_DNA.csv"
OUTPUT_SAVE_PATH = "Final_Ensemble_Predictions_DNA.csv"

# --- 模型权重前缀 ---
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = 'pca_fold_'

# --- 超参数 (必须与训练时完全一致) ---
PCA_OUTPUT_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2 
CBAM_LAYERS = 2

# 重要提示：这里的 MAX_SEQ_LENGTH 在理想情况下应该与训练时的完全一致。
# 如果您的测试集最长序列比训练集长，请手动写死训练时的 MAX_SEQ_LENGTH。
# 这里我们默认自动计算测试集的长度。
MANUAL_MAX_SEQ_LENGTH = None 

# =============================================================================
# 2. 模型结构定义 (保持与训练代码绝对一致)
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
        self.cnn_blocks = nn.ModuleList()
        in_planes = input_size
        for i in range(cnn_layers):
            kernels = [3] if i == 0 else [7, 9]
            self.cnn_blocks.append(CNNBlock(in_planes=in_planes, out_planes=cnn_dims, kernels=kernels, pool_size=pool_size))
            in_planes = cnn_dims

        after_cnn_length = max_seq_length
        for _ in range(cnn_layers):
            after_cnn_length = after_cnn_length // pool_size
            
        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])
        
        self.bilstm = nn.LSTM(
            input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        
        self.fc_blocks = nn.ModuleList()
        in_features = lstm_hidden_size * 2 * after_cnn_length
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
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)

# =============================================================================
# 3. 数据处理与特征提取工具
# =============================================================================

class RNADataset(Dataset):
    def __init__(self, sequences, features=None):
        self.sequences = sequences
        self.features = features

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.features is not None:
            return torch.tensor(self.features[idx], dtype=torch.float32)
        return self.sequences[idx]

def extract_Genos_features(sequences, max_seq_length, tokenizer, genos_model):
    num_samples = len(sequences)
    hidden_size = genos_model.config.hidden_size 
    
    print(f"Allocating memory for features: {num_samples} samples, shape=({num_samples}, {max_seq_length}, {hidden_size})")
    all_hidden = np.zeros((num_samples, max_seq_length, hidden_size), dtype=np.float16)
    
    genos_model.eval()
    for i in range(0, num_samples, BATCH_SIZE):
        batch_seq = sequences[i : i + BATCH_SIZE]
        
        inputs = tokenizer(
            batch_seq, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_seq_length
        ).to(device)
        
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        with torch.no_grad():
            outputs = genos_model(**inputs)
            batch_hidden = outputs[0] 
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, max_seq_length - batch_hidden.size(1)))
            
            current_batch_len = len(batch_seq)
            all_hidden[i : i + current_batch_len] = batch_hidden.cpu().numpy().astype(np.float16)
            
        del inputs, outputs, batch_hidden
        torch.cuda.empty_cache()
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"Processed batch {i // BATCH_SIZE + 1}/{(num_samples + BATCH_SIZE - 1) // BATCH_SIZE}", end='\r')

    print("\nFeature extraction complete.")
    return all_hidden

def transform_pca_data(X_data, ipca, n_components=50, pca_chunk_size=100):
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

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1}

# =============================================================================
# 4. 主推理函数
# =============================================================================

def test_ensemble():
    print("\n" + "=" * 60)
    print(">>> STARTING ENSEMBLE TEST & INFERENCE PIPELINE <<<")
    print("=" * 60)
    
    # 1. 加载测试集数据
    print(f"Reading testing data from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    assert "sequence" in test_df.columns, "CSV file must contain a 'sequence' column."
    
    test_sequences = test_df["sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values
        print(f"Test Data Size: {len(test_df)} | Pos: {sum(test_labels)} | Neg: {len(test_labels)-sum(test_labels)}")
    else:
        print(f"Test Data Size: {len(test_df)} | No ground truth labels found.")

    max_seq_length = MANUAL_MAX_SEQ_LENGTH if MANUAL_MAX_SEQ_LENGTH else max([len(seq) for seq in test_sequences])
    print(f"Using Max Sequence Length: {max_seq_length}")

    # 2. 提取 Genos 大模型特征 (提完即删)
    print("\n[Phase 1] Extracting Genos-1.2B Features...")
    tokenizer = AutoTokenizer.from_pretrained(GENOS_PATH, trust_remote_code=True)
    config = AutoConfig.from_pretrained(f"{GENOS_PATH}/config.json")
    genos_model = AutoModel.from_pretrained(GENOS_PATH, trust_remote_code=True, config=config).to(device)
    
    X_test_raw = extract_Genos_features(test_sequences, max_seq_length, tokenizer, genos_model)
    
    print("Deleting Genos model to free GPU memory...")
    del genos_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 3. 加载 5 折的 PCA 模型和深度学习权重进行推理
    print("\n[Phase 2] Ensemble Inference with 5 Folds...")
    ensemble_probs = np.zeros((KFOLD, len(test_sequences), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"-> Processing Fold {fold + 1}/{KFOLD}")
        
        # 加载对应的 PCA 模型降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found at: {pca_path}. Please train first.")
        ipca = joblib.load(pca_path)
            
        X_test_pca = transform_pca_data(X_test_raw, ipca, n_components=PCA_OUTPUT_DIM)
        
        # 准备 Dataloader
        test_dataset = RNADataset(test_sequences, features=X_test_pca)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 准备 模型架构
        model = Deep_dsRNAPred(
            max_seq_length=max_seq_length, input_size=PCA_OUTPUT_DIM, cnn_layers=3, cnn_dims=512, 
            pool_size=2, num_layers=3, num_dims=64, dropout_rate=DROPOUT_RATE, num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        # 加载对应的权重
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best_Nye_aug.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}. Please train first.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 预测打分
        fold_probs = []
        with torch.no_grad():
            for features in test_loader:
                probs = F.softmax(model(features.to(device)), dim=1)
                fold_probs.extend(probs.cpu().numpy())
                
        ensemble_probs[fold] = np.array(fold_probs)
        
        # 内存回收
        del ipca, X_test_pca, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 4. 集成打分与预测保存
    print("\n[Phase 3] Calculating Final Ensemble Results...")
    final_avg_probs = np.mean(ensemble_probs, axis=0) # [N, 2]
    final_preds = np.argmax(final_avg_probs, axis=1)

    test_df['ensemble_prob_class_1'] = final_avg_probs[:, 1]
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
        
    test_df.to_csv(OUTPUT_SAVE_PATH, index=False)
    print(f"==> Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # 5. 输出性能指标（如果存在真实标签）
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n" + "="*45)
        print("【 Test Set Final Metrics (Ensemble) 】")
        print(f" ACC: {metrics['ACC']:.4f}")
        print(f" AUC: {metrics['AUC']:.4f}")
        print(f" MCC: {metrics['MCC']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  Sn: {metrics['Sn']:.4f} (Sensitivity/Recall)")
        print(f"  Sp: {metrics['Sp']:.4f} (Specificity)")
        print("="*45)

if __name__ == "__main__":
    test_ensemble()