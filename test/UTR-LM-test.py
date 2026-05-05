import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import pickle
import gc
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# --- 引入 Tokenizer 和原生的 AutoModel ---
from multimolecule import RnaTokenizer
from transformers import AutoModel
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------- ⚠️ 关键配置 ⚠️ -----------------
# 必须填入你在训练时控制台打印出的 Max Seq Len！
# 例如：如果训练时打印 "Max Sequence Length (including special tokens): 616"，这里就填 616。
TRAIN_MAX_SEQ_LENGTH = 616  # <--- 修改为你的实际数值！！！
# ---------------------------------------------------

UTR_LM_PATH = r'/root/autodl-tmp/Big_Model/utrlm'
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx" # 你的测试集
SAVE_MODEL_PREFIX = 'stage1model_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'
RESULT_SAVE_PATH = "Ensemble_Test_Predictions_UtrLm.xlsx"

UTR_LM_HIDDEN_DIM = 50
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64

# 模型结构参数 (必须与训练时绝对一致)
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# =============================================================================
# 2. Attention Modules & Core Model (完全复用)
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

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)])

        self.bilstm = nn.LSTM(
            input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0
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
        for block in self.cnn_blocks: x = block(x)
        for cbam_block in self.cbam_blocks: x = cbam_block(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        x = lstm_out.flatten(start_dim=1)
        for block in self.fc_blocks: x = block(x)
        x = self.mid_dropout(F.relu(self.mid_fc(x)))
        return self.output_layer(x)

# =============================================================================
# 3. 特征提取与加载工具
# =============================================================================

def fix_and_load_weights(model, model_path):
    bin_path = os.path.join(model_path, "pytorch_model.bin")
    safe_path = os.path.join(model_path, "model.safetensors")
    weight_file = bin_path if os.path.exists(bin_path) else safe_path if os.path.exists(safe_path) else None

    if weight_file is None: return
    file_size_mb = os.path.getsize(weight_file) / (1024 * 1024)
    if file_size_mb < 1.0: return # Skip git lfs pointer checks for inference silence

    if weight_file.endswith(".bin"):
        state_dict = torch.load(weight_file, map_location="cpu")
    else:
        from safetensors.torch import load_file
        state_dict = load_file(weight_file)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ["utrlm_model.", "utrlm.", "roberta.", "bert.", "model.", "base_model."]:
            if new_k.startswith(prefix): new_k = new_k[len(prefix):]
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict, strict=False)

def resize_pos_embeddings(model, new_max_length):
    config = model.config
    current_max_pos = config.max_position_embeddings
    if new_max_length > current_max_pos:
        base_model = getattr(model, model.base_model_prefix, model)
        old_embeddings = base_model.embeddings.position_embeddings.weight.data
        old_embeddings_t = old_embeddings.t().unsqueeze(0)
        new_embeddings_t = F.interpolate(old_embeddings_t, size=new_max_length, mode='linear', align_corners=True)
        new_pos_layer = nn.Embedding(new_max_length, config.hidden_size)
        new_pos_layer.weight.data = new_embeddings_t.squeeze(0).t()
        new_pos_layer.to(model.device)
        base_model.embeddings.position_embeddings = new_pos_layer
        new_pos_ids = torch.arange(new_max_length).expand((1, -1)).to(model.device)
        base_model.embeddings.register_buffer("position_ids", new_pos_ids)
        model.config.max_position_embeddings = new_max_length

def extract_UTR_LM_features(sequences, max_seq_length, model, tokenizer):
    resize_pos_embeddings(model, max_seq_length)
    hidden_states_list = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(batch_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length).to(device)
            outputs = model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            hidden_states_list.append(batch_hidden.cpu())
            print(f"Extraction progress: {min(i + BATCH_SIZE, len(sequences))}/{len(sequences)}", end='\r')
            
    print("\nFeature extraction complete.")
    return torch.cat(hidden_states_list, dim=0).numpy()

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
# 4. 主执行程序 (Ensemble Inference)
# =============================================================================

if __name__ == "__main__":
    print(f"Loading test data from {TEST_EXCEL_PATH}...")
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    test_sequences = test_df["Sequence"].tolist()
    
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values
        print(f"Found labels in dataset. Total samples: {len(test_df)} (Pos: {sum(test_labels)}, Neg: {len(test_labels)-sum(test_labels)})")
    else:
        test_labels = np.zeros(len(test_df))
        print(f"No labels found. Proceeding with inference only. Total samples: {len(test_df)}")

    # 1. 初始化大模型并提取基座特征
    print(f"Loading UtrLm from {UTR_LM_PATH}...")
    tokenizer = RnaTokenizer.from_pretrained(UTR_LM_PATH)
    utr_lm_model = AutoModel.from_pretrained(UTR_LM_PATH, trust_remote_code=True).to(device)
    fix_and_load_weights(utr_lm_model, UTR_LM_PATH)
    utr_lm_model.eval()

    print("Extracting Raw UtrLm Features...")
    X_test_raw = extract_UTR_LM_features(test_sequences, TRAIN_MAX_SEQ_LENGTH, utr_lm_model, tokenizer)
    
    n_test, seq_len, feat_dim = X_test_raw.shape
    
    # 清理大模型显存
    del utr_lm_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 2. 5折模型集成预测 (Soft Voting)
    print("\nStarting 5-Fold Ensemble Inference...")
    all_fold_probs = np.zeros((len(test_df), NUM_CLASSES))

    for fold in range(1, KFOLD + 1):
        print(f"\n--- Processing Fold {fold} ---")
        pca_path = f"{PCA_SAVE_PREFIX}{fold}.pkl"
        model_path = f"{SAVE_MODEL_PREFIX}{fold}_best.pth"

        if not os.path.exists(pca_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing PCA or Model file for fold {fold}.")

        # ------------------ 核心 PCA 降维 ------------------
        print(f"Applying PCA for Fold {fold}...")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        # 展平为二维进行 PCA Transform
        X_test_flat = X_test_raw.reshape(-1, feat_dim)
        X_test_red = pca.transform(X_test_flat)
        # 再恢复回三维形状给深度学习模型 (N, L, C)
        X_test_pca = X_test_red.reshape(n_test, seq_len, UTR_LM_HIDDEN_DIM)
        # ----------------------------------------------------
        
        # 加载折对应的模型
        model = Deep_dsRNAPred(
            max_seq_length=TRAIN_MAX_SEQ_LENGTH, input_size=UTR_LM_HIDDEN_DIM, 
            cnn_dims=512, cbam_layers=CBAM_LAYERS
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # 推理
        fold_probs = []
        test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_pca, dtype=torch.float32))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.no_grad():
            for batch in test_loader:
                features = batch[0].to(device)
                outputs = model(features)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                fold_probs.extend(probs)
        
        fold_probs = np.array(fold_probs)
        all_fold_probs += fold_probs # 累加概率

        del model, X_test_pca, test_loader, pca
        torch.cuda.empty_cache()
        gc.collect()

    # 3. 计算最终结果
    print("\n" + "="*50)
    print("Ensemble Inference Complete!")
    avg_probs = all_fold_probs / KFOLD
    final_preds = np.argmax(avg_probs, axis=1)

    # 4. 评估 (如果有标签)
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, avg_probs)
        print("【Ensemble Test Metrics】")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("="*50)

    # 5. 保存预测结果
    test_df['Prob_Class_0'] = avg_probs[:, 0]
    test_df['Prob_Class_1'] = avg_probs[:, 1]
    test_df['Final_Prediction'] = final_preds
    
    test_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\n✅ Predictions saved successfully to: {RESULT_SAVE_PATH}")