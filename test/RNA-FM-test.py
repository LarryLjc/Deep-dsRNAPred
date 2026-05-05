import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import pickle
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from multimolecule import RnaTokenizer, RnaFmModel
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", category=UserWarning, module="multimolecule")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 【请修改这里的路径和参数】 ---
# 需要预测的新数据文件
NEW_TEST_DATA_PATH = "/root/autodl-tmp/data/test_RNA.xlsx" 
OUTPUT_SAVE_PATH = "RNA-FM-test.xlsx"

RNA_FM_PATH = r'/root/autodl-tmp/Big_Model/RNA-FM/ZhejiangLab-LifeScience/rnafm'
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'  

# 极其重要：这里必须填写你训练时打印出的 Global Max Sequence Length！
TRAIN_MAX_SEQ_LENGTH = 615  # <--- 请替换为真实值

KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
RNA_FM_HIDDEN_DIM = 50
CBAM_LAYERS = 2
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2

# 加载特征提取模型
print(f"Loading RNA-FM model from {RNA_FM_PATH} ...")
tokenizer = RnaTokenizer.from_pretrained(RNA_FM_PATH)
rna_fm_model = RnaFmModel.from_pretrained(RNA_FM_PATH).to(device)
rna_fm_model.eval()
print("RNA-FM Model loaded successfully.")

# =============================================================================
# 2. 依赖的模型组件 (保持与训练代码完全一致)
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
            else: kernels = [7, 9]
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
# 3. 工具函数与数据处理
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

def extract_RNA_FM_features(sequences, max_seq_length):
    hidden_states_list = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = rna_fm_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    pad_length = max_seq_length - batch_hidden.size(1)
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, pad_length))
            
            hidden_states_list.append(batch_hidden.cpu())
            torch.cuda.empty_cache()
    
    return torch.cat(hidden_states_list, dim=0).numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0) 
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1}

def get_model_predictions(model, dataloader):
    model.eval()
    all_prob = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            preds = model(features)
            probs = F.softmax(preds, dim=1) 
            all_prob.extend(probs.cpu().numpy())
    return np.array(all_prob)

# =============================================================================
# 4. 主干测试逻辑
# =============================================================================
def run_independent_test():
    print(f"Loading data from {NEW_TEST_DATA_PATH}...")
    test_df = pd.read_excel(NEW_TEST_DATA_PATH)
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    print(f"Extracting Raw RNA-FM Features (Padding to {TRAIN_MAX_SEQ_LENGTH})...")
    X_test_raw = extract_RNA_FM_features(test_sequences, TRAIN_MAX_SEQ_LENGTH)
    
    N_test, L_test, C_test = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, C_test)
    
    ensemble_probs = np.zeros((KFOLD, len(test_sequences), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"--- Inference: Processing Fold {fold + 1}/{KFOLD} ---")
        
        # 1. 加载对应折的 PCA 模型
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing PCA model: {pca_path}. Did you complete training?")
            
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(N_test, L_test, RNA_FM_HIDDEN_DIM).astype(np.float32)
        
        # 2. 构建 Dataloader
        test_dataset = RNADataset(test_sequences, features=test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 3. 初始化并加载对应折的 PyTorch 模型
        model = Deep_dsRNAPred(
            max_seq_length=TRAIN_MAX_SEQ_LENGTH, input_size=RNA_FM_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model weights: {model_path}.")
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 4. 获取预测概率
        fold_probs = get_model_predictions(model, test_loader)
        ensemble_probs[fold] = fold_probs

    print("\n" + "=" * 50)
    print("Calculating Ensemble Predictions (Soft Voting)...")
    final_avg_probs = np.mean(ensemble_probs, axis=0) # [N, 2]
    final_preds = np.argmax(final_avg_probs, axis=1)

    # 5. 保存结果
    test_df['ensemble_prob_class_1'] = final_avg_probs[:, 1]
    test_df['final_prediction'] = final_preds
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # 6. 如果输入文件包含真实标签，计算并打印指标
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n【Independent Test Performance】")
        print(f"ACC: {metrics['ACC']:.4f} | MCC: {metrics['MCC']:.4f} | F1: {metrics['F1']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f} | Sn (Recall): {metrics['Sn']:.4f} | Sp: {metrics['Sp']:.4f}")
        print("=" * 50)

if __name__ == "__main__":
    run_independent_test()