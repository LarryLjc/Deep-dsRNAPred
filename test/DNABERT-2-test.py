import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import gc
import joblib
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertConfig
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", message=".*Increasing alibi size.*")

###############################################################################
# 1. 全局配置与环境初始化
###############################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 路径配置 (请根据实际情况修改) ---
DNABERT2_PATH = r'/root/autodl-tmp/Big_Model/DNABERT2/ZhejiangLab-LifeScience/DNABERT-2-117M'
TEST_CSV_PATH = "/root/autodl-tmp/data/test_DNA.csv"  # 待测试的数据文件
OUTPUT_CSV_PATH = "/root/autodl-tmp/Model/DNA_Model/DNABert-2/DNABERT-2-test.xlsx" # 预测结果保存路径

SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
SAVE_PCA_PREFIX = 'Deep_dsRNAPred_pca_fold_'

# --- 模型超参数 ---
PCA_OUTPUT_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2 
CBAM_LAYERS = 2

###############################################################################
# 2. 注意力机制与模型架构 (必须保留以加载权重)
###############################################################################

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
        conv_outs = [conv(x) for conv in self.convs]
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
    def __init__(self, max_seq_length, input_size=50, cnn_layers=3, cnn_dims=512, 
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
            self.after_cnn_length = self.after_cnn_length // pool_size

        self.cbam_blocks = nn.ModuleList([CBAMBlock(channel=cnn_dims) for _ in range(cbam_layers)]) 

        self.bilstm = nn.LSTM(
            input_size=cnn_dims, hidden_size=lstm_hidden_size, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0
        )
        self.lstm_flatten_dim = lstm_hidden_size * 2 * self.after_cnn_length
        
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

###############################################################################
# 3. 数据处理与工具函数
###############################################################################

class RNATestDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32)

def extract_DNABERT2_features(sequences, max_seq_length, model, tokenizer):
    num_samples = len(sequences)
    hidden_size = model.config.hidden_size 
    print(f"Allocating memory for features: {num_samples} samples, shape=({num_samples}, {max_seq_length}, {hidden_size})")
    all_hidden = np.zeros((num_samples, max_seq_length, hidden_size), dtype=np.float16)
    
    for i in range(0, num_samples, BATCH_SIZE):
        batch_seq = sequences[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
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

def transform_pca_inference(X_data, ipca_model, n_components=50):
    N, L, C = X_data.shape
    pca_chunk_size = 100
    X_reduced_list = []
    for i in range(0, N, pca_chunk_size):
        chunk = X_data[i : i + pca_chunk_size]
        chunk_flat = chunk.reshape(-1, C)
        chunk_reduced = ipca_model.transform(chunk_flat)
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

###############################################################################
# 4. 主执行流程
###############################################################################

if __name__ == "__main__":
    # 1. 加载数据
    print(f"Reading testing data from: {TEST_CSV_PATH}")
    test_df = pd.read_csv(TEST_CSV_PATH)
    assert "sequence" in test_df.columns, "CSV file must contain a 'sequence' column."
    test_sequences = test_df["sequence"].tolist()
    
    # 动态获取 max_seq_length，或者手动指定如果你在训练时固定了长度
    # 注意：如果训练时的 max_length 是固定的全局最大值，推理时最好保持一致以确保维度对齐
    MAX_SEQ_LENGTH = max([len(seq) for seq in test_sequences])
    print(f"Max Sequence Length for inference: {MAX_SEQ_LENGTH}")

    # 2. 加载 DNABERT-2
    print(f"Loading DNABERT-2 from {DNABERT2_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_PATH, trust_remote_code=True)
    config = BertConfig.from_pretrained(f"{DNABERT2_PATH}/config.json")
    dnabert2_model = AutoModel.from_pretrained(
        DNABERT2_PATH, trust_remote_code=True, config=config, add_pooling_layer=False
    ).to(device)
    dnabert2_model.eval()

    # 3. 提取特征
    X_test_raw = extract_DNABERT2_features(test_sequences, MAX_SEQ_LENGTH, dnabert2_model, tokenizer)
    del dnabert2_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 5折交叉集成推理
    print("\nStarting Ensemble Inference (Soft Voting over 5 Folds)...")
    ensemble_probs = np.zeros((len(test_sequences), NUM_CLASSES), dtype=np.float32)

    for fold in range(KFOLD):
        print(f"-> Processing Fold {fold + 1}/{KFOLD}")
        
        # 加载 PCA 模型并转换数据
        pca_path = f"{SAVE_PCA_PREFIX}{fold + 1}.pkl"
        ipca_model = joblib.load(pca_path)
        X_test_pca = transform_pca_inference(X_test_raw, ipca_model, n_components=PCA_OUTPUT_DIM)
        
        # 准备 Dataloader
        test_dataset = RNATestDataset(X_test_pca)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 加载 PyTorch 模型
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=PCA_OUTPUT_DIM, cnn_layers=3, 
            cnn_dims=512, pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, 
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS
        ).to(device)
        model.load_state_dict(torch.load(f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth", map_location=device))
        model.eval()
        
        # 收集概率
        fold_probs = []
        with torch.no_grad():
            for features in test_loader:
                features = features.to(device)
                preds = model(features)
                probs = F.softmax(preds, dim=1).cpu().numpy()
                fold_probs.extend(probs)
                
        ensemble_probs += np.array(fold_probs)
        
        # 清理内存
        del model, ipca_model, X_test_pca
        gc.collect()
        torch.cuda.empty_cache()

    # 5. 输出最终结果
    ensemble_probs /= KFOLD
    final_preds = np.argmax(ensemble_probs, axis=1)
    
    # 构造输出的 DataFrame
    output_df = test_df.copy()
    output_df['Prob_Class_0'] = ensemble_probs[:, 0]
    output_df['Prob_Class_1'] = ensemble_probs[:, 1]
    output_df['Prediction'] = final_preds

    # 6. 如果有真实标签，计算性能指标
    if "label" in test_df.columns:
        test_labels = test_df["label"].values
        metrics = calculate_metrics(test_labels, final_preds, ensemble_probs)
        print("\n" + "=" * 50)
        print("【Inference Evaluation Metrics】")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 50)
    
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nInference complete. Predictions saved to: {OUTPUT_CSV_PATH}")