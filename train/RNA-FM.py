import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc   # 核心：用于显式内存回收
import os
import random
import pickle  # 用于保存和加载 PCA 模型
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from multimolecule import RnaTokenizer, RnaFmModel
from sklearn.model_selection import KFold
# Metrics
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
# PCA
from sklearn.decomposition import PCA

# =============================================================================
# 0. 设置全局随机种子
# =============================================================================
def set_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(3407)

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", category=UserWarning, module="multimolecule")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
RNA_FM_PATH = r'/root/autodl-tmp/Big_Model/RNA-FM/ZhejiangLab-LifeScience/rnafm'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

# 保存配置
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'  
OUTPUT_SAVE_PATH = 'Ensemble_Test_Predictions.xlsx'
RESULT_SAVE_PATH = "Deep_dsRNAPred_Performance.xlsx"

# 维度与超参数配置
RNA_FM_HIDDEN_DIM = 50  # PCA目标维度
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5

# --- 【修改】优化器与学习率调度超参数 ---
LEARNING_RATE = 1e-4      # Adam 学习率
WEIGHT_DECAY = 1e-5       # Adam 权重衰减
STEP_SIZE = 20            # StepLR 每训练 20 个 epoch 调整一次
GAMMA = 0.5               # StepLR 每次调整将学习率减半
# ---------------------------------------------

# 模型结构参数
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# 全局加载特征提取模型 (仅用于提取 Raw Feature)
print(f"Loading RNA-FM model from {RNA_FM_PATH} ...")
tokenizer = RnaTokenizer.from_pretrained(RNA_FM_PATH)
rna_fm_model = RnaFmModel.from_pretrained(RNA_FM_PATH).to(device)
rna_fm_model.eval()
print("Model loaded successfully.")

# =============================================================================
# 2. 辅助配置类
# =============================================================================

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='model_'):
        self.device = device
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix

# =============================================================================
# 3. 注意力机制模块与核心网络 (Attention Modules & Core Net)
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
# 4. 数据处理与工具函数
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

def extract_RNA_FM_features(sequences, max_seq_length):
    """提取 RNA-FM 原始特征"""
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
            
            del inputs, outputs, batch_hidden
            torch.cuda.empty_cache()
    
    all_hidden = torch.cat(hidden_states_list, dim=0)
    return all_hidden.numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0) # TPR
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # 兼容 y_prob: 如果是 [N, 2] 则取第 1 列
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1, 'FPR': fpr, 'TPR': sn}


# =============================================================================
# 5. 训练与验证流程 (仅保留交叉验证部分)
# =============================================================================

def train_one_epoch(dataloader, model, loss_fn, optimizer):
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

def validate_one_epoch(dataloader, model):
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

def run_training_cv(cv_df, MAX_SEQ_LENGTH):
    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].values

    print("Extracting Raw Features (CV)...")
    X_cv_raw = extract_RNA_FM_features(cv_sequences, MAX_SEQ_LENGTH)
    gc.collect()

    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=RNA_FM_HIDDEN_DIM, 
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nProcessing Fold {fold + 1} PCA...")
        
        train_features_raw = X_cv_raw[train_idx]
        train_labels = cv_labels[train_idx]
        train_sequences_fold = [cv_sequences[i] for i in train_idx]
        
        val_features_raw = X_cv_raw[val_idx]
        val_labels = cv_labels[val_idx]
        val_sequences_fold = [cv_sequences[i] for i in val_idx]
        
        # PCA拟合与保存 (仅在 Train 上 Fit)
        N_train, L_train, C_train = train_features_raw.shape
        train_flat = train_features_raw.reshape(-1, C_train)
        
        pca = PCA(n_components=RNA_FM_HIDDEN_DIM, random_state=42)
        pca.fit(train_flat) 
        
        # --- 保存当前折的 PCA 模型 ---
        pca_save_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_save_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"Saved PCA model to {pca_save_path}")
        
        # Transform Train
        train_red = pca.transform(train_flat)
        train_features = train_red.reshape(N_train, L_train, RNA_FM_HIDDEN_DIM).astype(np.float32)
        
        # Transform Val
        N_val, L_val, C_val = val_features_raw.shape
        val_flat = val_features_raw.reshape(-1, C_val)
        val_red = pca.transform(val_flat)
        val_features = val_red.reshape(N_val, L_val, RNA_FM_HIDDEN_DIM).astype(np.float32)

        del train_flat, train_red, val_flat, val_red
        del train_features_raw, val_features_raw
        gc.collect()
        
        train_loader = build_dataloader(train_sequences_fold, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_sequences_fold, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=config.num_classes, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        loss_fn = nn.CrossEntropyLoss().to(device)
        
        # --- 【修改】使用统一标准 ---
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        # ----------------------------
        
        best_val_acc = 0.0
        best_val_metrics = None
        print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM: {CBAM_LAYERS}) =====")
        
        for epoch in range(EPOCHS):
            train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer)
            val_metrics = validate_one_epoch(val_loader, model)
            scheduler.step()
            
            if val_metrics['ACC'] > best_val_acc:
                best_val_acc = val_metrics['ACC']
                best_val_metrics = val_metrics
                torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
            
            print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
                  f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | Val F1: {val_metrics['F1']:.3f}")
        
        # 记录内部纯验证集结果
        all_fold_results.append({
            'fold': fold + 1,
            'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
            'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
            'best_val_AUC': best_val_metrics['AUC'], 'best_val_F1': best_val_metrics['F1']
        })

        del train_features, val_features, train_loader, val_loader, pca, model
        torch.cuda.empty_cache()
        gc.collect()

    # 汇总并保存 CV 结果
    result_df = pd.DataFrame(all_fold_results)
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\nTraining complete. CV Performance metrics saved to: {RESULT_SAVE_PATH}")

# =============================================================================
# 6. 推理流程 (加载多模型集成与软投票机制)
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

def run_ensemble_inference(test_df, MAX_SEQ_LENGTH):
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    print("\nExtracting Raw RNA-FM Features (Inference)...")
    X_test_raw = extract_RNA_FM_features(test_sequences, MAX_SEQ_LENGTH)
    gc.collect()

    N_test, L_test, C_test = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, C_test)
    
    # 储存每折产生的二维概率矩阵
    ensemble_probs = np.zeros((KFOLD, len(test_sequences), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"\n--- Inference: Processing Fold {fold + 1}/{KFOLD} ---")
        
        # 1. 加载 PCA 并降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(N_test, L_test, RNA_FM_HIDDEN_DIM).astype(np.float32)
        
        # 2. 构建无 Label 的 Dataloader
        test_dataset = RNADataset(test_sequences, features=test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 3. 初始化并加载最佳 Model
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=RNA_FM_HIDDEN_DIM,
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
# 7. 主程序入口 (执行控制)
# =============================================================================
if __name__ == "__main__":
    
    # --- 控制开关 ---
    DO_TRAIN = True         # 执行模型训练和 PCA 保存
    DO_INFERENCE = True     # 执行多模型加载、集成打分与预测
    
    # 统一读取数据以确认全局最大长度
    cv_df_global = pd.read_excel(TRAIN_EXCEL_PATH)
    test_df_global = pd.read_excel(TEST_EXCEL_PATH)
    all_seqs = cv_df_global["Sequence"].tolist() + test_df_global["Sequence"].tolist()
    GLOBAL_MAX_SEQ_LENGTH = max([len(seq) for seq in all_seqs])
    print(f"Global Max Sequence Length: {GLOBAL_MAX_SEQ_LENGTH}")
    
    if DO_TRAIN:
        print("\n" + "#" * 50)
        print(">>> STARTING TRAINING PIPELINE <<<")
        print("#" * 50)
        run_training_cv(cv_df_global, GLOBAL_MAX_SEQ_LENGTH)
        
    if DO_INFERENCE:
        print("\n" + "#" * 50)
        print(">>> STARTING ENSEMBLE INFERENCE PIPELINE <<<")
        print("#" * 50)
        run_ensemble_inference(test_df_global, GLOBAL_MAX_SEQ_LENGTH)