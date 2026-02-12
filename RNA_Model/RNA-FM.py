import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.nn import init

# 引入 transformers 的日志管理
from transformers import logging as hf_logging
from multimolecule import RnaTokenizer, RnaFmModel
from sklearn.model_selection import KFold
# 导入评估指标 
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
# 导入 PCA
from sklearn.decomposition import PCA

# =============================================================================
# 1. 全局配置与环境初始化 (Global Configuration)
# =============================================================================

# 设置日志级别，屏蔽良性警告
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")
warnings.filterwarnings("ignore", category=UserWarning, module="multimolecule")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 路径配置
RNA_FM_PATH = r'/root/autodl-tmp/Big_Model/RNA-FM/ZhejiangLab-LifeScience/rnafm'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/Run/Run/yuchuli/RNAi_train616_New_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/Run/Run/yuchuli/RNAi_test616_New_RNA.xlsx"
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'

# 维度与超参数配置
RNA_FM_HIDDEN_DIM = 50  # PCA目标维度
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 256
EPOCHS = 40
DROPOUT_RATE = 0.2

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
# 2. 辅助配置类 (Config Class)
# =============================================================================

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='model_'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix

# =============================================================================
# 3. 注意力机制模块 (Attention Modules)
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

# =============================================================================
# 4. 核心网络架构 (Deep_dsRNAPred)
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
        
        # --- CNN Blocks ---
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

        # --- CBAM Modules ---
        self.cbam_blocks = nn.ModuleList()
        for _ in range(self.cbam_layers):
            self.cbam_blocks.append(CBAMBlock(channel=cnn_dims)) 

        # --- BiLSTM ---
        self.bilstm = nn.LSTM(
            input_size=cnn_dims,           
            hidden_size=lstm_hidden_size,  
            num_layers=lstm_layers,         
            bidirectional=True,             
            batch_first=True,               
            dropout=dropout_rate if lstm_layers > 1 else 0 
        )
        self.lstm_output_dim = lstm_hidden_size * 2

        # --- Fully Connected ---
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
        # [B, L, C] -> [B, C, L, 1]
        x = x.permute(0, 2, 1).unsqueeze(-1) 
        
        for block in self.cnn_blocks:
            x = block(x) 
        
        for cbam_block in self.cbam_blocks:
            x = cbam_block(x) 
        
        # [B, C, L, 1] -> [B, L, C]
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
# 5. 数据处理与工具函数
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

def extract_RNA_FM_features(sequences, max_seq_length):
    """提取 RNA-FM 原始特征 (不在此处做PCA)"""
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
            
            # 显式清理显存
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
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1, 'FPR': fpr, 'TPR': sn}

# =============================================================================
# 6. 训练与验证流程 (无 Loss 输出)
# =============================================================================

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_acc, total_count = 0, 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(features)
        
        # 仅计算 Backward，不打印/返回 Loss
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        
        preds_cls = preds.argmax(1)
        total_acc += (preds_cls == labels).sum().item()
        total_count += labels.size(0)
        
    avg_acc = total_acc / total_count
    return avg_acc

def validate_one_epoch(dataloader, model, device):
    # 移除 loss_fn 和 loss 计算
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
                train_sequences, val_sequences, config, epochs, train_loader, val_loader, test_loader):
    # 初始化模型
    model = Deep_dsRNAPred(
        max_seq_length=config.max_time_steps,
        input_size=config.input_size,
        cnn_layers=3,        
        cnn_dims=512,        
        pool_size=2,
        num_layers=3,
        num_dims=64,
        dropout_rate=0.2,
        num_classes=config.num_classes,
        cbam_layers=CBAM_LAYERS 
    ).to(config.device)
    
    loss_fn = nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    best_val_acc = 0.0
    best_val_metrics = None
    print(f"\n===== Fold {fold + 1}/{KFOLD} (CBAM: {CBAM_LAYERS}) =====")
    
    for epoch in range(epochs):
        train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer, config.device)
        val_metrics = validate_one_epoch(val_loader, model, config.device)
        scheduler.step()
        
        if val_metrics['ACC'] > best_val_acc:
            best_val_acc = val_metrics['ACC']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), f"{config.save_model_prefix}{fold + 1}_best.pth")
        
        # 按照要求顺序打印，无 Loss
        print(f"Epoch {epoch + 1:3d} | Train Acc: {train_acc:.1%} | "
              f"Val Sn: {val_metrics['Sn']:.1%} | Val Sp: {val_metrics['Sp']:.1%} | "
              f"Val ACC: {val_metrics['ACC']:.1%} | Val MCC: {val_metrics['MCC']:.3f} | "
              f"Val F1: {val_metrics['F1']:.3f} | Val AUC: {val_metrics['AUC']:.3f}")
    
    # 测试集评估
    model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=config.device))
    test_metrics = validate_one_epoch(test_loader, model, config.device)
    
    print(f"Fold {fold + 1} Best Result | "
          f"Test Sn: {test_metrics['Sn']:.1%} | Test Sp: {test_metrics['Sp']:.1%} | "
          f"Test ACC: {test_metrics['ACC']:.1%} | Test MCC: {test_metrics['MCC']:.3f} | "
          f"Test F1: {test_metrics['F1']:.3f} | Test AUC: {test_metrics['AUC']:.3f}")
    
    return {
        'fold': fold + 1,
        'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
        'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
        'best_val_AUC': best_val_metrics['AUC'], 'best_val_F1': best_val_metrics['F1'],
        'test_Sn': test_metrics['Sn'],
        'test_Sp': test_metrics['Sp'], 
        'test_ACC': test_metrics['ACC'],
        'test_MCC': test_metrics['MCC'], 
        'test_AUC': test_metrics['AUC'],
        'test_F1': test_metrics['F1'],
        'test_FPR': test_metrics['FPR'], 
        'test_TPR': test_metrics['TPR']
    }

# =============================================================================
# 7. 主执行程序
# =============================================================================

if __name__ == "__main__":
    
    

    # --- 步骤 1: 读取数据 ---
    print(f"Reading training data from: {TRAIN_EXCEL_PATH}")
    cv_df = pd.read_excel(TRAIN_EXCEL_PATH)
    print(f"Reading testing data from: {TEST_EXCEL_PATH}")
    test_df = pd.read_excel(TEST_EXCEL_PATH)

    assert "label" in cv_df.columns and "Sequence" in cv_df.columns
    assert "label" in test_df.columns and "Sequence" in test_df.columns

    print(f"CV Data Size: {len(cv_df)} | Pos: {sum(cv_df['label'])} | Neg: {len(cv_df)-sum(cv_df['label'])}")
    print(f"Test Data Size: {len(test_df)} | Pos: {sum(test_df['label'])} | Neg: {len(test_df)-sum(test_df['label'])}")

    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].tolist()

    MAX_SEQ_LENGTH = max([len(seq) for seq in cv_sequences + test_sequences])
    print(f"Auto-detected Max Sequence Length: {MAX_SEQ_LENGTH}")

    # --- 步骤 2: 仅提取原始特征 (Feature Extraction Only) ---
    # 注意：这里只提取 Raw Feature，不做 PCA，防止数据泄露
    print("Starting Feature Extraction for CV set (RNA-FM)...")
    X_cv_raw = extract_RNA_FM_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values
    
    print("Starting Feature Extraction for Test set (RNA-FM)...")
    X_test_raw = extract_RNA_FM_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values
    
    gc.collect()

    # --- 步骤 3: 5折交叉验证 (PCA Inside Loop) ---
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
        
        # 1. 划分 Raw Data
        train_features_raw = X_cv_raw[train_idx]
        train_labels = y_cv[train_idx]
        train_sequences = [cv_sequences[i] for i in train_idx]
        
        val_features_raw = X_cv_raw[val_idx]
        val_labels = y_cv[val_idx]
        val_sequences = [cv_sequences[i] for i in val_idx]
        
        # 2. PCA Fitting (仅在 Train 上 Fit)
        # Reshape: [N, L, C] -> [N*L, C] for PCA
        N_train, L_train, C_train = train_features_raw.shape
        train_flat = train_features_raw.reshape(-1, C_train)
        
        pca = PCA(n_components=RNA_FM_HIDDEN_DIM, random_state=42)
        pca.fit(train_flat) 
        
        # 3. Transform Train
        train_red = pca.transform(train_flat)
        train_features = train_red.reshape(N_train, L_train, RNA_FM_HIDDEN_DIM).astype(np.float32)
        
        # 4. Transform Val (使用 Train 的 PCA 参数)
        N_val, L_val, C_val = val_features_raw.shape
        val_flat = val_features_raw.reshape(-1, C_val)
        val_red = pca.transform(val_flat)
        val_features = val_red.reshape(N_val, L_val, RNA_FM_HIDDEN_DIM).astype(np.float32)
        
        # 5. Transform Test (使用当前折 Train 的 PCA 参数)
        N_test, L_test, C_test = X_test_raw.shape
        test_flat = X_test_raw.reshape(-1, C_test)
        test_red = pca.transform(test_flat)
        test_features = test_red.reshape(N_test, L_test, RNA_FM_HIDDEN_DIM).astype(np.float32)

        # 内存清理：删除中间变量
        del train_flat, train_red, val_flat, val_red, test_flat, test_red
        gc.collect()
        
        # 6. Build Loaders
        train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = build_dataloader(test_sequences, test_labels, test_features, batch_size=BATCH_SIZE, shuffle=False)
        
        # 7. Train
        fold_result = kfold_train(fold, train_features, train_labels, val_features, val_labels, 
                                  train_sequences, val_sequences, config, EPOCHS, 
                                  train_loader, val_loader, test_loader)
        all_fold_results.append(fold_result)
        
        # 清理当前折数据
        del train_features, val_features, test_features
        gc.collect()
    
    # --- 步骤 4: 结果汇总 ---
    get_avg = lambda key: np.mean([res[key] for res in all_fold_results])
    get_std = lambda key: np.std([res[key] for res in all_fold_results])
    
    print("\n" + "=" * 80)
    print(f"5-Fold CV Summary (Model: Deep_dsRNAPred, Feature: RNA-FM+StrictPCA(50))")
    print(f"Max Seq Len: {MAX_SEQ_LENGTH} | CBAM Layers: {CBAM_LAYERS}")
    
    print("\n【Validation Avg Metrics】")
    print(f"Sn:  {get_avg('best_val_Sn'):.3f} ± {get_std('best_val_Sn'):.3f}")
    print(f"Sp:  {get_avg('best_val_Sp'):.3f} ± {get_std('best_val_Sp'):.3f}")
    print(f"ACC: {get_avg('best_val_ACC'):.3f} ± {get_std('best_val_ACC'):.3f}")
    print(f"MCC: {get_avg('best_val_MCC'):.3f} ± {get_std('best_val_MCC'):.3f}")
    print(f"F1:  {get_avg('best_val_F1'):.3f} ± {get_std('best_val_F1'):.3f}")
    print(f"AUC: {get_avg('best_val_AUC'):.3f} ± {get_std('best_val_AUC'):.3f}")
    
    print("\n【Test Avg Metrics】")
    print(f"Sn:  {get_avg('test_Sn'):.3f} ± {get_std('test_Sn'):.3f}")
    print(f"Sp:  {get_avg('test_Sp'):.3f} ± {get_std('test_Sp'):.3f}")
    print(f"ACC: {get_avg('test_ACC'):.3f} ± {get_std('test_ACC'):.3f}")
    print(f"MCC: {get_avg('test_MCC'):.3f} ± {get_std('test_MCC'):.3f}")
    print(f"F1:  {get_avg('test_F1'):.3f} ± {get_std('test_F1'):.3f}")
    print(f"AUC: {get_avg('test_AUC'):.3f} ± {get_std('test_AUC'):.3f}")
    print("=" * 80)

    # --- 步骤 5: 保存结果 ---
    result_df = pd.DataFrame(all_fold_results)
    result_save_path = "Deep_dsRNAPred_Performance.xlsx"
    result_df.to_excel(result_save_path, index=False)
    print(f"\nPerformance metrics saved to: {result_save_path}")