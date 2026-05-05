import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import math
import gc  
import joblib  # 核心新增：用于保存和加载 PCA 的 .pkl 文件
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

# Transformers
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from sklearn.model_selection import KFold
# Metrics
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.decomposition import PCA

# =============================================================================
# 1. 全局配置与环境初始化
# =============================================================================

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Unable to import Triton")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 路径配置 ---
MP_RNA_PATH = r'/root/autodl-tmp/Big_Model/MP-RNA'
TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_RNA.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"

# --- 保存前缀配置 ---
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_PREFIX = 'pca_fold_'
OUTPUT_SAVE_PATH = 'Ensemble_Test_Predictions.xlsx'
RESULT_SAVE_PATH = "Deep_dsRNAPred_Performance.xlsx"

# --- 维度与超参数配置 ---
MP_RNA_HIDDEN_DIM = 50  
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 240
DROPOUT_RATE = 0.5
THRESHOLD = 0.5  # 集成概率判断阈值

# --- 模型结构参数 ---
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# 模型加载 (全局只需加载一次)
print(f"Loading MP-RNA from {MP_RNA_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True)
mp_rna_model = AutoModel.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True).to(device)
mp_rna_model.eval()

class ModelConfig:
    def __init__(self, max_time_steps, input_size=50, num_classes=2, dropout=0.5, save_model_prefix='model_'):
        self.device = device
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.save_model_prefix = save_model_prefix

# =============================================================================
# 2. 模型架构定义 (训练和推理共用)
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
# 3. 数据处理与工具函数
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

def extract_MP_RNA_features(sequences, max_seq_length):
    hidden_states_list = []
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = mp_rna_model(**inputs)
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
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # y_prob 兼容性处理：如果是二维数组取第二列，如果是一维直接用
    prob_for_auc = y_prob[:, 1] if len(y_prob.shape) == 2 else y_prob
    auc = roc_auc_score(y_true, prob_for_auc) if len(np.unique(y_true)) == 2 else 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'AUC': auc, 'F1': f1, 'FPR': fpr, 'TPR': sn}

# =============================================================================
# 4. 训练流程 (包含交叉验证与 PCA 保存)
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

def run_training_cv(cv_df, test_df, MAX_SEQ_LENGTH):
    cv_sequences = cv_df["Sequence"].tolist()
    cv_labels = cv_df["label"].tolist()
    test_sequences = test_df["Sequence"].tolist()
    test_labels = test_df["label"].tolist()

    print("Extracting Raw Features (CV)...")
    X_cv_raw = extract_MP_RNA_features(cv_sequences, MAX_SEQ_LENGTH)
    y_cv = cv_df["label"].values

    print("Extracting Raw Features (Test)...")
    X_test_raw = extract_MP_RNA_features(test_sequences, MAX_SEQ_LENGTH)
    y_test = test_df["label"].values
    gc.collect()

    config = ModelConfig(
        max_time_steps=MAX_SEQ_LENGTH,
        input_size=MP_RNA_HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT_RATE,
        save_model_prefix=SAVE_MODEL_PREFIX
    )
    
    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
    all_fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
        print(f"\nPreparing Fold {fold + 1}...")
        
        train_features_raw = X_cv_raw[train_idx]
        train_labels = y_cv[train_idx]
        train_sequences = [cv_sequences[i] for i in train_idx]
        
        val_features_raw = X_cv_raw[val_idx]
        val_labels = y_cv[val_idx]
        val_sequences = [cv_sequences[i] for i in val_idx]
        
        # PCA拟合与保存
        N_train, L_train, C_train = train_features_raw.shape
        train_flat = train_features_raw.reshape(-1, C_train)
        
        pca = PCA(n_components=MP_RNA_HIDDEN_DIM, random_state=42)
        pca.fit(train_flat)
        # 核心：保存PCA模型供后续推理使用
        joblib.dump(pca, f"{PCA_PREFIX}{fold + 1}.pkl")
        print(f"Saved PCA model to {PCA_PREFIX}{fold + 1}.pkl")
        
        # Transform Train
        train_red = pca.transform(train_flat)
        train_features = train_red.reshape(N_train, L_train, MP_RNA_HIDDEN_DIM).astype(np.float32)
        
        # Transform Val
        N_val, L_val, C_val = val_features_raw.shape
        val_flat = val_features_raw.reshape(-1, C_val)
        val_red = pca.transform(val_flat)
        val_features = val_red.reshape(N_val, L_val, MP_RNA_HIDDEN_DIM).astype(np.float32)
        
        # Transform Test
        N_test, L_test, C_test = X_test_raw.shape
        test_flat = X_test_raw.reshape(-1, C_test)
        test_red = pca.transform(test_flat)
        test_features = test_red.reshape(N_test, L_test, MP_RNA_HIDDEN_DIM).astype(np.float32)

        del train_flat, train_red, val_flat, val_red, test_flat, test_red
        del train_features_raw, val_features_raw  
        gc.collect()
        
        train_loader = build_dataloader(train_sequences, train_labels, train_features, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = build_dataloader(val_sequences, val_labels, val_features, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = build_dataloader(test_sequences, test_labels, test_features, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型
        model = Deep_dsRNAPred(
            max_seq_length=config.max_time_steps,
            input_size=config.input_size,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=config.num_classes, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        loss_fn = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
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
        
        # 测试集评估
        model.load_state_dict(torch.load(f"{config.save_model_prefix}{fold + 1}_best.pth", map_location=device))
        test_metrics = validate_one_epoch(test_loader, model)
        
        print(f"Fold {fold + 1} Best Result | Test ACC: {test_metrics['ACC']:.1%} | Test F1: {test_metrics['F1']:.3f}")
        
        all_fold_results.append({
            'fold': fold + 1,
            'best_val_Sn': best_val_metrics['Sn'], 'best_val_Sp': best_val_metrics['Sp'],
            'best_val_ACC': best_val_metrics['ACC'], 'best_val_MCC': best_val_metrics['MCC'],
            'best_val_AUC': best_val_metrics['AUC'], 'best_val_F1': best_val_metrics['F1'],
            'test_Sn': test_metrics['Sn'], 'test_Sp': test_metrics['Sp'], 
            'test_ACC': test_metrics['ACC'], 'test_MCC': test_metrics['MCC'], 
            'test_AUC': test_metrics['AUC'], 'test_F1': test_metrics['F1']
        })

        # 清理
        del train_features, val_features, test_features, train_loader, val_loader, test_loader, pca
        torch.cuda.empty_cache()
        gc.collect()
    
    # 汇总并保存
    result_df = pd.DataFrame(all_fold_results)
    result_df.to_excel(RESULT_SAVE_PATH, index=False)
    print(f"\nTraining complete. Performance metrics saved to: {RESULT_SAVE_PATH}")


# =============================================================================
# 5. 推理流程 (加载多模型集成与软投票机制)
# =============================================================================

def get_model_predictions(model, dataloader):
    model.eval()
    all_prob = []
    with torch.no_grad():
        for features in dataloader:
            # 这里的 Dataloader 吐出的只有 features (无 labels)
            features = features.to(device)
            preds = model(features)
            probs = F.softmax(preds, dim=1)[:, 1] # 获取 class 1 的概率
            all_prob.extend(probs.cpu().numpy())
    return np.array(all_prob)

def run_ensemble_inference(test_df, MAX_SEQ_LENGTH):
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    print("\nExtracting Raw MP-RNA Features (Inference)...")
    X_test_raw = extract_MP_RNA_features(test_sequences, MAX_SEQ_LENGTH)
    gc.collect()

    N_test, L_test, C_test = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, C_test)
    
    ensemble_probs = np.zeros((KFOLD, len(test_sequences)))

    for fold in range(KFOLD):
        print(f"\n--- Inference: Processing Fold {fold + 1}/{KFOLD} ---")
        
        # 1. 加载 PCA 并进行降维
        pca_path = f"{PCA_PREFIX}{fold + 1}.pkl"
        pca = joblib.load(pca_path)
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(N_test, L_test, MP_RNA_HIDDEN_DIM).astype(np.float32)
        
        # 2. 构建无 Label 的 Dataloader
        test_dataset = RNADataset(test_sequences, features=test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # 3. 初始化并加载 Model
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=MP_RNA_HIDDEN_DIM,
            cnn_layers=3, cnn_dims=512, pool_size=2,
            num_layers=3, num_dims=64, dropout_rate=0.2,
            num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # 4. 获取概率并保存
        fold_probs = get_model_predictions(model, test_loader)
        ensemble_probs[fold] = fold_probs
        
        del pca, test_red, test_features_reduced, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 5. 集成计算、概率平均与阈值打分
    print("\n" + "=" * 50)
    print("Calculating Ensemble Predictions...")
    final_avg_probs = np.mean(ensemble_probs, axis=0)
    final_preds = (final_avg_probs >= THRESHOLD).astype(int)

    # 保存结果
    test_df['ensemble_prob'] = final_avg_probs
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob'] = ensemble_probs[fold]
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # 若含有真实标签，计算性能
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n【Ensemble Test Final Metrics】")
        print(f"Threshold: {THRESHOLD} | ACC: {metrics['ACC']:.3f} | MCC: {metrics['MCC']:.3f} | F1: {metrics['F1']:.3f} | AUC: {metrics['AUC']:.3f}")
        print("=" * 50)


# =============================================================================
# 6. 主程序入口 (执行控制)
# =============================================================================
if __name__ == "__main__":
    
    # --- 控制开关，你可以按需将它们设置为 True 或 False ---
    DO_TRAIN = True         # 执行模型训练和 PCA 保存
    DO_INFERENCE = True     # 执行多模型加载、打分与预测
    
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
        run_training_cv(cv_df_global, test_df_global, GLOBAL_MAX_SEQ_LENGTH)
        
    if DO_INFERENCE:
        print("\n" + "#" * 50)
        print(">>> STARTING ENSEMBLE INFERENCE PIPELINE <<<")
        print("#" * 50)
        # 这里仅传入需要测试的数据集
        run_ensemble_inference(test_df_global, GLOBAL_MAX_SEQ_LENGTH)