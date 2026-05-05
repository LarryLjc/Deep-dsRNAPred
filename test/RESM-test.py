import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import os
import sys
import gc
import pickle
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

# Import transformers components
from transformers import EsmModel, EsmTokenizer, EsmConfig
from sklearn.metrics import confusion_matrix, recall_score, matthews_corrcoef, roc_auc_score, f1_score

# =============================================================================
# 0. 缺失类配置补丁 (保持与训练代码一致)
# =============================================================================
class Config: pass
current_module = sys.modules[__name__]
missing_classes = [
    'Config', 'OptimizerConfig', 'DataConfig', 'TrainConfig', 'TransformerConfig',
    'LoggingConfig', 'ModelConfig', 'LossConfig', 'SchedulerConfig', 'TrainerConfig',
    'LogConfig', 'CallbackConfig', 'CheckpointConfig', 'ExperimentConfig', 'RunnerConfig'
]
for class_name in missing_classes:
    if not hasattr(current_module, class_name):
        setattr(current_module, class_name, Config)

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 全局配置与路径 (请根据实际情况修改)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for testing: {device}")

# --- 待测试数据路径 ---
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_RNA.xlsx"
OUTPUT_SAVE_PATH = "Final_Ensemble_Predictions.xlsx"

# --- 模型依赖路径 ---
RESM_CKPT_PATH = r'/root/autodl-tmp/Big_Model/RESM/RESM-150M-KDNY.ckpt' 
ESM_BASE_MODEL_LOCAL_PATH = '/root/autodl-tmp/Big_Model/esm2_t6_150'

# --- 训练好的权重文件前缀 ---
SAVE_MODEL_PREFIX = 'Deep_dsRNAPred_fold_'
PCA_SAVE_PREFIX = 'pca_model_fold_'

# --- 超参数 (必须与训练时完全一致) ---
PCA_OUTPUT_DIM = 50 
KFOLD = 5
NUM_CLASSES = 2
BATCH_SIZE = 64
DROPOUT_RATE = 0.5
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
CBAM_LAYERS = 2

# =============================================================================
# 2. 模型结构定义 (包含网络架构、注意力机制等)
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
        self.lstm_flatten_dim = lstm_hidden_size * 2 * self.after_cnn_length
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
        x = F.relu(self.mid_fc(x))
        x = self.mid_dropout(x)
        return self.output_layer(x)

# =============================================================================
# 3. 数据处理与 RESM 提特征工具
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

def rna_to_pseudo_protein(seq):
    seq = seq.replace('T', 'U')
    mapping = str.maketrans({'A': 'K', 'U': 'D', 'C': 'N', 'G': 'Y'})
    return seq.translate(mapping)

def get_esm_150m_config():
    return EsmConfig(
        vocab_size=33, hidden_size=640, num_hidden_layers=30, num_attention_heads=20, 
        intermediate_size=2560, max_position_embeddings=1024, pad_token_id=1, mask_token_id=32,
        token_dropout=True, position_embedding_type="rotary", use_cache=False
    )

def load_resm_model_offline(ckpt_path, local_model_path, device):
    tokenizer = EsmTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = EsmModel(get_esm_150m_config())
    if model.config.pad_token_id is None: model.config.pad_token_id = 1
    if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'padding_idx'):
        model.embeddings.padding_idx = 1
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith('model.'): name = name[6:]
            if name.startswith('esm_model.'): name = name[10:]
            if name.startswith('embed_tokens'): name = name.replace('embed_tokens', 'embeddings.word_embeddings')
            name = name.replace('layers.', 'encoder.layer.')
            name = name.replace('self_attn.k_proj', 'attention.self.key')
            name = name.replace('self_attn.v_proj', 'attention.self.value')
            name = name.replace('self_attn.q_proj', 'attention.self.query')
            name = name.replace('self_attn.out_proj', 'attention.output.dense')
            name = name.replace('self_attn_layer_norm', 'attention.LayerNorm')
            name = name.replace('final_layer_norm', 'LayerNorm')
            name = name.replace('fc1', 'intermediate.dense')
            name = name.replace('fc2', 'output.dense')
            if name == 'emb_layer_norm_after.weight': name = 'encoder.emb_layer_norm_after.weight'
            if name == 'emb_layer_norm_after.bias': name = 'encoder.emb_layer_norm_after.bias'
            if 'embed_positions' in name or "contact_head" in name or "lm_head" in name or "regression" in name:
                continue
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model

def extract_RESM_features(sequences, max_seq_length, tokenizer, resm_model):
    mapped_sequences = [rna_to_pseudo_protein(seq) for seq in sequences]
    hidden_states_list = []
    resm_model.to(device)
    for i in range(0, len(mapped_sequences), BATCH_SIZE):
        batch_seq = mapped_sequences[i : i + BATCH_SIZE]
        inputs = tokenizer(batch_seq, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length)
        inputs = {k: v.to(device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs.to(device)
        
        with torch.no_grad():
            outputs = resm_model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            if batch_hidden.size(1) != max_seq_length:
                if batch_hidden.size(1) > max_seq_length:
                    batch_hidden = batch_hidden[:, :max_seq_length, :]
                else:
                    batch_hidden = F.pad(batch_hidden, (0, 0, 0, max_seq_length - batch_hidden.size(1)))
            hidden_states_list.append(batch_hidden.cpu())
            
        del inputs, outputs, batch_hidden
        torch.cuda.empty_cache()
    return torch.cat(hidden_states_list, dim=0).numpy()

def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) == 2 else 0.0
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. 主推理函数
# =============================================================================
def test_ensemble():
    print(">>> STARTING TEST & INFERENCE PIPELINE <<<")
    
    # 1. 加载测试数据
    test_df = pd.read_excel(TEST_EXCEL_PATH)
    assert "Sequence" in test_df.columns, "Excel file must contain a 'Sequence' column."
    test_sequences = test_df["Sequence"].tolist()
    has_labels = "label" in test_df.columns
    if has_labels:
        test_labels = test_df["label"].values

    MAX_SEQ_LENGTH = max([len(seq) for seq in test_sequences])
    print(f"Test Dataset Size: {len(test_sequences)} | Max Sequence Length: {MAX_SEQ_LENGTH}")

    # 2. 提取大模型特征 (提完即删)
    print("\n[Phase 1] Extracting RESM Features...")
    tokenizer, resm_model = load_resm_model_offline(RESM_CKPT_PATH, ESM_BASE_MODEL_LOCAL_PATH, device)
    X_test_raw = extract_RESM_features(test_sequences, MAX_SEQ_LENGTH, tokenizer, resm_model)
    
    print("Deleting RESM model to free GPU memory...")
    del resm_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 3. 加载 5 折模型进行推理
    print("\n[Phase 2] Ensemble Inference with 5 Folds...")
    n_test, seq_len, feat_dim = X_test_raw.shape
    test_flat = X_test_raw.reshape(-1, feat_dim)
    ensemble_probs = np.zeros((KFOLD, len(test_sequences), NUM_CLASSES))

    for fold in range(KFOLD):
        print(f"-> Processing Fold {fold + 1}/{KFOLD}")
        
        # 加载对应的 PCA 模型降维
        pca_path = f"{PCA_SAVE_PREFIX}{fold + 1}.pkl"
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found at: {pca_path}. Please train first.")
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            
        test_red = pca.transform(test_flat)
        test_features_reduced = test_red.reshape(n_test, seq_len, PCA_OUTPUT_DIM).astype(np.float32)
        
        # 准备 Dataloader 和 模型架构
        test_dataset = RNADataset(test_sequences, features=test_features_reduced)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model = Deep_dsRNAPred(
            max_seq_length=MAX_SEQ_LENGTH, input_size=PCA_OUTPUT_DIM, cnn_layers=3, cnn_dims=512, 
            pool_size=2, num_layers=3, num_dims=64, dropout_rate=0.2, num_classes=NUM_CLASSES, cbam_layers=CBAM_LAYERS 
        ).to(device)
        
        # 加载对应的权重
        model_path = f"{SAVE_MODEL_PREFIX}{fold + 1}_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}. Please train first.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # 预测打分
        all_prob = []
        with torch.no_grad():
            for features in test_loader:
                probs = F.softmax(model(features.to(device)), dim=1)
                all_prob.extend(probs.cpu().numpy())
                
        ensemble_probs[fold] = np.array(all_prob)
        
        # 内存回收
        del pca, test_red, test_features_reduced, model, test_loader, test_dataset
        torch.cuda.empty_cache()
        gc.collect()

    # 4. 集成与保存
    print("\n[Phase 3] Calculating Final Results...")
    final_avg_probs = np.mean(ensemble_probs, axis=0) # [N, 2]
    final_preds = np.argmax(final_avg_probs, axis=1)

    test_df['ensemble_prob_class_1'] = final_avg_probs[:, 1]
    test_df['final_prediction'] = final_preds
    for fold in range(KFOLD):
        test_df[f'fold_{fold+1}_prob_class_1'] = ensemble_probs[fold][:, 1]
        
    test_df.to_excel(OUTPUT_SAVE_PATH, index=False)
    print(f"==> Predictions successfully saved to {OUTPUT_SAVE_PATH}")

    # 5. 输出性能指标（如果存在真实标签）
    if has_labels:
        metrics = calculate_metrics(test_labels, final_preds, final_avg_probs)
        print("\n" + "="*40)
        print("【 Test Set Final Metrics (Ensemble) 】")
        print(f" ACC: {metrics['ACC']:.4f}")
        print(f" AUC: {metrics['AUC']:.4f}")
        print(f" MCC: {metrics['MCC']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  Sn: {metrics['Sn']:.4f} (Sensitivity/Recall)")
        print(f"  Sp: {metrics['Sp']:.4f} (Specificity)")
        print("="*40)

if __name__ == "__main__":
    test_ensemble()