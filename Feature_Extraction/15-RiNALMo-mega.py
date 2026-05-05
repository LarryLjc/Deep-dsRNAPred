import os
import sys
import time
import numpy as np
import pandas as pd
import random
import warnings
import gc
from collections import defaultdict
from tqdm import tqdm  

# PyTorch & Transformers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
from multimolecule import RnaTokenizer

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# Linear models & Calibrator
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Tree & Ensemble models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# --- GPU Libraries Check (RAPIDS cuML) ---
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.naive_bayes import GaussianNB as cuNB
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_CUML = True
    print("✅ RAPIDS cuML library detected. Using GPU for classic ML models.")
except ImportError:
    HAS_CUML = False
    print("⚠️ RAPIDS cuML not found. Falling back to CPU for classic ML.")

# =============================================================================
# 1. Global Configuration
# =============================================================================
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

KFOLD = 5
RANDOM_SEED = 3407

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for feature extraction: {device}")

MP_RNA_PATH = r'/root/autodl-tmp/Big_Model/rinalmo-mega'
BATCH_SIZE = 128  

# 特征本地保存路径
TRAIN_FEAT_CACHE = "/root/autodl-tmp/data/X_cv_pooled_features_rinalmo_mega.npy"
TEST_FEAT_CACHE = "/root/autodl-tmp/data/X_test_pooled_features_rinalmo_mega.npy"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(RANDOM_SEED)

# =============================================================================
# 2. Optimized Feature Extraction 
# =============================================================================
def extract_MP_RNA_features(sequences, max_seq_length, tokenizer, model, desc="Extracting Features"):
    hidden_states_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), BATCH_SIZE), desc=desc):
            batch_seq = sequences[i : i + BATCH_SIZE]
            batch_seq = [str(s).upper().replace('U', 'T').strip() for s in batch_seq]
            batch_seq = [s if s else "A" for s in batch_seq]
            
            inputs = tokenizer(
                batch_seq, return_tensors="pt", padding="max_length",
                truncation=True, max_length=max_seq_length
            ).to(device)
            
            outputs = model(**inputs)
            batch_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            mask = inputs['attention_mask'].unsqueeze(-1).expand(batch_hidden.size()).float()
            
            # Mean Pooling
            sum_embeddings = torch.sum(batch_hidden * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # Max Pooling
            batch_hidden_masked = batch_hidden.masked_fill(mask == 0, -1e9)
            max_pooled = torch.max(batch_hidden_masked, 1)[0]
            
            final_pooled = torch.cat([mean_pooled, max_pooled], dim=1)
            hidden_states_list.append(final_pooled.cpu().half())
            
            del inputs, outputs, batch_hidden
            torch.cuda.empty_cache()
            
    all_hidden = torch.cat(hidden_states_list, dim=0)
    return all_hidden.numpy()

# =============================================================================
# 3. Utilities & Metrics
# =============================================================================
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df = df[df["Sequence"] != "NAN"] 
    df["Sequence"] = df["Sequence"].replace("", "ATG")
    return df.reset_index(drop=True)

def calculate_metrics(y_true, y_pred, y_prob):
    # Handle both CPU numpy arrays and GPU cupy arrays smoothly
    if hasattr(y_true, 'to_numpy'): y_true = y_true.to_numpy()
    elif hasattr(y_true, 'get'): y_true = y_true.get()
    if hasattr(y_pred, 'to_numpy'): y_pred = y_pred.to_numpy()
    elif hasattr(y_pred, 'get'): y_pred = y_pred.get()
    if hasattr(y_prob, 'to_numpy'): y_prob = y_prob.to_numpy()
    elif hasattr(y_prob, 'get'): y_prob = y_prob.get()

    y_true = np.nan_to_num(y_true, nan=0).astype(int)
    y_pred = np.nan_to_num(y_pred, nan=0).astype(int)
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    
    try: tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except: tn, fp, fn, tp = 0, 0, 0, 0
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try: auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else 0.5
    except: auc = 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. Comprehensive Model Initialization
# =============================================================================
def init_models(random_seed=3407):
    models = {}
    
    # SVM Setup with Calibration
    fast_linear_svm = LinearSVC(C=0.05, max_iter=3000, random_state=random_seed, dual=False)
    models['SVM_Calibrated'] = CalibratedClassifierCV(fast_linear_svm, cv=3)
    
    # Conditional RAPIDS vs SKLearn models
    if HAS_CUML:
        models['LogisticRegression'] = cuLogReg(max_iter=3000)
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = cuNB()
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_seed, max_iter=3000, n_jobs=-1)
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = GaussianNB()
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)

    # Scikit-learn Tree Ensembles
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)
    models['HistGradientBoost'] = HistGradientBoostingClassifier(random_state=random_seed, max_iter=100)

    # Boosting Giants
    models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, verbose=-1, device='cuda', n_jobs=-1)
    
    try:
        models['CatBoost'] = CatBoostClassifier(
            random_state=random_seed, verbose=0, allow_writing_files=False, task_type="GPU", devices='0'
        )
    except:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, verbose=0)
    
    try:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=random_seed, use_label_encoder=False, eval_metric='logloss', device='cuda', tree_method='hist'
        )
    except:
        models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, n_jobs=-1)

    return models

# =============================================================================
# 5. Training Logic (Standard Scaler + Soft Voting Integration)
# =============================================================================
def kfold_train_ml(fold, X_train, y_train, X_val, y_val, X_test):
    start_time = time.time()
    
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # StandardScaler is MANDATORY for SVM, LR, and KNN. 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n--- Fold {fold+1}/{KFOLD} ---")
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Validation Evaluation
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            # Test Probabilities (For external Soft Voting)
            y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            if hasattr(y_test_prob, 'get'): y_test_prob = y_test_prob.get()
            
            fold_results[name] = {'val': val_metrics, 'test_prob': y_test_prob}
            print(f"[{name:<20}] Val ACC: {val_metrics['ACC']:.3f} | AUC: {val_metrics['AUC']:.3f}")
            
        except Exception as e:
            print(f"[{name:<20}] Failed: {e}")
            fold_results[name] = None
            
    print(f"Fold completed in {time.time() - start_time:.2f}s")
    del models
    gc.collect()
    return fold_results

# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    train_path = "/root/autodl-tmp/data/train_RNA.xlsx"
    test_path = "/root/autodl-tmp/data/test_RNA.xlsx"

    if os.path.exists(train_path) and os.path.exists(test_path):
        cv_df = read_excel_data(train_path)
        test_df = read_excel_data(test_path)
        
        print(f"🚀 Data Loaded. Training on: {len(cv_df)} | Testing on: {len(test_df)}")
        
        cv_sequences = cv_df["Sequence"].tolist()
        y_cv = cv_df["label"].values
        test_sequences = test_df["Sequence"].tolist()
        y_test = test_df["label"].values

        all_lens = [len(str(s)) for s in cv_sequences + test_sequences]
        max_len = max(all_lens) if all_lens else 0

        print("\n" + "="*50)
        
        if os.path.exists(TRAIN_FEAT_CACHE) and os.path.exists(TEST_FEAT_CACHE):
            print(f"🎯 发现 {MP_RNA_PATH.split('/')[-1]} 模型的本地特征缓存！直接加载...")
            X_cv = np.load(TRAIN_FEAT_CACHE)
            X_test = np.load(TEST_FEAT_CACHE)
            print(f"✅ 加载成功！特征维度: CV {X_cv.shape}, Test {X_test.shape}")
        else:
            print("💡 未发现本地缓存。正在提取特征...")
            tokenizer = RnaTokenizer.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True)
            mp_rna_model = AutoModel.from_pretrained(MP_RNA_PATH, trust_remote_code=True, local_files_only=True).to(device)
            mp_rna_model.eval()
            
            X_cv = extract_MP_RNA_features(cv_sequences, max_len, tokenizer, mp_rna_model, desc="CV Features")
            X_test = extract_MP_RNA_features(test_sequences, max_len, tokenizer, mp_rna_model, desc="Test Features")
            
            np.save(TRAIN_FEAT_CACHE, X_cv)
            np.save(TEST_FEAT_CACHE, X_test)
            del mp_rna_model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
        print("="*50 + "\n")

        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        
        sample_models = init_models()
        model_names = sample_models.keys()
        
        final_results = {name: {'val': []} for name in model_names}
        test_prob_accumulators = {name: np.zeros(len(y_test)) for name in model_names}
        
        print("🔥 开始交叉验证并对 Test 集执行 Soft Voting...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(cv_df)):
            
            X_train_fold = X_cv[train_idx]
            y_train_fold = y_cv[train_idx]
            X_val_fold = X_cv[val_idx]
            y_val_fold = y_cv[val_idx]
            
            fold_res = kfold_train_ml(
                fold, X_train_fold, y_train_fold, X_val_fold, y_val_fold, X_test
            )
            
            for name, res in fold_res.items():
                if res:
                    final_results[name]['val'].append(res['val'])
                    test_prob_accumulators[name] += res['test_prob']

        # =====================================================================
        # 统一输出 Val(5折平均) 和 Test(软投票) 结果
        # =====================================================================
        print("\n" + "="*160)
        print(f"{'Model (RiNALMo-Mega + SoftVote)':<32} | {'Set':<5} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("="*160)
        
        for name in model_names:
            metrics_list = final_results[name]['val']
            if not metrics_list: continue
            
            # --- Val Metrics (5折均值 ± 标准差) ---
            avg = {k: np.mean([x[k] for x in metrics_list]) for k in metrics_list[0]}
            std = {k: np.std([x[k] for x in metrics_list]) for k in metrics_list[0]}
            
            row_val = f"{name:<32} | VAL   | "
            row_val += " | ".join([f"{avg[k]:.3f}±{std[k]:.3f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
            print(row_val)
            
            # --- Test Metrics (5 折 Soft Voting 软投票集成) ---
            avg_test_prob = test_prob_accumulators[name] / KFOLD
            final_y_test_pred = (avg_test_prob >= 0.5).astype(int)
            test_metrics = calculate_metrics(y_test, final_y_test_pred, avg_test_prob)
            
            row_test = f"{' ' * 32} | TEST  | "
            row_test += " | ".join([f"{test_metrics[k]:.3f}      " for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
            print(row_test)
            print("-" * 160)
            
    else:
        print("❌ Error: Data files not found.")