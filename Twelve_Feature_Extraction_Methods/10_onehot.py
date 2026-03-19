import os
import sys
import time
import numpy as np
import pandas as pd
import random
import warnings
from collections import defaultdict

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# --- GPU Libraries Check (RAPIDS cuML) ---
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.naive_bayes import GaussianNB as cuNB
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_CUML = True
    print("✅ RAPIDS cuML library detected. Using GPU for classic ML models.")
except ImportError:
    HAS_CUML = False
    print("⚠️ RAPIDS cuML not found. Falling back to CPU.")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =============================================================================
# 1. Global Configuration
# =============================================================================
warnings.filterwarnings("ignore")
KFOLD = 5
RANDOM_SEED = 3407

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)

# =============================================================================
# 2. Optimized Feature Extraction
# =============================================================================
class OneHotFeatureExtractor:
    def __init__(self, max_length=None):
        self.alphabet = "ACGT"
        self.char_to_int = {c: i for i, c in enumerate(self.alphabet)}
        self.max_length = max_length

    def transform(self, sequences):
        processed_seqs = []
        lengths = []
        for s in sequences:
            s = str(s).upper().replace('U', 'T').strip()
            if not s: s = "A"
            processed_seqs.append(s)
            lengths.append(len(s))
            
        if self.max_length is None:
            self.max_length = max(lengths) if lengths else 0
            
        n_samples = len(sequences)
        n_vocab = len(self.alphabet)
        
        one_hot = np.zeros((n_samples, self.max_length, n_vocab), dtype=np.float32)
        
        for idx, seq in enumerate(processed_seqs):
            seq = seq[:self.max_length]
            indices = [self.char_to_int.get(c, -1) for c in seq]
            valid_pos = [i for i, x in enumerate(indices) if x != -1]
            valid_vals = [x for x in indices if x != -1]
            
            if valid_pos:
                one_hot[idx, valid_pos, valid_vals] = 1.0
                
        return one_hot.reshape(n_samples, -1)

# =============================================================================
# 3. Utilities & Metrics
# =============================================================================
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip().replace("", "ATG")
    return df.reset_index(drop=True)

def augment_rna_sequence(seq):
    if not seq: return 'ATG'
    if random.random() < 0.1:
        base_swap = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        idx = random.randint(0, len(seq)-1)
        if seq[idx] in base_swap:
            seq = seq[:idx] + base_swap[seq[idx]] + seq[idx+1:]
    return seq

def calculate_metrics(y_true, y_pred, y_prob):
    if hasattr(y_true, 'to_numpy'): y_true = y_true.to_numpy()
    elif hasattr(y_true, 'get'): y_true = y_true.get()
    if hasattr(y_pred, 'to_numpy'): y_pred = y_pred.to_numpy()
    elif hasattr(y_pred, 'get'): y_pred = y_pred.get()
    if hasattr(y_prob, 'to_numpy'): y_prob = y_prob.to_numpy()
    elif hasattr(y_prob, 'get'): y_prob = y_prob.get()

    y_true = np.nan_to_num(y_true, nan=0).astype(int)
    y_pred = np.nan_to_num(y_pred, nan=0).astype(int)
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        if len(np.unique(y_true)) < 2: auc = 0.5
        else: auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. Model Initialization
# =============================================================================
def init_models(random_seed=3407):
    models = {}
    
    if HAS_CUML:
        models['LogisticRegression'] = cuLogReg(max_iter=1000)
        models['SVM'] = cuSVC(probability=True, random_state=random_seed, C=1.0)
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = cuNB()
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_seed, max_iter=1000, n_jobs=-1)
        models['SVM'] = SVC(probability=True, random_state=random_seed)
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = GaussianNB()
        models['RandomForest'] = RandomForestClassifier(random_state=random_seed, n_jobs=-1)

    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
    models['GradientBoosting'] = GradientBoostingClassifier(random_state=random_seed)
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)

    models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, verbose=-1, n_jobs=-1)
    
    try:
        models['CatBoost'] = CatBoostClassifier(
            random_state=random_seed, verbose=0, allow_writing_files=False,
            task_type="GPU", devices='0'
        )
    except:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, verbose=0)
    
    try:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=random_seed, use_label_encoder=False, eval_metric='logloss',
            device='cuda', tree_method='hist'
        )
    except:
        models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, n_jobs=-1)

    return models

# =============================================================================
# 5. Training Logic 
# =============================================================================
def kfold_train_ml(fold, train_seqs, train_labs, val_seqs, val_labs, test_seqs, test_labs):
    start_time = time.time()
    
    all_lens = [len(str(s)) for s in train_seqs]
    max_len = max(all_lens) if all_lens else 0
    extractor = OneHotFeatureExtractor(max_length=max_len)
    
    train_seqs_aug = []
    train_labs_aug = []
    for seq, lab in zip(train_seqs, train_labs):
        train_seqs_aug.append(seq)
        train_labs_aug.append(lab)
        train_seqs_aug.append(augment_rna_sequence(seq))
        train_labs_aug.append(lab)
        
    X_train = extractor.transform(train_seqs_aug)
    y_train = np.array(train_labs_aug, dtype=np.float32)
    X_val = extractor.transform(val_seqs)
    y_val = np.array(val_labs, dtype=np.float32)
    X_test = extractor.transform(test_seqs)
    y_test = np.array(test_labs, dtype=np.float32)

    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n--- Fold {fold+1}/{KFOLD} ---")
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            fold_results[name] = {'val': val_metrics, 'test': test_metrics}
            print(f"[{name:<18}] Val ACC: {val_metrics['ACC']:.3f} | AUC: {val_metrics['AUC']:.3f}")
            
        except Exception as e:
            print(f"[{name}] Failed: {e}")
            fold_results[name] = None
            
    print(f"Fold completed in {time.time() - start_time:.2f}s")
    return fold_results

# =============================================================================
# 6. Main Execution
# =============================================================================
if __name__ == "__main__":
    train_path = "/root/autodl-tmp/data/train_combined.xlsx"
    test_path = "/root/autodl-tmp/data/test_combined.xlsx"
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        cv_df = read_excel_data(train_path)
        test_df = read_excel_data(test_path)
        
        print(f"🚀 Data Loaded. Training on: {len(cv_df)} | Testing on: {len(test_df)}")
        
        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        
        sample_models = init_models()
        model_names = sample_models.keys()
        final_results = {name: {'val': [], 'test': []} for name in model_names}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(cv_df)):
            train_sub = cv_df.iloc[train_idx]
            val_sub = cv_df.iloc[val_idx]
            
            fold_res = kfold_train_ml(
                fold, 
                train_sub["Sequence"].tolist(), train_sub["label"].values,
                val_sub["Sequence"].tolist(), val_sub["label"].values,
                test_df["Sequence"].tolist(), test_df["label"].values
            )
            
            for name, res in fold_res.items():
                if res:
                    final_results[name]['val'].append(res['val'])
                    final_results[name]['test'].append(res['test'])

        # --- Report Summary (Deep-dsRNAPred) ---
        print("\n" + "="*160)
        print(f"{'Model (Deep-dsRNAPred)':<22} | {'Set':<5} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("="*160)
        
        for name in model_names:
            res = final_results[name]
            for dtype in ['val', 'test']:
                metrics = res[dtype]
                if not metrics: continue
                avg = {k: np.mean([x[k] for x in metrics]) for k in metrics[0]}
                std = {k: np.std([x[k] for x in metrics]) for k in metrics[0]}
                
                row = f"{name:<22} | {dtype.upper():<5} | "
                row += " | ".join([f"{avg[k]:.3f}±{std[k]:.3f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
                print(row)
            print("-" * 160)
    else:
        print("❌ Error: Data files not found.")