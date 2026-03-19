import numpy as np
import pandas as pd
import random
import warnings
import os
from collections import defaultdict
from itertools import product

# Scikit-learn related
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

# Filter warnings
warnings.filterwarnings("ignore")

# Global Parameters
KFOLD = 5
RANDOM_SEED = 3407  # Set Global Seed to 3407

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)
print(f"Global Random Seed set to: {RANDOM_SEED}")

# =============================================================================
# 2. Feature Extraction: K-mer Counter (k=3)
# =============================================================================

class KmerCounter:
    """
    Converts sequences into 4^k dimensional frequency vectors.
    Normalized by sequence length (per 1000 bp).
    """
    def __init__(self, k=3):
        self.k = k
        self.kmers = [''.join(i) for i in product('ACGT', repeat=k)]
        self.map = {kmer: i for i, kmer in enumerate(self.kmers)}

    def extract_features(self, seq):
        seq = seq.upper().replace('U', 'T').strip()
        counts = defaultdict(int)
        length = len(seq)
        
        if length < self.k:
            return np.zeros(len(self.kmers), dtype=np.float32)

        # Normalization factor (per kb)
        increment = 1000.0 / length 
        
        for i in range(length - self.k + 1):
            kmer = seq[i:i + self.k]
            if kmer in self.map:
                counts[kmer] += increment
        
        row = np.zeros(len(self.kmers), dtype=np.float32)
        for kmer, n in counts.items():
            row[self.map[kmer]] = n
            
        return row

# Global Feature Extractor
extractor = KmerCounter(k=3)

# =============================================================================
# 3. Data Processing & Augmentation Utils
# =============================================================================

def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    assert "Sequence" in df.columns, "Excel file must contain 'Sequence' column"
    assert "label" in df.columns, "Excel file must contain 'label' column"
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df["Sequence"] = df["Sequence"].replace("", "ATG")
    return df.reset_index(drop=True)

def random_base_swap(seq, swap_prob=0.1):
    base_swap = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    seq_list = list(seq)
    for i in range(len(seq_list)):
        if random.random() < swap_prob and seq_list[i] in base_swap:
            seq_list[i] = base_swap[seq_list[i]]
    return ''.join(seq_list)

def reverse_complement(seq):
    base_complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join([base_complement.get(base, 'N') for base in reversed(seq)])

def augment_rna_sequence(seq):
    if not seq: return 'ATG'
    augmentations = [
        lambda s: random_base_swap(s, swap_prob=0.1),
        reverse_complement,
    ]
    selected = random.sample(augmentations, k=random.randint(1, 2))
    for aug in selected:
        seq = aug(seq)
    return seq if seq else 'ATG'

def generate_feature_matrix(sequences, use_augmentation=False):
    """
    Generates feature matrix using K-mer extraction.
    """
    features = []
    for seq in sequences:
        target_seq = augment_rna_sequence(seq) if use_augmentation else seq
        feat = extractor.extract_features(target_seq)
        features.append(feat)
    return np.array(features, dtype=np.float32)

# =============================================================================
# 4. Metrics Calculation
# =============================================================================

def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.nan_to_num(y_true, nan=0).astype(int)
    y_pred = np.nan_to_num(y_pred, nan=0).astype(int)
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
        tn = fp = fn = tp = 0
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Added F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 5. Model Initialization
# =============================================================================

def init_models(random_seed=42):
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_seed, max_iter=1000),
        'SVM': SVC(probability=True, random_state=random_seed),
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB(),
        'LightGBM': lgb.LGBMClassifier(random_state=random_seed, verbose=-1),
        'DecisionTree': DecisionTreeClassifier(random_state=random_seed),
        'RandomForest': RandomForestClassifier(random_state=random_seed),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_seed),
        'CatBoost': CatBoostClassifier(random_state=random_seed, verbose=0, allow_writing_files=False),
        'XGBoost': xgb.XGBClassifier(random_state=random_seed, use_label_encoder=False, eval_metric='logloss'),
        'AdaBoost': AdaBoostClassifier(random_state=random_seed)
    }
    return models

# =============================================================================
# 6. Training & Evaluation Pipeline
# =============================================================================

def kfold_train_ml(fold, train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels):
    # Generate Feature Matrices
    X_train = generate_feature_matrix(train_sequences, use_augmentation=True)
    y_train = np.array(train_labels, dtype=int)
    
    X_val = generate_feature_matrix(val_sequences, use_augmentation=False)
    y_val = np.array(val_labels, dtype=int)
    
    X_test = generate_feature_matrix(test_sequences, use_augmentation=False)
    y_test = np.array(test_labels, dtype=int)
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Train Models
    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n===== Fold {fold+1}/{KFOLD} - Training (K-mer Features) =====")
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Validation
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            # Test
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_test_pred
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            fold_results[model_name] = {'val': val_metrics, 'test': test_metrics}
            # Progress Log
            # print(f"[{model_name}] Val ACC: {val_metrics['ACC']:.3f} | AUC: {val_metrics['AUC']:.3f}")
        except Exception as e:
            print(f"[{model_name}] Error: {e}")
            empty_metrics = {'Sn':0,'Sp':0,'ACC':0,'MCC':0,'F1':0,'AUC':0}
            fold_results[model_name] = {'val': empty_metrics, 'test': empty_metrics}
    
    return fold_results

# =============================================================================
# 7. Main Execution
# =============================================================================

if __name__ == "__main__":
    # Define Paths
    train_path = "/root/autodl-tmp/data/train_combined.xlsx"
    test_path = "/root/autodl-tmp/data/test_combined.xlsx"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Data files not found. Please check paths.")
    else:
        # Load Data
        print("Loading Data...")
        cv_df = read_excel_data(train_path)
        test_df = read_excel_data(test_path)

        print(f"Train set: {len(cv_df)} | Test set: {len(test_df)}")

        cv_sequences = cv_df["Sequence"].tolist()
        cv_labels = cv_df["label"].values
        test_sequences = test_df["Sequence"].tolist()
        test_labels = test_df["label"].values

        # KFold Cross Validation
        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        
        model_names = init_models().keys()
        all_results = {name: {'val': [], 'test': []} for name in model_names}

        for fold, (train_idx, val_idx) in enumerate(kf.split(cv_sequences)):
            train_seq = [cv_sequences[i] for i in train_idx]
            train_lab = [cv_labels[i] for i in train_idx]
            val_seq = [cv_sequences[i] for i in val_idx]
            val_lab = [cv_labels[i] for i in val_idx]
            
            fold_res = kfold_train_ml(fold, train_seq, train_lab, val_seq, val_lab, test_sequences, test_labels)
            
            for model_name in model_names:
                all_results[model_name]['val'].append(fold_res[model_name]['val'])
                all_results[model_name]['test'].append(fold_res[model_name]['test'])

        # --- Summary Output (Deep-dsRNAPred Benchmark) ---
        print("\n" + "="*160)
        print(f"{'Model':<20} | {'Set':<5} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("="*160)
        
        for model_name, res in all_results.items():
            # 1. Validation Metrics
            metrics = res['val']
            val_sn_m, val_sn_s = np.mean([m['Sn'] for m in metrics]), np.std([m['Sn'] for m in metrics])
            val_sp_m, val_sp_s = np.mean([m['Sp'] for m in metrics]), np.std([m['Sp'] for m in metrics])
            val_acc_m, val_acc_s = np.mean([m['ACC'] for m in metrics]), np.std([m['ACC'] for m in metrics])
            val_mcc_m, val_mcc_s = np.mean([m['MCC'] for m in metrics]), np.std([m['MCC'] for m in metrics])
            val_f1_m, val_f1_s = np.mean([m['F1'] for m in metrics]), np.std([m['F1'] for m in metrics])
            val_auc_m, val_auc_s = np.mean([m['AUC'] for m in metrics]), np.std([m['AUC'] for m in metrics])

            print(f"{model_name:<20} | {'Val':<5} | "
                  f"{val_sn_m:.3f}±{val_sn_s:.3f}   | {val_sp_m:.3f}±{val_sp_s:.3f}   | "
                  f"{val_acc_m:.3f}±{val_acc_s:.3f}   | {val_mcc_m:.3f}±{val_mcc_s:.3f}   | "
                  f"{val_f1_m:.3f}±{val_f1_s:.3f}   | {val_auc_m:.3f}±{val_auc_s:.3f}")

            # 2. Test Metrics
            metrics = res['test']
            test_sn_m, test_sn_s = np.mean([m['Sn'] for m in metrics]), np.std([m['Sn'] for m in metrics])
            test_sp_m, test_sp_s = np.mean([m['Sp'] for m in metrics]), np.std([m['Sp'] for m in metrics])
            test_acc_m, test_acc_s = np.mean([m['ACC'] for m in metrics]), np.std([m['ACC'] for m in metrics])
            test_mcc_m, test_mcc_s = np.mean([m['MCC'] for m in metrics]), np.std([m['MCC'] for m in metrics])
            test_f1_m, test_f1_s = np.mean([m['F1'] for m in metrics]), np.std([m['F1'] for m in metrics])
            test_auc_m, test_auc_s = np.mean([m['AUC'] for m in metrics]), np.std([m['AUC'] for m in metrics])

            print(f"{'':<20} | {'Test':<5} | "
                  f"{test_sn_m:.3f}±{test_sn_s:.3f}   | {test_sp_m:.3f}±{test_sp_s:.3f}   | "
                  f"{test_acc_m:.3f}±{test_acc_s:.3f}   | {test_mcc_m:.3f}±{test_mcc_s:.3f}   | "
                  f"{test_f1_m:.3f}±{test_f1_s:.3f}   | {test_auc_m:.3f}±{test_auc_s:.3f}")
            
            print("-" * 160)