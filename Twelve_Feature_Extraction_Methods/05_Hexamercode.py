import numpy as np
import pandas as pd
import random
import warnings
import os
import math
import itertools
from collections import defaultdict, Counter

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
# 2. Feature Extraction: Hexamer Score (CPAT-style)
# =============================================================================

class HexamerFeatureExtractor:
    """
    Calculates Log-Likelihood Ratio based on Hexamer frequencies 
    observed in Coding vs Non-coding training data.
    """
    def __init__(self, word_size=6):
        self.word_size = word_size
        self.coding_freq = {}
        self.noncoding_freq = {}
        self.is_fitted = False

    def _word_generator(self, seq, step_size=1, frame=0):
        length = len(seq)
        for i in range(frame, length, step_size):
            word = seq[i:i + self.word_size]
            if len(word) == self.word_size:
                yield word

    def _get_freq_table(self, sequences, step_size=1, frame=0):
        count_table = Counter()
        for seq in sequences:
            if len(seq) < self.word_size: 
                continue
            count_table.update(self._word_generator(seq, step_size=step_size, frame=frame))
        
        total = sum(count_table.values())
        freq_dict = {}
        if total > 0:
            for kmer, count in count_table.items():
                if 'N' in kmer: continue
                freq_dict[kmer] = float(count) / total
        return freq_dict

    def fit(self, sequences, labels):
        sequences = [s.upper().replace('U', 'T') for s in sequences]
        
        coding_seqs = [s for s, l in zip(sequences, labels) if l == 1]
        noncoding_seqs = [s for s, l in zip(sequences, labels) if l == 0]
        
        # Coding: step=3 (frame specific), Non-coding: step=1 (background)
        self.coding_freq = self._get_freq_table(coding_seqs, step_size=3, frame=0)
        self.noncoding_freq = self._get_freq_table(noncoding_seqs, step_size=1, frame=0)
        
        self.is_fitted = True
        return self

    def _calculate_score(self, seq):
        if not self.is_fitted:
            raise ValueError("Extractor not fitted.")
        
        if len(seq) < self.word_size:
            return 0.0

        seq = seq.upper().replace('U', 'T')
        sum_of_log_ratio = 0.0
        count = 0
        
        for kmer in self._word_generator(seq, step_size=1, frame=0):
            if kmer not in self.coding_freq and kmer not in self.noncoding_freq:
                continue
            
            cod_p = self.coding_freq.get(kmer, 0.0)
            noncod_p = self.noncoding_freq.get(kmer, 0.0)
            
            if cod_p > 0 and noncod_p > 0:
                sum_of_log_ratio += math.log(cod_p / noncod_p)
            elif cod_p > 0 and noncod_p == 0:
                sum_of_log_ratio += 1
            elif cod_p == 0 and noncod_p > 0:
                sum_of_log_ratio -= 1
            else:
                continue
                
            count += 1
            
        if count > 0:
            return sum_of_log_ratio / count
        else:
            return -1.0

    def transform(self, sequences):
        scores = []
        for seq in sequences:
            scores.append(self._calculate_score(seq))
        return np.array(scores, dtype=np.float32).reshape(-1, 1)

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
    # 1. Initialize Extractor per fold to prevent leakage
    extractor = HexamerFeatureExtractor(word_size=6)
    
    # 2. Augment Training Data
    train_seqs_aug = []
    train_labs_aug = []
    for seq, lab in zip(train_sequences, train_labels):
        train_seqs_aug.append(seq)
        train_labs_aug.append(lab)
        aug_seq = augment_rna_sequence(seq)
        train_seqs_aug.append(aug_seq)
        train_labs_aug.append(lab)
        
    # 3. Fit Extractor
    extractor.fit(train_seqs_aug, train_labs_aug)
    
    # 4. Transform Data
    X_train = extractor.transform(train_seqs_aug)
    y_train = np.array(train_labs_aug, dtype=int)
    
    X_val = extractor.transform(val_sequences)
    y_val = np.array(val_labels, dtype=int)
    
    X_test = extractor.transform(test_sequences)
    y_test = np.array(test_labels, dtype=int)
    
    # 5. Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # 6. Train Models
    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n===== Fold {fold+1}/{KFOLD} - Training (Hexamer Score) =====")
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

        print(f"Train Set Size: {len(cv_df)} | Pos/Neg: {sum(cv_df['label'])}/{len(cv_df)-sum(cv_df['label'])}")
        print(f"Test Set Size:  {len(test_df)} | Pos/Neg: {sum(test_df['label'])}/{len(test_df)-sum(test_df['label'])}")

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