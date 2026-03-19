import numpy as np
import pandas as pd
import random
import warnings
import os

# Scikit-learn related
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score # Added f1_score
from sklearn.preprocessing import StandardScaler

# 11 ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# BioPython
from Bio import Seq
import Bio.SeqIO as SeqO

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================

# Filter specific warnings
warnings.filterwarnings("ignore")

# Global Parameters
ORF_FEATURE_DIM = 4
KFOLD = 5
RANDOM_SEED = 3407  # Updated Global Seed

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)
print(f"Global Random Seed set to: {RANDOM_SEED}")

# =============================================================================
# 2. ORF Feature Extraction Classes
# =============================================================================

class ExtractORF:
    def __init__(self, seq):
        self.seq = seq
        self.result = (0, 0, 0, 0)
        self.longest = 0

    def codons(self, frame):
        start_coord = frame
        while start_coord + 3 <= len(self.seq):
            yield (self.seq[start_coord:start_coord + 3], start_coord)
            start_coord += 3

    def longest_orf_in_seq(self, frame_number, start_codon, stop_codon):
        codon_posi = self.codons(frame_number)
        start_codons = start_codon
        stop_codons = stop_codon
        while True:
            try:
                codon, index = next(codon_posi)
            except StopIteration:
                break
            if codon in start_codons and codon not in stop_codons:
                ORF_start = index
                end = False
                while True:
                    try:
                        codon, index = next(codon_posi)
                    except StopIteration:
                        end = True
                        integrity = -1
                    if codon in stop_codons:
                        integrity = 1
                        end = True
                    if end:
                        ORF_end = index + 3
                        ORF_Length = (ORF_end - ORF_start)
                        if ORF_Length > self.longest:
                            self.longest = ORF_Length
                            self.result = (integrity, ORF_start, ORF_end, ORF_Length)
                        if ORF_Length == self.longest and ORF_start < self.result[1]:
                            self.result = (integrity, ORF_start, ORF_end, ORF_Length)
                        break

    def longest_ORF(self, start=['ATG'], stop=['TAA', 'TAG', 'TGA']):
        orf_seq = ""
        for frame in range(3):
            self.longest_orf_in_seq(frame, start, stop)
        orf_seq = self.seq[self.result[1]:self.result[2]] if self.result[1] < self.result[2] else ""
        ORF_integrity = self.result[0]
        ORF_length = self.result[3]
        return ORF_length, ORF_integrity, orf_seq

class ORF_count():
    def __init__(self):
        self.start_codons = 'ATG'
        self.stop_codons = 'TAG,TAA,TGA'
        self.Coverage = 0

    def extract_feature_from_seq(self, seq, stt, stp):
        stt_coden = stt.strip().split(',')
        stp_coden = stp.strip().split(',')
        mRNA_seq = seq.upper()
        mRNA_size = len(seq)
        tmp = ExtractORF(mRNA_seq)
        (CDS_size1, CDS_integrity, CDS_seq1) = tmp.longest_ORF(start=stt_coden, stop=stp_coden)
        return (mRNA_size, CDS_size1, CDS_integrity)

    def len_cov(self, seq):
        if len(seq.strip()) == 0:
            return (0.0, 0.0, 0.0)
        
        (mRNA_size, CDS_size, CDS_integrity) = self.extract_feature_from_seq(
            seq=seq, stt=self.start_codons, stp=self.stop_codons
        )
        mRNA_len = mRNA_size if mRNA_size != 0 else 1
        CDS_len = CDS_size if CDS_size >= 0 else 0
        self.Coverage = float(CDS_len) / mRNA_len
        Integrity = CDS_integrity if CDS_integrity in (-1, 1) else 0
        return (CDS_len, self.Coverage, Integrity)

    def get_orf(self, seq, stt, stp):
        if len(seq.strip()) == 0:
            return ("", 0)
        stt_coden = stt.strip().split(',')
        stp_coden = stp.strip().split(',')
        mRNA_seq = seq.upper()
        tmp = ExtractORF(mRNA_seq)
        (CDS_size1, CDS_integrity, CDS_seq1) = tmp.longest_ORF(start=stt_coden, stop=stp_coden)
        return (CDS_seq1, CDS_integrity)

    def get_orf_frame_score(self, seq):
        if len(seq.strip()) == 0:
            return 0.0
        
        ORF_length_in_frame1, _ = self.get_orf(seq, stt=self.start_codons, stp=self.stop_codons)
        len1 = len(ORF_length_in_frame1) if ORF_length_in_frame1 is not None else 0
        len1 = max(0, len1)

        ORF_length_in_frame2, _ = self.get_orf(seq[1:], stt=self.start_codons, stp=self.stop_codons)
        len2 = len(ORF_length_in_frame2) if ORF_length_in_frame2 is not None else 0
        len2 = max(0, len2)

        ORF_length_in_frame3, _ = self.get_orf(seq[2:], stt=self.start_codons, stp=self.stop_codons)
        len3 = len(ORF_length_in_frame3) if ORF_length_in_frame3 is not None else 0
        len3 = max(0, len3)

        ORF_len = [len1, len2, len3]
        try:
            ORF_frame = ((ORF_len[0] - ORF_len[1]) **2 + 
                         (ORF_len[0] - ORF_len[2])** 2 + 
                         (ORF_len[1] - ORF_len[2]) **2) / 2
        except:
            ORF_frame = 0.0
        
        return ORF_frame

    def extract_orf_features(self, seq):
        seq = seq.replace('U', 'T').strip()
        if not seq:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        ORF_len, ORF_cov, ORF_inte = self.len_cov(seq)
        ORF_frame = self.get_orf_frame_score(seq)
        
        features = np.array([ORF_len, ORF_cov, ORF_inte, ORF_frame], dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

orf_extractor = ORF_count()

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
    features = []
    for seq in sequences:
        if use_augmentation:
            aug_seq = augment_rna_sequence(seq)
            feat = orf_extractor.extract_orf_features(aug_seq)
        else:
            feat = orf_extractor.extract_orf_features(seq)
        features.append(feat)
    return np.array(features, dtype=np.float32)

# =============================================================================
# 4. Metrics Calculation
# =============================================================================

def calculate_metrics(y_true, y_pred, y_prob):
    y_true = np.nan_to_num(y_true, nan=0)
    y_pred = np.nan_to_num(y_pred, nan=0)
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    y_true = np.clip(y_true, 0, 1).astype(int)
    y_pred = np.clip(y_pred, 0, 1).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1:
            if y_true[0] == 1:
                tp = np.sum(y_pred == 1)
                fn = np.sum(y_pred == 0)
            else:
                tn = np.sum(y_pred == 0)
                fp = np.sum(y_pred == 1)
    
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0.0
    
    # F1 Score
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0
    try:
        if len(np.unique(y_true)) < 2:
            auc = 0.5
        else:
            y_prob_clamped = np.clip(y_prob, 0.0001, 0.9999)
            auc = roc_auc_score(y_true, y_prob_clamped)
    except:
        auc = 0.0
    
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 5. Model Initialization
# =============================================================================

def init_models(random_seed=42):
    """
    Initialize 11 ML models with default parameters.
    """
    models = {
        'LogisticRegression': LogisticRegression(random_state=random_seed),
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
    # Feature Extraction
    X_train = generate_feature_matrix(train_sequences, use_augmentation=True)
    y_train = np.array(train_labels, dtype=int)
    
    X_val = generate_feature_matrix(val_sequences, use_augmentation=False)
    y_val = np.array(val_labels, dtype=int)
    
    X_test = generate_feature_matrix(test_sequences, use_augmentation=False)
    y_test = np.array(test_labels, dtype=int)
    
    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Model Training
    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n[Fold {fold+1}/{KFOLD}] Training Models...")
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Validation
            y_val_pred = model.predict(X_val)
            if hasattr(model, 'predict_proba'):
                y_val_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_val_prob = y_val_pred
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            
            # Testing
            y_test_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_test_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_test_prob = y_test_pred
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)
            
            fold_results[model_name] = {
                'val': val_metrics,
                'test': test_metrics
            }
            
            # Simple progress log
            # print(f"  > {model_name} Val ACC: {val_metrics['ACC']:.3f} | AUC: {val_metrics['AUC']:.3f}")
        except Exception as e:
            print(f"  > Error training {model_name}: {e}")
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
            
            for model_name in all_results.keys():
                all_results[model_name]['val'].append(fold_res[model_name]['val'])
                all_results[model_name]['test'].append(fold_res[model_name]['test'])

        # --- Summary Output ---
        print("\n" + "="*160)
        # Header with F1 added
        print(f"{'Model':<20} | {'Set':<5} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("="*160)
        
        for model_name, res in all_results.items():
            # 1. Validation Metrics (Average across folds)
            metrics = res['val']
            val_sn_m, val_sn_s = np.mean([m['Sn'] for m in metrics]), np.std([m['Sn'] for m in metrics])
            val_sp_m, val_sp_s = np.mean([m['Sp'] for m in metrics]), np.std([m['Sp'] for m in metrics])
            val_acc_m, val_acc_s = np.mean([m['ACC'] for m in metrics]), np.std([m['ACC'] for m in metrics])
            val_mcc_m, val_mcc_s = np.mean([m['MCC'] for m in metrics]), np.std([m['MCC'] for m in metrics])
            val_f1_m, val_f1_s = np.mean([m['F1'] for m in metrics]), np.std([m['F1'] for m in metrics])
            val_auc_m, val_auc_s = np.mean([m['AUC'] for m in metrics]), np.std([m['AUC'] for m in metrics])

            # Print Val Row
            print(f"{model_name:<20} | {'Val':<5} | "
                  f"{val_sn_m:.3f}±{val_sn_s:.3f}   | {val_sp_m:.3f}±{val_sp_s:.3f}   | "
                  f"{val_acc_m:.3f}±{val_acc_s:.3f}   | {val_mcc_m:.3f}±{val_mcc_s:.3f}   | "
                  f"{val_f1_m:.3f}±{val_f1_s:.3f}   | {val_auc_m:.3f}±{val_auc_s:.3f}")

            # 2. Test Metrics (Average across folds)
            metrics = res['test']
            test_sn_m, test_sn_s = np.mean([m['Sn'] for m in metrics]), np.std([m['Sn'] for m in metrics])
            test_sp_m, test_sp_s = np.mean([m['Sp'] for m in metrics]), np.std([m['Sp'] for m in metrics])
            test_acc_m, test_acc_s = np.mean([m['ACC'] for m in metrics]), np.std([m['ACC'] for m in metrics])
            test_mcc_m, test_mcc_s = np.mean([m['MCC'] for m in metrics]), np.std([m['MCC'] for m in metrics])
            test_f1_m, test_f1_s = np.mean([m['F1'] for m in metrics]), np.std([m['F1'] for m in metrics])
            test_auc_m, test_auc_s = np.mean([m['AUC'] for m in metrics]), np.std([m['AUC'] for m in metrics])

            # Print Test Row
            print(f"{'':<20} | {'Test':<5} | "
                  f"{test_sn_m:.3f}±{test_sn_s:.3f}   | {test_sp_m:.3f}±{test_sp_s:.3f}   | "
                  f"{test_acc_m:.3f}±{test_acc_s:.3f}   | {test_mcc_m:.3f}±{test_mcc_s:.3f}   | "
                  f"{test_f1_m:.3f}±{test_f1_s:.3f}   | {test_auc_m:.3f}±{test_auc_s:.3f}")
            
            print("-" * 160)