import numpy as np
import pandas as pd
import random
import warnings
import os
import math
import itertools
import re
from collections import defaultdict, Counter

# BioPython related (Added for new feature extraction)
from Bio.Seq import Seq
from Bio.SeqUtils import ProtParam

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
# 2. Feature Extraction: Protein Physicochemical Properties
# =============================================================================

class ExtractORF:
    """
    Extract the most probable ORF in a given sequence 
    The most probable ORF is the longest open reading frame found in the sequence
    When having same length, the upstream ORF is selected.
    """
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
        
        # If result is still initial state, return empty or full seq depending on logic
        # Here we return what was found (or empty string if nothing found)
        if self.result[2] > 0:
            orf_seq = self.seq[self.result[1]:self.result[2]]
        else:
            orf_seq = "" 
            
        ORF_integrity = self.result[0]
        ORF_length = self.result[3]
        return ORF_length, ORF_integrity, orf_seq

class ProteinFeatureExtractor:
    """
    Calculates physicochemical properties from the longest predicted ORF:
    1. Instability Index
    2. Isoelectric Point (PI)
    3. Gravy (Grand average of hydropathicity)
    4. Molecular Weight
    5. pI/Mw score
    """
    def __init__(self):
        self.strinfoAmbiguous = re.compile("X|B|Z|J|U", re.I)
        self.ptU = re.compile("U", re.I)

    def fit(self, sequences, labels):
        # Protein properties are static calculations, no training required.
        # Method kept for compatibility with pipeline structure.
        return self

    def _calculate_single_seq_features(self, seq):
        # 1. Preprocess Sequence
        seqRNA = self.ptU.sub("T", str(seq).strip())
        seqRNA = seqRNA.upper()
        
        # 2. Extract Longest ORF
        extractor = ExtractORF(seqRNA)
        CDS_size1, CDS_integrity, seqCDS = extractor.longest_ORF(start=['ATG'], stop=['TAA', 'TAG', 'TGA'])
        
        # Handle case where no ORF is found
        if not seqCDS:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # 3. Translate to Protein
        try:
            seqprot = Seq(seqCDS).translate()
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        # 4. Clean Protein Sequence (Handle stop codons and ambiguous chars)
        seqprot_str = str(seqprot)
        if '*' in seqprot_str:
            # If stop codon is not at the end, truncate
            nPos = seqprot_str.index('*')
            if nPos != len(seqprot_str) - 1:
                seqprot_str = seqprot_str.split('*')[0] + '*'
        
        pep_len = len(seqprot_str.strip("*"))
        newseqprot = self.strinfoAmbiguous.sub("", seqprot_str)
        cleaned_prot = newseqprot.strip("*")
        
        # 5. Calculate Properties using BioPython
        if pep_len > 0 and len(cleaned_prot) > 0:
            try:
                protparam_obj = ProtParam.ProteinAnalysis(cleaned_prot)
                
                Instability_index = protparam_obj.instability_index()
                PI = protparam_obj.isoelectric_point()
                Gravy = protparam_obj.gravy()
                Mw = protparam_obj.molecular_weight()
                
                # pI/Mw calculation as per provided code
                # Avoid division by zero if PI is 0 (though rare for proteins)
                if PI != 0:
                    pI_Mw = np.log10((float(Mw) / PI) + 1)
                else:
                    pI_Mw = 0.0
                    
                return [Instability_index, PI, Gravy, Mw, pI_Mw]
            except Exception:
                # Fallback for errors in ProtParam (e.g., weird characters)
                return [0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

    def transform(self, sequences):
        feature_matrix = []
        for seq in sequences:
            feats = self._calculate_single_seq_features(seq)
            feature_matrix.append(feats)
        
        # Convert to numpy array
        # Shape: (n_samples, 5)
        return np.array(feature_matrix, dtype=np.float32)

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
    # 1. Initialize Extractor per fold (Using ProteinFeatureExtractor now)
    extractor = ProteinFeatureExtractor()
    
    # 2. Augment Training Data
    train_seqs_aug = []
    train_labs_aug = []
    for seq, lab in zip(train_sequences, train_labels):
        train_seqs_aug.append(seq)
        train_labs_aug.append(lab)
        aug_seq = augment_rna_sequence(seq)
        train_seqs_aug.append(aug_seq)
        train_labs_aug.append(lab)
        
    # 3. Fit Extractor (Pass-through for Protein Features, but kept for flow)
    extractor.fit(train_seqs_aug, train_labs_aug)
    
    # 4. Transform Data
    # Note: Processing speed will be slower than K-mer due to ORF finding + translation
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
    
    print(f"\n===== Fold {fold+1}/{KFOLD} - Training (Protein Features) =====")
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