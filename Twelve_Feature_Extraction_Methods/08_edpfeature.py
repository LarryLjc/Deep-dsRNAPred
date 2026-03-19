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
# 2. Feature Extraction: EDP (Entropy Density Profile) & ORF/UTR Features
# =============================================================================

class EDPFeatureExtractor:
    """
    Extracts features based on Entropy Density Profile (EDP), ORF, and UTR properties.
    Features include:
    1. Transcript Length
    2. UTR Lengths (5', 3')
    3. UTR Coverages (5', 3')
    4. EDP of Transcript (20 AA dimensions)
    5. EDP of longest ORF (20 AA dimensions)
    """
    def __init__(self):
        self._AA_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
        # IUPAC code map
        self._IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': 'AG', 'Y': 'CT', 'M': 'AC', 'K': 'GT', 'S': 'CG',
                       'W': 'AT', 'H': 'ACT', 'B': 'CGT', 'V': 'ACG', 'D': 'AGT', 'N': 'ACGT'}

    def fit(self, sequences, labels):
        # No training required for EDP calculation
        return self

    # --- Helper Methods from EDPcoder ---

    def _codon2aa(self, codon):
        '''convert codon to aa'''
        if codon == "TTT" or codon == "TTC": return 'F'
        elif codon == 'TTA' or codon == 'TTG' or codon == 'CTT' or codon == 'CTA' or codon == 'CTC' or codon == 'CTG': return 'L'
        elif codon == 'ATT' or codon == 'ATC' or codon == 'ATA': return 'I'
        elif codon == 'ATG': return 'M'
        elif codon == 'GTA' or codon == 'GTC' or codon == 'GTG' or codon == 'GTT': return 'V'
        elif codon == 'GAT' or codon == 'GAC': return 'D'
        elif codon == 'GAA' or codon == 'GAG': return 'E'
        elif codon == 'TCA' or codon == 'TCC' or codon == 'TCG' or codon == 'TCT': return 'S'
        elif codon == 'CCA' or codon == 'CCC' or codon == 'CCG' or codon == 'CCT': return 'P'
        elif codon == 'ACA' or codon == 'ACG' or codon == 'ACT' or codon == 'ACC': return 'T'
        elif codon == 'GCA' or codon == 'GCC' or codon == 'GCG' or codon == 'GCT': return 'A'
        elif codon == 'TAT' or codon == 'TAC': return 'Y'
        elif codon == 'CAT' or codon == 'CAC': return 'H'
        elif codon == 'CAA' or codon == 'CAG': return 'Q'
        elif codon == 'AAT' or codon == 'AAC': return 'N'
        elif codon == 'AAA' or codon == 'AAG': return 'K'
        elif codon == 'TGT' or codon == 'TGC': return 'C'
        elif codon == 'TGG': return 'W'
        elif codon == 'CGA' or codon == 'CGC' or codon == 'CGG' or codon == 'CGT': return 'R'
        elif codon == 'AGT' or codon == 'AGC': return 'S'
        elif codon == 'AGA' or codon == 'AGG': return 'R'
        elif codon == 'GGA' or codon == 'GGC' or codon == 'GGG' or codon == 'GGT': return 'G'
        elif codon == 'TAA' or codon == 'TAG' or codon == 'TGA': return 'J' # Stop
        else: return 'Z' # Ambiguous

    def _iupac_3mer(self, seq):
        '''Return a list of all possible 3mers of the sequence'''
        kmer_list = []
        # Safety check for length
        if len(seq) < 3: return []
        
        try:
            list1 = self._IUPAC.get(seq[0], [seq[0]])
            list2 = self._IUPAC.get(seq[1], [seq[1]])
            list3 = self._IUPAC.get(seq[2], [seq[2]])
            
            for dna1 in list1:
                for dna2 in list2:
                    for dna3 in list3:
                        if self._codon2aa(dna1 + dna2 + dna3) != "J":
                            kmer_list.append(dna1 + dna2 + dna3)
        except:
            return []
        return kmer_list

    def _get_orf_utr(self, seq):
        '''Get ORF and UTR from sequence'''
        STP = {0: [0], 1: [1], 2: [2]}
        AAnum = int(len(seq) / 3)

        for i in range(0, 3):
            for j in range(0, AAnum):
                tmp = seq[(i+3*j):(i+3*(j+1))]
                if tmp == 'TAG' or tmp == 'TAA' or tmp == 'TGA':
                    STP[i].append(i+3*j)

        ORF = {}
        for i in range(0,3):
            for j in range(1, len(STP[i])):
                tmpN = int((STP[i][j] - STP[i][j-1])/3)
                for k in range(0, tmpN):
                    tmpS = seq[ (STP[i][j-1] + 3*k):(STP[i][j-1] + 3*(k+1)) ] 
                    if tmpS == 'ATG':
                        ORF[3*k + STP[i][j-1]] = STP[i][j] + 3
                        break
            
            # Check for ORF at end of sequence
            if STP[i]:
                codonNum = int((len(seq) - STP[i][-1]) / 3)
                for k in range(codonNum):
                    if seq[ (STP[i][-1] + 3*k): (STP[i][-1] + 3*(k+1)) ] == "ATG":
                        ORF[ STP[i][-1] + 3*k ] = len(seq)
                        break

        # longest ORF
        if ORF:
            ORFlen = []
            ORFstart = []
            ORFend = []
            for (k,v) in ORF.items():
                ORFlen.append(v - k)
                ORFstart.append(k)
                ORFend.append(v)
            
            idx = np.argmax(ORFlen)
            ORF_l = seq[ORFstart[idx]:ORFend[idx]]
            UTR5 = seq[0:ORFstart[idx]] if len(seq[0:ORFstart[idx]]) > 0 else ''
            UTR3 = seq[ORFend[idx]:] if len(seq[ORFend[idx]:]) > 0 else ''
            return ORF_l, UTR5, UTR3, ORFstart[idx], ORFend[idx]
        else:
            return '', '', '', 0, 0

    def _get_edp_no_orf(self):
        return [0.0] * 20

    def _get_edp(self, seq):
        '''get EDP of codon'''
        Codon = {aa: 1e-9 for aa in self._AA_list}
        sum_codon = 1e-9 * 20 

        if len(seq) > 3:
            num = int(len(seq) / 3)
            for i in range(0,num):
                codon_seq = seq[i*3:(i+1)*3]
                aa = self._codon2aa(codon_seq)
                
                if aa == "J":
                    continue
                elif aa == "Z":
                    tmp_kmer_list = self._iupac_3mer(codon_seq)
                    if tmp_kmer_list:
                        weight = 1.0 / len(tmp_kmer_list)
                        for tmp_kmer in tmp_kmer_list:
                            mapped_aa = self._codon2aa(tmp_kmer)
                            if mapped_aa in Codon:
                                Codon[mapped_aa] += weight
                        sum_codon += 1.0
                else:
                    if aa in Codon:
                        Codon[aa] += 1.0
                        sum_codon += 1.0

            H = 0.0
            for k in Codon:
                Codon[k] /= sum_codon
                if Codon[k] > 0:
                    Codon[k] = -Codon[k] * np.log2(Codon[k])
                else:
                    Codon[k] = 0.0
                H += Codon[k]

            value = []
            if H == 0:
                return self._get_edp_no_orf()

            for k in self._AA_list: # Preserve order
                value.append(Codon[k] / H)
            
            return value
        else:
            return self._get_edp_no_orf()

    def _calculate_single_seq_features(self, seq):
        # Clean sequence
        seq = str(seq).strip().upper().replace('U', 'T')
        transcript_len = len(seq)

        # 1. Get ORF and UTRs
        ORF, UTR5, UTR3, start, end = self._get_orf_utr(seq)
        
        # 2. UTR Features
        utr5_len = len(UTR5)
        utr3_len = len(UTR3)
        utr5_cov = utr5_len / transcript_len if transcript_len > 0 else 0
        utr3_cov = utr3_len / transcript_len if transcript_len > 0 else 0

        # 3. EDP Features (Transcript)
        if len(seq) < 6:
            edp_transcript = self._get_edp_no_orf()
        else:
            edp_transcript = self._get_edp(seq)

        # 4. EDP Features (ORF)
        if len(ORF) < 6:
            edp_orf = self._get_edp_no_orf()
        else:
            edp_orf = self._get_edp(ORF)

        # Combine all features
        # [Transcript_Len, UTR5_Len, UTR3_Len, UTR5_Cov, UTR3_Cov] + EDP_Transcript(20) + EDP_ORF(20)
        # Total dims: 1 + 1 + 1 + 1 + 1 + 20 + 20 = 45
        features = [transcript_len, utr5_len, utr3_len, utr5_cov, utr3_cov] + edp_transcript + edp_orf
        return features

    def transform(self, sequences):
        feature_matrix = []
        for seq in sequences:
            feats = self._calculate_single_seq_features(seq)
            feature_matrix.append(feats)
        
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
    # 1. Initialize Extractor (Using EDPFeatureExtractor)
    extractor = EDPFeatureExtractor()
    
    # 2. Augment Training Data
    train_seqs_aug = []
    train_labs_aug = []
    for seq, lab in zip(train_sequences, train_labels):
        train_seqs_aug.append(seq)
        train_labs_aug.append(lab)
        aug_seq = augment_rna_sequence(seq)
        train_seqs_aug.append(aug_seq)
        train_labs_aug.append(lab)
        
    # 3. Fit Extractor (Pass-through for EDP)
    extractor.fit(train_seqs_aug, train_labs_aug)
    
    # 4. Transform Data
    # Note: Processing speed depends on ORF finding and iteration over sequences
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
    
    print(f"\n===== Fold {fold+1}/{KFOLD} - Training (EDP Features) =====")
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