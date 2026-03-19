import os
import sys
import time
import re
import numpy as np
import pandas as pd
import random
import warnings
from collections import defaultdict
from pandas import DataFrame

# BioPython for the new feature extraction
try:
    from Bio import SeqIO
except ImportError:
    print("⚠️ BioPython not found. Please install it using: pip install biopython")
    sys.exit(1)

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
# 2. Optimized Feature Extraction (CTD Implementation)
# =============================================================================

class CTDcoder:
    """Generates CTD counts for sequences"""
    def __init__(self, infasta, ACGT_encode):
        self.infasta = infasta
        self.ACGT_encode = ACGT_encode

    def CTD(self, fas, encode='0101'):
        seq = str(fas.seq)
        # Ensure sequence is upper case and T/U are consistent before encoding
        seq = seq.upper().replace('U', 'T')
        seq = seq.replace('A', encode[0]).replace('C', encode[1]).replace('G', encode[2]).replace('T', encode[3])

        num_X = seq.count('0')
        num_Y = seq.count('1')
        XY_trans = seq.count('01') + seq.count('10')

        n = len(seq) - 1
        if n < 1: return [0.0] * 13 # Safety for empty/single base seqs

        X_dis = [i.start() for i in re.finditer('0', seq)]
        if not X_dis: X_dis.append(-1)
        
        X0_dis = (X_dis[0] + 1) / n
        X1_dis = (X_dis[round(num_X / 4.0) - 1] + 1) / n
        X2_dis = (X_dis[round(num_X / 4.0 * 2) - 1] + 1) / n
        X3_dis = (X_dis[round(num_X / 4.0 * 3) - 1] + 1) / n
        X4_dis = (X_dis[-1] + 1) / n

        Y_dis = [i.start() for i in re.finditer('1', seq)]
        if not Y_dis: Y_dis.append(-1)
        
        Y0_dis = (Y_dis[0] + 1) / n
        Y1_dis = (Y_dis[round(num_Y / 4.0) - 1] + 1) / n
        Y2_dis = (Y_dis[round(num_Y / 4.0 * 2) - 1] + 1) / n
        Y3_dis = (Y_dis[round(num_Y / 4.0 * 3) - 1] + 1) / n
        Y4_dis = (Y_dis[-1] + 1) / n

        return (list(map(float, [num_X / n, num_Y / n, XY_trans / (n - 1),
                               X0_dis, X1_dis, X2_dis, X3_dis, X4_dis,
                               Y0_dis, Y1_dis, Y2_dis, Y3_dis, Y4_dis])))

    def get_ctd(self):
        dictname = {
            '0010': 'Strong H-Bond donors',
            '0110': 'Linear free energy',
            '0101': 'Molar refractivity',
            '1000': 'Lipoaffinity index',
            '0100': 'Gas-hexadecane PC', 
            '0011': 'NH- count',
            '0001': 'Primary or secondary nitrogens'}
        dictname_sh = {'0010': 'SHD', '0110': 'MLF',
                    '0101': 'MRe', 
                    '1000': 'LFI', '0100': 'HPC',
                    '0011': 'CNH',
                '0001': 'PSN'}

        SEQ_sh = ['CA','CB','AB','A0','A1','A2','A3','A4','B0','B1','B2','B3','B4']
        SEQ = {'CA': ' composition of A',
                'CB': ' composition of B',
                'AB': ' transition between A and B',
                'A0': ' distribution of 0.00A',
                'A1': ' distribution of 0.25A',
                'A2': ' distribution of 0.50A',
                'A3': ' distribution of 0.75A',
                'A4': ' distribution of 1.00A',
                'B0': ' distribution of 0.00B',
                'B1': ' distribution of 0.25B',
                'B2': ' distribution of 0.50B',
                'B3': ' distribution of 0.75B',
                'B4': ' distribution of 1.00B'}
        
        feaname = [dictname_sh[encode]+ j + ": " + dictname[encode] + SEQ[j] for encode in self.ACGT_encode for j in SEQ_sh]
        seqname = []
        feature = []
        
        # Modified to handle both file path (str) and list of SeqRecords
        iterator = SeqIO.parse(self.infasta, 'fasta') if isinstance(self.infasta, str) else self.infasta
        
        for fas in iterator:
            seqid = fas.id
            seqname.append(seqid)
            for encode in self.ACGT_encode:
                feature += self.CTD(fas=fas, encode=encode)

        data = np.array(feature).reshape((len(seqname), 13 * len(self.ACGT_encode)))
        df = DataFrame(data=data, index=seqname, columns=feaname)
        return df

class CTDcoder_3class:
    """Generates CTD counts for 3-class encoded properties"""
    def __init__(self, infasta, ACGT_encode):
        self.infasta = infasta
        self.ACGT_encode = ACGT_encode

    def CTD(self, fas, encode='0021'):
        seq = str(fas.seq)
        seq = seq.upper().replace('U', 'T')
        seq = seq.replace('A', encode[0]).replace('C', encode[1]).replace('G', encode[2]).replace('T', encode[3])

        num_X = seq.count('0')
        num_Y = seq.count('1')
        num_Z = seq.count('2')
        XY_trans = seq.count('01') + seq.count('10')
        XZ_trans = seq.count('02') + seq.count('20')
        YZ_trans = seq.count('12') + seq.count('21')

        n = len(seq) - 1
        if n < 1: return [0.0] * 21 # Safety

        X_dis = [i.start() for i in re.finditer('0', seq)]
        if not X_dis: X_dis.append(-1)
        X0_dis = (X_dis[0] + 1) / n
        X1_dis = (X_dis[round(num_X / 4.0) - 1] + 1) / n
        X2_dis = (X_dis[round(num_X / 4.0 * 2) - 1] + 1) / n
        X3_dis = (X_dis[round(num_X / 4.0 * 3) - 1] + 1) / n
        X4_dis = (X_dis[-1] + 1) / n

        Y_dis = [i.start() for i in re.finditer('1', seq)]
        if not Y_dis: Y_dis.append(-1)
        Y0_dis = (Y_dis[0] + 1) / n
        Y1_dis = (Y_dis[round(num_Y / 4.0) - 1] + 1) / n
        Y2_dis = (Y_dis[round(num_Y / 4.0 * 2) - 1] + 1) / n
        Y3_dis = (Y_dis[round(num_Y / 4.0 * 3) - 1] + 1) / n
        Y4_dis = (Y_dis[-1] + 1) / n

        Z_dis = [i.start() for i in re.finditer('2', seq)]
        if not Z_dis: Z_dis.append(-1)
        Z0_dis = (Z_dis[0] + 1) / n
        Z1_dis = (Z_dis[round(num_Z / 4.0) - 1] + 1) / n
        Z2_dis = (Z_dis[round(num_Z / 4.0 * 2) - 1] + 1) / n
        Z3_dis = (Z_dis[round(num_Z / 4.0 * 3) - 1] + 1) / n
        Z4_dis = (Z_dis[-1] + 1) / n

        return (list(map(float, [num_X / n, num_Y / n, num_Z / n, 
                               XY_trans / (n - 1),XZ_trans / (n - 1),YZ_trans / (n - 1),
                               X0_dis, X1_dis, X2_dis, X3_dis, X4_dis,
                               Y0_dis, Y1_dis, Y2_dis, Y3_dis, Y4_dis,
                               Z0_dis, Z1_dis, Z2_dis, Z3_dis, Z4_dis])))

    def get_ctd(self):
        dictname = {'1020': 'Lipoaffinity index_3','0102': 'Gas-hexadecane PC_3', '1200':'Strong H-Bond acceptors_3','0120':'Potential Hydrogen Bonds_3','1002':'Sum of path lengths starting from oxygens_3','0021':'Topological polar surface area_3'}
        dictname_sh = {'1020': 'LFI','0102': 'HPC','1200':'SHA','0120':'PHB','1002':'SLF','0021':'TPS'}

        SEQ_sh = ['CA','CB','CC','AB','AC','BC','A0','A1','A2','A3','A4','B0','B1','B2','B3','B4','C0','C1','C2','C3','C4']
        SEQ = {'CA': ' composition of A', 'CB': ' composition of B', 'CC': ' composition of C',
                'AB': ' transition between A and B', 'AC': ' transition between A and C', 'BC': ' transition between B and C',
                'A0': ' distribution of 0.00A', 'A1': ' distribution of 0.25A', 'A2': ' distribution of 0.50A', 'A3': ' distribution of 0.75A', 'A4': ' distribution of 1.00A',
                'B0': ' distribution of 0.00B', 'B1': ' distribution of 0.25B', 'B2': ' distribution of 0.50B', 'B3': ' distribution of 0.75B', 'B4': ' distribution of 1.00B',
                'C0': ' distriCution of 0.00C', 'C1': ' distriCution of 0.25C', 'C2': ' distriCution of 0.50C', 'C3': ' distriCution of 0.75C', 'C4': ' distriCution of 1.00C'}
        
        feaname = [dictname_sh[encode]+ j + ": " + dictname[encode] + SEQ[j] for encode in self.ACGT_encode for j in SEQ_sh]
        seqname = []
        feature = []
        
        # Modified to handle both file path (str) and list of SeqRecords
        iterator = SeqIO.parse(self.infasta, 'fasta') if isinstance(self.infasta, str) else self.infasta
        
        for fas in iterator:
            seqid = fas.id
            seqname.append(seqid)
            for encode in self.ACGT_encode:
                feature += self.CTD(fas=fas, encode=encode)

        data = np.array(feature).reshape((len(seqname), 21 * len(self.ACGT_encode)))
        df = DataFrame(data=data, index=seqname, columns=feaname)
        return df

class CombinedCTDExtractor:
    """Wrapper to integrate CTD extraction into the existing ML pipeline"""
    def __init__(self):
        # Encodings found in the provided classes
        self.encode_2class = ['0010', '0110', '0101', '1000', '0100', '0011', '0001']
        self.encode_3class = ['1020', '0102', '1200', '0120', '1002', '0021']

    def transform(self, sequences):
        # 1. Convert simple string list to objects mimicking SeqRecords (for compatibility with CTD classes)
        class SimpleRecord:
            def __init__(self, s, i):
                self.seq = s
                self.id = str(i)
        
        records = [SimpleRecord(s, i) for i, s in enumerate(sequences)]
        
        # 2. Extract 2-class CTD features
        coder2 = CTDcoder(records, self.ACGT_encode=self.encode_2class)
        df2 = coder2.get_ctd()
        
        # 3. Extract 3-class CTD features
        coder3 = CTDcoder_3class(records, self.ACGT_encode=self.encode_3class)
        df3 = coder3.get_ctd()
        
        # 4. Concatenate features
        # Reset index to ensure safe concat (though order is preserved)
        df2 = df2.reset_index(drop=True)
        df3 = df3.reset_index(drop=True)
        combined_df = pd.concat([df2, df3], axis=1)
        
        # 5. Handle potential NaNs (e.g., division by zero for very short sequences) and return numpy array
        return combined_df.fillna(0).values.astype(np.float32)

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
    
    # --- CHANGED: Using CombinedCTDExtractor ---
    extractor = CombinedCTDExtractor()
    # -------------------------------------------
    
    train_seqs_aug = []
    train_labs_aug = []
    for seq, lab in zip(train_seqs, train_labs):
        train_seqs_aug.append(seq)
        train_labs_aug.append(lab)
        train_seqs_aug.append(augment_rna_sequence(seq))
        train_labs_aug.append(lab)
    
    print("Extracting features for Training set...")
    X_train = extractor.transform(train_seqs_aug)
    y_train = np.array(train_labs_aug, dtype=np.float32)
    
    print("Extracting features for Validation set...")
    X_val = extractor.transform(val_seqs)
    y_val = np.array(val_labs, dtype=np.float32)
    
    print("Extracting features for Test set...")
    X_test = extractor.transform(test_seqs)
    y_test = np.array(test_labs, dtype=np.float32)

    # Standardize Features (CTD values are floats, scaling helps SVM/LogReg)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    models = init_models(RANDOM_SEED)
    fold_results = {}
    
    print(f"\n--- Fold {fold+1}/{KFOLD} | Features: {X_train.shape[1]} ---")
    
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

        # --- Report Summary (Deep-dsRNAPred with CTD Features) ---
        print("\n" + "="*160)
        print(f"{'Model (CTD-dsRNAPred)':<22} | {'Set':<5} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
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