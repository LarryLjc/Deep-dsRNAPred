import os
import time
import numpy as np
import pandas as pd
import random
import warnings
import gc
import math
import itertools
from collections import defaultdict, Counter

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# 🚀 提速核心：引入直方图加速版 Gradient Boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# --- GPU Libraries Check (RAPIDS cuML) ---
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.naive_bayes import GaussianNB as cuNB
    from cuml.ensemble import RandomForestClassifier as cuRF
    # ⚠️ 移除了 from cuml.svm import SVC as cuSVC，避开底层 Bug
    HAS_CUML = True
    print("✅ RAPIDS cuML library detected. Using GPU for classic ML models (Except SVM).")
except ImportError:
    HAS_CUML = False
    print("⚠️ RAPIDS cuML not found. Falling back to CPU with full-core acceleration (n_jobs=-1).")

# 引入普通的 sklearn 模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # 👈 核心：无论有没有 GPU，SVM 都强制用纯 CPU 版本
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Modern GBDT libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =============================================================================
# 1. Global Configuration & Environment Setup
# =============================================================================
warnings.filterwarnings("ignore")

KFOLD = 5
RANDOM_SEED = 3407

TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_combined.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
OUTPUT_SAVE_PATH = "Ensemble_Test_Performance_11_ML_EDP.xlsx"

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

        # Combine all features (Total dims: 1 + 1 + 1 + 1 + 1 + 20 + 20 = 45)
        features = [transcript_len, utr5_len, utr3_len, utr5_cov, utr3_cov] + edp_transcript + edp_orf
        return features

    def transform(self, sequences):
        feature_matrix = []
        for seq in sequences:
            feats = self._calculate_single_seq_features(seq)
            feature_matrix.append(feats)
        
        return np.array(feature_matrix, dtype=np.float32)

def generate_feature_matrix(sequences):
    """
    Generates EDP & ORF feature matrix (No augmentation).
    """
    extractor = EDPFeatureExtractor()
    return extractor.transform(sequences)

# =============================================================================
# 3. Utilities & Metrics
# =============================================================================
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    assert "Sequence" in df.columns, "Excel file must contain 'Sequence' column"
    assert "label" in df.columns, "Excel file must contain 'label' column"
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip()
    df["Sequence"] = df["Sequence"].replace("", "ATG")
    return df.reset_index(drop=True)

def calculate_metrics(y_true, y_pred, y_prob):
    # 处理 GPU 数据类型转换 (cuML)
    if hasattr(y_true, 'get'): y_true = y_true.get()
    if hasattr(y_pred, 'get'): y_pred = y_pred.get()
    if hasattr(y_prob, 'get'): y_prob = y_prob.get()

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
# 4. MODEL INITIALIZATION (11 Accelerated ML Algorithms)
# =============================================================================
def init_models(random_seed=3407):
    models = {}
    
    # 1. 经典机器学习
    if HAS_CUML:
        models['LogisticRegression'] = cuLogReg(max_iter=1000)
        # 🌟 定向修复：强制使用 Sklearn 的纯 CPU 版 SVM
        models['SVM'] = SVC(probability=True, random_state=random_seed) 
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = cuNB()
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_seed, max_iter=1000, n_jobs=-1)
        models['SVM'] = SVC(probability=True, random_state=random_seed)
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = GaussianNB()
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)

    # 2. 纯 CPU 提速树模型
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)
    models['HistGradientBoost'] = HistGradientBoostingClassifier(random_state=random_seed, max_iter=100)

    # 3. 三大现代 GBDT 框架
    try:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, verbose=-1, device='cuda')
    except:
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
            random_state=random_seed, eval_metric='logloss',
            device='cuda', tree_method='hist'
        )
    except:
        models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, eval_metric='logloss', n_jobs=-1)

    return models

# =============================================================================
# 5. Main Execution (K-Fold + Test Ensemble)
# =============================================================================
if __name__ == "__main__":
    if os.path.exists(TRAIN_EXCEL_PATH) and os.path.exists(TEST_EXCEL_PATH):
        cv_df = read_excel_data(TRAIN_EXCEL_PATH)
        test_df = read_excel_data(TEST_EXCEL_PATH)
        
        print(f"🚀 Data Loaded. Training on: {len(cv_df)} | Testing on: {len(test_df)}")
        
        cv_sequences = cv_df["Sequence"].tolist()
        y_cv = cv_df["label"].values
        test_sequences = test_df["Sequence"].tolist()
        y_test = test_df["label"].values

        # ---------------------------------------------------------
        # 全局预先提取 EDP & ORF 特征 (无数据泄露风险，全局提取大幅省时)
        # ---------------------------------------------------------
        print("Extracting EDP & ORF Features...")
        X_cv_raw = generate_feature_matrix(cv_sequences)
        X_test_raw = generate_feature_matrix(test_sequences)
        print("Feature extraction complete.")

        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        
        dummy_models = init_models()
        model_names = list(dummy_models.keys())
        
        # 记录验证集结果与测试集概率累计
        val_results = {name: [] for name in model_names}
        test_probs_sum = {name: np.zeros(len(y_test)) for name in model_names}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
            print(f"\n--- Training Fold {fold+1}/{KFOLD} ---")
            start_time = time.time()
            
            # 1. 提取当前折的特征与标签
            X_train_fold = X_cv_raw[train_idx]
            y_train_fold = y_cv[train_idx].astype(np.float32)
            
            X_val_fold = X_cv_raw[val_idx]
            y_val_fold = y_cv[val_idx].astype(np.float32)
            
            # 2. 特征标准化 (必须在折内重新 Fit 防止数据泄露，并转换为 float32 匹配 cuML)
            scaler = StandardScaler()
            X_train_fold = scaler.fit_transform(X_train_fold).astype(np.float32)
            X_val_fold = scaler.transform(X_val_fold).astype(np.float32)
            X_test_fold = scaler.transform(X_test_raw).astype(np.float32)
            
            # 3. 初始化模型与训练
            models = init_models(RANDOM_SEED)
            
            for name, model in models.items():
                try:
                    # 训练
                    model.fit(X_train_fold, y_train_fold)
                    
                    # 验证集评估
                    y_val_pred = model.predict(X_val_fold)
                    y_val_prob = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
                    val_metrics = calculate_metrics(y_val_fold, y_val_pred, y_val_prob)
                    val_results[name].append(val_metrics)
                    
                    print(f"[{name:<18}] Val ACC: {val_metrics['ACC']:.4f} | AUC: {val_metrics['AUC']:.4f}")
                    
                    # 测试集预测 (累加至 Soft Voting 字典)
                    y_test_prob = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_fold)
                    if hasattr(y_test_prob, 'get'): y_test_prob = y_test_prob.get() # GPU 张量转 CPU numpy
                    test_probs_sum[name] += y_test_prob
                    
                except Exception as e:
                    print(f"[{name}] Failed: {e}")
                    
            print(f"Fold {fold+1} completed in {time.time() - start_time:.2f}s")
            del models, scaler
            gc.collect()

        # =====================================================================
        # 6. Ensemble Test Metrics Calculation & Saving
        # =====================================================================
        print("\n" + "="*160)
        print(" 🏆 FINAL ENSEMBLE RESULTS ON TEST SET (11 Machine Learning Models)")
        print("="*160)
        print(f"{'Model (11 ML Algorithms)':<25} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("-" * 160)
        
        final_output_list = []
        
        for name in model_names:
            if len(val_results[name]) == 0: continue
                
            # 1. 计算 5折平均预测概率
            avg_test_probs = test_probs_sum[name] / KFOLD
            
            # 2. 以 0.5 作为阈值计算硬分类预测
            final_test_preds = (avg_test_probs >= 0.5).astype(int)
            
            # 3. 计算 Test Metrics
            test_metrics = calculate_metrics(y_test, final_test_preds, avg_test_probs)
            
            metrics_to_save = {'Model': name}
            metrics_to_save.update(test_metrics)
            final_output_list.append(metrics_to_save)
            
            row = f"{name:<25} | "
            row += " | ".join([f"{test_metrics[k]:.4f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
            print(row)

        print("="*160)
        
        output_df = pd.DataFrame(final_output_list)
        output_df.to_excel(OUTPUT_SAVE_PATH, index=False)
        print(f"\n✅ All 11 Models Ensemble Test Performance successfully saved to:\n   {OUTPUT_SAVE_PATH}")

    else:
        print("❌ Error: Data files not found.")