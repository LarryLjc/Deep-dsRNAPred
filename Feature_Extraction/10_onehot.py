import os
import time
import numpy as np
import pandas as pd
import random
import warnings
import gc

# Scikit-learn
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix, f1_score
from sklearn.preprocessing import MaxAbsScaler # 🌟 稀疏矩阵专用缩放
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# --- GPU Libraries Check ---
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_CUML = True
    print("✅ RAPIDS cuML library detected.")
except ImportError:
    HAS_CUML = False
    print("⚠️ RAPIDS cuML not found. Falling back to CPU.")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # 🌟 抢救核心：回归原生带概率的 SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# Modern GBDT libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =============================================================================
# 1. Global Configuration
# =============================================================================
warnings.filterwarnings("ignore")
KFOLD = 5
RANDOM_SEED = 3407

TRAIN_EXCEL_PATH = "/root/autodl-tmp/data/train_combined.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/data/test_combined.xlsx"
OUTPUT_SAVE_PATH = "Ensemble_Test_Performance_11_ML_OneHot_SVM_Fixed.xlsx"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)

# =============================================================================
# 2. One-Hot Feature Extractor
# =============================================================================
class OneHotFeatureExtractor:
    def __init__(self, max_length=None):
        self.alphabet = "ACGT"
        self.char_to_int = {c: i for i, c in enumerate(self.alphabet)}
        self.max_length = max_length

    def transform(self, sequences):
        processed_seqs = [str(s).upper().replace('U', 'T').strip() or "A" for s in sequences]
        if self.max_length is None:
            self.max_length = max([len(s) for s in processed_seqs])
        
        n_samples = len(sequences)
        one_hot = np.zeros((n_samples, self.max_length, 4), dtype=np.float32)
        for idx, seq in enumerate(processed_seqs):
            seq = seq[:self.max_length]
            for i, char in enumerate(seq):
                val = self.char_to_int.get(char, -1)
                if val != -1: one_hot[idx, i, val] = 1.0
        return one_hot.reshape(n_samples, -1)

# =============================================================================
# 3. Utilities & Metrics
# =============================================================================
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    df["Sequence"] = df["Sequence"].astype(str).str.upper().str.strip().replace("", "ATG")
    return df.reset_index(drop=True)

def calculate_metrics(y_true, y_pred, y_prob):
    if hasattr(y_true, 'get'): y_true = y_true.get()
    if hasattr(y_pred, 'get'): y_pred = y_pred.get()
    if hasattr(y_prob, 'get'): y_prob = y_prob.get()
    y_true, y_pred, y_prob = np.nan_to_num(y_true), np.nan_to_num(y_pred), np.nan_to_num(y_prob)
    try: tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except: tn = fp = fn = tp = 0
    sn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    acc = (tp + tn) / len(y_true)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {'Sn': sn, 'Sp': sp, 'ACC': acc, 'MCC': mcc, 'F1': f1, 'AUC': auc}

# =============================================================================
# 4. Model Initialization (SVM Fixed)
# =============================================================================
def init_models(random_seed=3407):
    models = {}
    # 🌟 抢救版 SVM：使用线性核，开启概率预测，防止高维坍塌
    # 增加 class_weight 以平衡序列数据，max_iter 防止过慢
    fixed_svm = SVC(kernel='linear', probability=True, random_state=random_seed, 
                    class_weight='balanced', max_iter=1000)
    
    if HAS_CUML:
        models['LogisticRegression'] = cuLogReg(max_iter=1000)
        models['SVM'] = fixed_svm
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = BernoulliNB()
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(max_iter=1000, n_jobs=-1)
        models['SVM'] = fixed_svm
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = BernoulliNB()
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)
    models['HistGradientBoost'] = HistGradientBoostingClassifier(random_state=random_seed, max_iter=100)
    models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, verbose=-1, n_jobs=-1)
    models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, eval_metric='logloss', n_jobs=-1, tree_method='hist')
    try:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, verbose=0, task_type="GPU", devices='0')
    except:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, verbose=0)
    return models

# =============================================================================
# 5. Main Execution
# =============================================================================
if __name__ == "__main__":
    if os.path.exists(TRAIN_EXCEL_PATH) and os.path.exists(TEST_EXCEL_PATH):
        cv_df, test_df = read_excel_data(TRAIN_EXCEL_PATH), read_excel_data(TEST_EXCEL_PATH)
        y_cv, y_test = cv_df["label"].values.astype(int), test_df["label"].values.astype(int)
        
        max_len = max(cv_df["Sequence"].str.len().max(), test_df["Sequence"].str.len().max())
        extractor = OneHotFeatureExtractor(max_length=max_len)
        X_cv_raw = extractor.transform(cv_df["Sequence"].tolist())
        X_test_raw = extractor.transform(test_df["Sequence"].tolist())

        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        model_names = list(init_models().keys())
        val_results = {name: [] for name in model_names}
        test_probs_sum = {name: np.zeros(len(y_test)) for name in model_names}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
            print(f"\n--- Fold {fold+1}/{KFOLD} ---")
            X_train_fold, y_train_fold = X_cv_raw[train_idx], y_cv[train_idx]
            X_val_fold, y_val_fold = X_cv_raw[val_idx], y_cv[val_idx]

            # 🌟 抢救性缩放：使用 MaxAbsScaler 保持稀疏性
            scaler = MaxAbsScaler()
            X_train_fold = scaler.fit_transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)
            X_test_fold = scaler.transform(X_test_raw)

            models = init_models(RANDOM_SEED)
            for name, model in models.items():
                try:
                    model.fit(X_train_fold, y_train_fold)
                    y_val_prob = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val_fold)
                    y_val_pred = (y_val_prob >= 0.5).astype(int)
                    val_results[name].append(calculate_metrics(y_val_fold, y_val_pred, y_val_prob))
                    
                    y_test_prob = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_fold)
                    if hasattr(y_test_prob, 'get'): y_test_prob = y_test_prob.get()
                    test_probs_sum[name] += y_test_prob
                    print(f"[{name:<20}] Val ACC: {val_results[name][-1]['ACC']:.4f} | AUC: {val_results[name][-1]['AUC']:.4f}")
                except Exception as e: print(f"[{name}] Failed: {e}")
            gc.collect()

        print("\n" + "="*140 + "\n 🏆 FINAL ENSEMBLE RESULTS ON TEST SET\n" + "="*140)
        final_list = []
        for name in model_names:
            if not val_results[name]: continue
            avg_test_probs = test_probs_sum[name] / KFOLD
            test_res = calculate_metrics(y_test, (avg_test_probs >= 0.5).astype(int), avg_test_probs)
            final_list.append({'Model': name, **test_res})
            print(f"{name:<25} | " + " | ".join([f"{test_res[k]:.4f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']]))
        pd.DataFrame(final_list).to_excel(OUTPUT_SAVE_PATH, index=False)