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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
# 🚀 提速核心：引入直方图加速版 Gradient Boosting
from sklearn.ensemble import HistGradientBoostingClassifier

# --- GPU Libraries Check (RAPIDS cuML) ---
try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.naive_bayes import BernoulliNB as cuBNB # cuML 目前缺少稳定的 BernoulliNB，我们将回退到 sklearn
    from cuml.ensemble import RandomForestClassifier as cuRF
    HAS_CUML = True
    print("✅ RAPIDS cuML library detected. Using GPU for classic ML models (Except SVM & NB).")
except ImportError:
    HAS_CUML = False
    print("⚠️ RAPIDS cuML not found. Falling back to CPU with full-core acceleration (n_jobs=-1).")

# 引入普通的 sklearn 模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB  # 👈 针对 0/1 特征，必须使用伯努利朴素贝叶斯
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

# 🌟 更新读取路径为预处理后带有 Structure 列的 Excel 文件
TRAIN_EXCEL_PATH = "/root/autodl-tmp/Visualization/Result/ML/train_RNA_with_struct.xlsx"
TEST_EXCEL_PATH = "/root/autodl-tmp/Visualization/Result/ML/test_RNA_with_struct.xlsx" 
OUTPUT_SAVE_PATH = "Ensemble_Test_Performance_11_ML_StructureSparseEncoding.xlsx"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)
print(f"Global Random Seed set to: {RANDOM_SEED}")

# =============================================================================
# 2. Optimized Feature Extraction: RNA Structure Sparse Encoding (7-Dim)
# =============================================================================
class StructureSparseFeatureExtractor:
    """
    将 bpRNA 提取的 7 种二级结构状态进行 Sparse 编码。
    S: Stem (茎)
    H: Hairpin loop (发夹环)
    B: Bulge (凸起)
    I: Internal loop (内部环)
    M: Multi-loop (多分支环)
    E: External (外部结构)
    X: Unknown (未知/其他)
    """
    def __init__(self, max_length=None):
        self.max_length = max_length
        # 🌟 7位正交向量对应 7 维结构序列
        self.sparse_dict = {
            'S': [1, 0, 0, 0, 0, 0, 0],
            'H': [0, 1, 0, 0, 0, 0, 0],
            'B': [0, 0, 1, 0, 0, 0, 0],
            'I': [0, 0, 0, 1, 0, 0, 0],
            'M': [0, 0, 0, 0, 1, 0, 0],
            'E': [0, 0, 0, 0, 0, 1, 0],
            'X': [0, 0, 0, 0, 0, 0, 1]
        }

    def transform(self, structures):
        processed_structs = []
        lengths = []
        
        # Preprocessing structures
        for s in structures:
            # 统一转为大写，去除空白
            s = str(s).upper().strip()
            if not s or s == 'NAN': s = "X" # 异常情况处理为未知结构
            processed_structs.append(s)
            lengths.append(len(s))
            
        # Determine N (max_length)
        if self.max_length is None:
            self.max_length = max(lengths) if lengths else 0
            
        n_samples = len(structures)
        # 🌟 Dimensions: Samples x Length x 7 (Sparse vector size)
        encoding_array = np.zeros((n_samples, self.max_length, 7), dtype=np.float32)
        
        for idx, struct in enumerate(processed_structs):
            # Truncate if longer than max_length
            loop_len = min(len(struct), self.max_length)
            
            for i in range(loop_len):
                char = struct[i]
                # 遇到字典外字符默认映射为 X [0,0,0,0,0,0,1]
                vec = self.sparse_dict.get(char, [0, 0, 0, 0, 0, 0, 1])
                encoding_array[idx, i, :] = vec
                
        # Flatten for ML models: (n_samples, max_length * 7)
        return encoding_array.reshape(n_samples, -1)

# =============================================================================
# 3. Utilities & Metrics
# =============================================================================
def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    assert "Structure" in df.columns, f"Excel file {file_path} must contain 'Structure' column"
    
    # 🌟 【核心修改】：如果不存在 label 列，自动从 ID 列中拆解提取
    if "label" not in df.columns:
        if "ID" in df.columns:
            # 例如将 "test|1|positive1" 按 "|" 切割，取索引 1 的部分并转换为整数
            df["label"] = df["ID"].apply(lambda x: int(str(x).split('|')[1]))
            print(f"✅ Automatically extracted labels from 'ID' column in {os.path.basename(file_path)}")
        else:
            raise ValueError(f"Excel file {file_path} must contain either 'label' or 'ID' column for ML training.")
        
    df["Structure"] = df["Structure"].astype(str).str.upper().str.strip().replace("", "X")
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
    
    base_svm = LinearSVC(
        random_state=random_seed, 
        class_weight='balanced', 
        loss='hinge',             
        max_iter=2000
    )
    fast_proba_svm = CalibratedClassifierCV(base_svm, method='isotonic', cv=3)
    
    # 1. 经典机器学习
    if HAS_CUML:
        models['LogisticRegression'] = cuLogReg(max_iter=1000)
        models['SVM'] = fast_proba_svm 
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = BernoulliNB() 
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_seed, max_iter=1000, n_jobs=-1)
        models['SVM'] = fast_proba_svm 
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = BernoulliNB() 
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)

    # 2. 纯 CPU 提速树模型
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed)
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)
    models['HistGradientBoost'] = HistGradientBoostingClassifier(random_state=random_seed, max_iter=100)

    # 3. 三大现代 GBDT 框架
    models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, verbose=-1, n_jobs=-1)
    models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, eval_metric='logloss', n_jobs=-1, tree_method='hist')
    
    try:
        models['CatBoost'] = CatBoostClassifier(
            random_state=random_seed, verbose=0, allow_writing_files=False,
            task_type="GPU", devices='0'
        )
    except:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, verbose=0)

    return models

# =============================================================================
# 5. Main Execution (K-Fold + Test Ensemble)
# =============================================================================
if __name__ == "__main__":
    if os.path.exists(TRAIN_EXCEL_PATH) and os.path.exists(TEST_EXCEL_PATH):
        cv_df = read_excel_data(TRAIN_EXCEL_PATH)
        test_df = read_excel_data(TEST_EXCEL_PATH)
        
        print(f"🚀 Data Loaded. Training on: {len(cv_df)} | Testing on: {len(test_df)}")
        
        # 🌟 获取 Structure 列表
        cv_structures = cv_df["Structure"].tolist()
        y_cv = cv_df["label"].values.astype(int)
        test_structures = test_df["Structure"].tolist()
        y_test = test_df["label"].values.astype(int)

        # 确定特征提取最大长度
        all_lens = [len(str(s)) for s in cv_structures + test_structures]
        max_len = max(all_lens) if all_lens else 0
        
        # 实例化更新后的 StructureSparseFeatureExtractor
        extractor = StructureSparseFeatureExtractor(max_length=max_len)
        
        # ---------------------------------------------------------
        # 全局预先提取 7-Dim Sparse Encoding 特征
        # ---------------------------------------------------------
        print(f"Extracting Structure Sparse Encoding Features globally (Max Length: {max_len})...")
        X_cv_raw = extractor.transform(cv_structures)
        X_test_raw = extractor.transform(test_structures)
        print(f"Feature Dimension: {X_cv_raw.shape[1]}")

        kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_SEED)
        
        dummy_models = init_models()
        model_names = list(dummy_models.keys())
        
        val_results = {name: [] for name in model_names}
        test_probs_sum = {name: np.zeros(len(y_test)) for name in model_names}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_raw)):
            print(f"\n--- Training Fold {fold+1}/{KFOLD} ---")
            start_time = time.time()
            
            X_train_fold = X_cv_raw[train_idx]
            y_train_fold = y_cv[train_idx].astype(np.float32)
            
            X_val_fold = X_cv_raw[val_idx]
            y_val_fold = y_cv[val_idx].astype(np.float32)
            
            X_test_fold = X_test_raw.copy()
            
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
                    
                    print(f"[{name:<20}] Val ACC: {val_metrics['ACC']:.4f} | AUC: {val_metrics['AUC']:.4f}")
                    
                    # 测试集预测
                    y_test_prob = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_fold)
                    if hasattr(y_test_prob, 'get'): y_test_prob = y_test_prob.get()
                    test_probs_sum[name] += y_test_prob
                    
                except Exception as e:
                    print(f"[{name}] Failed: {e}")
                    
            print(f"Fold {fold+1} completed in {time.time() - start_time:.2f}s")
            del models
            gc.collect()

        # =====================================================================
        # 6. Ensemble Test Metrics Calculation & Saving
        # =====================================================================
        print("\n" + "="*160)
        print(" 🏆 FINAL ENSEMBLE RESULTS ON TEST SET (11 Machine Learning Models)")
        print("="*160)
        print(f"{'Model':<25} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("-" * 160)
        
        final_output_list = []
        
        for name in model_names:
            if len(val_results[name]) == 0: continue
                
            avg_test_probs = test_probs_sum[name] / KFOLD
            final_test_preds = (avg_test_probs >= 0.5).astype(int)
            test_metrics = calculate_metrics(y_test, final_test_preds, avg_test_probs)
            
            metrics_to_save = {'Model': name}
            metrics_to_save.update(test_metrics)
            final_output_list.append(metrics_to_save)
            
            row = f"{name:<25} | "
            row += " | ".join([f"{test_metrics[k]:.4f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
            print(row)

        print("="*160)

        # ---------------------------------------------------------------------
        # 7. Validation Set Average Performance Output
        # ---------------------------------------------------------------------
        print("\n" + "="*160)
        print(" 📊 AVERAGE VALIDATION METRICS (Across 5 Folds)")
        print("="*160)
        print(f"{'Model':<25} | {'Sn':<15} | {'Sp':<15} | {'ACC':<15} | {'MCC':<15} | {'F1':<15} | {'AUC':<15}")
        print("-" * 160)

        for name, metrics_list in val_results.items():
            if not metrics_list: continue
            avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
            std_metrics = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}
            
            row = f"{name:<25} | "
            row += " | ".join([f"{avg_metrics[k]:.4f}±{std_metrics[k]:.4f}" for k in ['Sn', 'Sp', 'ACC', 'MCC', 'F1', 'AUC']])
            print(row)
        print("="*160)
        
        output_df = pd.DataFrame(final_output_list)
        output_df.to_excel(OUTPUT_SAVE_PATH, index=False)
        print(f"\n✅ All 11 Models Ensemble Test Performance (Structure Features) successfully saved to:\n   {OUTPUT_SAVE_PATH}")

    else:
        print(f"❌ Error: Data files not found. Please check paths:\nTrain: {TRAIN_EXCEL_PATH}\nTest: {TEST_EXCEL_PATH}")