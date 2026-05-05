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
# 🌟 修复贝叶斯核心：引入 PowerTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
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
    # ⚠️ 移除了 from cuml.svm import SVC as cuSVC, 避开底层 Bug
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
# 1. GLOBAL CONFIGURATION & SEED SETUP
# =============================================================================
warnings.filterwarnings("ignore")

KFOLD = 5
RANDOM_SEED = 3407

TRAIN_EXCEL_PATH = "autodl-tmp/Visualization/Result/ML/train_SSfeatures.xlsx"
TEST_EXCEL_PATH = "autodl-tmp/Visualization/Result/ML/test_SSfeatures.xlsx"
OUTPUT_SAVE_PATH = "Ensemble_Test_Performance_11_ML_SSfeatures8.xlsx"

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

setup_seed(RANDOM_SEED)

# =============================================================================
# 2. FEATURE EXTRACTION & DATASET
# =============================================================================
def extract_ss_features(df):
    """提取 Excel 中的 8 个 SSfeatures 数值列"""
    ss_cols = [
        "SLDLD", "SLDPD", "SLDRD", "SLDLN", 
        "SLDPN", "SLDRN", "SDMFE", "SFPUS"
    ]
    
    actual_cols = []
    for col in ss_cols:
        match = [c for c in df.columns if col in c]
        if match:
            actual_cols.append(match[0])
        else:
            raise ValueError(f"列 '{col}' 未在 Excel 中找到，请检查数据列名！")
            
    features = df[actual_cols].values.astype(np.float32)
    return features

def get_labels_from_id(df):
    """根据 ID 列的内容自动生成标签"""
    return np.array([1 if 'positive' in str(val).lower() else 0 for val in df['ID']])

# =============================================================================
# 3. UTILITIES & METRICS
# =============================================================================
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
        # cuML 的 LogReg 原生不支持 balanced，如果数据极度不平衡建议回退到 sklearn
        models['LogisticRegression'] = cuLogReg(max_iter=1000)
        # 🌟 定向修复：使用纯 CPU 版 SVM，并加上 class_weight='balanced'
        models['SVM'] = SVC(probability=True, class_weight='balanced', random_state=random_seed) 
        models['KNN'] = cuKNN(n_neighbors=5)
        models['NaiveBayes'] = cuNB()
        models['RandomForest'] = cuRF(n_estimators=100, random_state=random_seed)
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_seed, class_weight='balanced', max_iter=1000, n_jobs=-1)
        models['SVM'] = SVC(probability=True, class_weight='balanced', random_state=random_seed)
        models['KNN'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = GaussianNB()
        models['RandomForest'] = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_seed, n_jobs=-1)

    # 2. 纯 CPU 提速树模型
    models['DecisionTree'] = DecisionTreeClassifier(random_state=random_seed, class_weight='balanced')
    models['AdaBoost'] = AdaBoostClassifier(random_state=random_seed)
    models['HistGradientBoost'] = HistGradientBoostingClassifier(random_state=random_seed, max_iter=100)

    # 3. 三大现代 GBDT 框架
    try:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, class_weight='balanced', verbose=-1, device='cuda')
    except:
        models['LightGBM'] = lgb.LGBMClassifier(random_state=random_seed, class_weight='balanced', verbose=-1, n_jobs=-1)
    
    try:
        models['CatBoost'] = CatBoostClassifier(
            random_state=random_seed, verbose=0, allow_writing_files=False,
            auto_class_weights='Balanced', task_type="GPU", devices='0'
        )
    except:
        models['CatBoost'] = CatBoostClassifier(random_state=random_seed, auto_class_weights='Balanced', verbose=0)
    
    try:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=random_seed, eval_metric='logloss',
            device='cuda', tree_method='hist'
        )
    except:
        models['XGBoost'] = xgb.XGBClassifier(random_state=random_seed, eval_metric='logloss', n_jobs=-1)

    return models

# =============================================================================
# 5. MAIN EXECUTION (K-Fold + Test Ensemble)
# =============================================================================
if __name__ == "__main__":
    if os.path.exists(TRAIN_EXCEL_PATH) and os.path.exists(TEST_EXCEL_PATH):
        print(f"Loading extracted SSfeatures datasets...")
        
        train_df = pd.read_excel(TRAIN_EXCEL_PATH).dropna().reset_index(drop=True)
        test_df = pd.read_excel(TEST_EXCEL_PATH).dropna().reset_index(drop=True)
        
        # 全局预先提取 8 维特征 (由于是直接读取 Excel 中的数值特征，无任何泄漏风险)
        X_cv_raw = extract_ss_features(train_df)
        X_test_raw = extract_ss_features(test_df)
        
        y_cv = get_labels_from_id(train_df)
        y_test = get_labels_from_id(test_df)
        
        print(f"Train samples: {len(y_cv)} (0: {sum(y_cv==0)}, 1: {sum(y_cv==1)})")
        print(f"Test samples: {len(y_test)} (0: {sum(y_test==0)}, 1: {sum(y_test==1)})\n")
        
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
            
            # 🌟 修复贝叶斯核心：使用 PowerTransformer (Yeo-Johnson) 强制正态化
            # 替代原来的 StandardScaler，它能消除偏态，完美契合 GaussianNB 的底层假设
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
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
                    
                    print(f"[{name:<20}] Val ACC: {val_metrics['ACC']:.4f} | AUC: {val_metrics['AUC']:.4f}")
                    
                    # 测试集预测 (累加至 Soft Voting 字典)
                    y_test_prob = model.predict_proba(X_test_fold)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test_fold)
                    if hasattr(y_test_prob, 'get'): y_test_prob = y_test_prob.get() # GPU 张量转 CPU numpy
                    test_probs_sum[name] += y_test_prob
                    
                except Exception as e:
                    print(f"[{name}] Fold {fold+1} Failed: {e}")
                    
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
        
        # Save to Excel
        output_df = pd.DataFrame(final_output_list)
        output_df.to_excel(OUTPUT_SAVE_PATH, index=False)
        print(f"\n✅ All 11 Models Ensemble Test Performance successfully saved to:\n   {OUTPUT_SAVE_PATH}")

    else:
        print("❌ Error: Data files not found.")