# ==============================================================================
# SCRIPT PELATIHAN & SERIALISASI MODEL BUNDLE PRODUKSI (WELLUM AI ENGINE)
# Menghasilkan models/wellum_model_bundle.joblib untuk Serving Dashboard
# ==============================================================================
import os, sys, warnings, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import f1_score

import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

os.makedirs('models', exist_ok=True)

print("[INFO] Membaca train.csv dan test.csv...")
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

TARGET_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
INV_TARGET_MAP = {0: 'Low', 1: 'Medium', 2: 'High'}
y = train_raw['digital_academic_wellbeing_level'].map(TARGET_MAP).values

# ------------------------------------------------------------------------------
# 1. FEATURE PIPELINE GENERATOR DENGAN RECORDING PARAMETER
# ------------------------------------------------------------------------------
print("[INFO] Membangun pipeline fitur & mengekstrak metadata...")

def extract_features(df_train, df_test=None):
    df_tr = df_train.copy()
    dfs = [df_tr]
    if df_test is not None:
        df_te = df_test.copy()
        dfs.append(df_te)

    # 1. Sanitasi format dasar
    for df in dfs:
        df['age'] = df['age'].astype(str).str.extract(r'(\d+)').astype(float)
        df['gender'] = df['gender'].astype(str).str.strip().str.capitalize()
        df['notifications_isna'] = (df['notifications_per_day'] < 0).astype(int)
        df['notifications_clean'] = df['notifications_per_day'].replace(-5, np.nan)
        df['is_extreme_screentime'] = (df['screen_time_hrs_per_day'] > 24).astype(int)
        df['screen_time_clipped'] = df['screen_time_hrs_per_day'].clip(upper=24)

        missing_cols = ['caffeine_intake_mg_per_day', 'social_media_hrs_per_day',
                        'uses_productivity_app', 'favorite_content_type',
                        'study_hours_per_week', 'class_attendance_percent']
        for col in missing_cols:
            df[f'{col}_isna'] = df[col].isnull().astype(int)

        df['n_missing_total'] = df[missing_cols].isnull().sum(axis=1)
        df['productivity_app_num'] = df['uses_productivity_app'].map({'Yes': 1, 'No': 0}).fillna(-1)
        df['frequently_procrastinates_num'] = df['frequently_procrastinates'].map({'Yes': 1, 'No': 0})
        df['uses_productivity_app_clean'] = df['uses_productivity_app'].fillna('Missing')
        df['favorite_content_type_clean'] = df['favorite_content_type'].fillna('Missing')

    # Imputasi Median
    medians = {}
    for col in ['notifications_clean', 'caffeine_intake_mg_per_day', 'social_media_hrs_per_day',
                'study_hours_per_week', 'class_attendance_percent']:
        medians[col] = float(df_tr[col].median())
    
    for df in dfs:
        for col, med in medians.items():
            df[col] = df[col].fillna(med)

    # 2. Rekayasa Fitur Domain
    for df in dfs:
        sleep = df['sleep_hours']
        gpa = df['gpa']
        screen = df['screen_time_clipped']
        soc = df['social_media_hrs_per_day']
        att = df['class_attendance_percent']
        caff = df['caffeine_intake_mg_per_day']
        study = df['study_hours_per_week']
        proc = df['procrastination_level_1_10']
        stress = df['academic_stress_level_1_10']
        anx = df['pre_exam_anxiety_level_1_10']
        bed_scr = df['screen_time_before_bed_minutes']
        notif = df['notifications_clean']
        diet = df['diet_quality_1_10']
        phys = df['physical_activity_hrs_per_week']
        late = df['late_assignments_per_semester']
        hr = df['resting_heart_rate_bpm']
        prod_app = df['productivity_app_num']

        # Ratios
        pos = (sleep / 7.5) * (gpa / 2.5) * (att / 80.0)
        neg = (proc / 5.0) * (bed_scr / 30.0 + 0.1) * (soc / 4.0 + 0.1) * (stress / 5.0)
        df['wellbeing_force_ratio'] = pos / (neg + 0.05)
        df['log_wellbeing_ratio'] = np.log1p(df['wellbeing_force_ratio'])
        df['academic_health'] = gpa * (att / 100.0)
        df['academic_vitality'] = (gpa / 4.0) * (att / 100.0) * (sleep / 8.0)
        df['academic_strain'] = (stress + anx + proc) / (gpa + 0.1)

        # Chronobiology
        df['sleep_deficit'] = sleep - 7.5
        df['abs_sleep_deficit'] = np.abs(sleep - 7.5)
        df['sleep_x_gpa'] = sleep * gpa
        df['sleep_minus_proc'] = sleep - proc
        df['sleep_minus_stress'] = sleep - stress
        df['sleep_efficiency'] = sleep / (sleep + (bed_scr / 60.0) + 0.1)
        df['sleep_to_screen_bed'] = sleep * (100.0 / (bed_scr + 1.0))
        df['bedtime_procrastination_index'] = (bed_scr / 30.0) * (proc / 5.0)
        df['circadian_disruption'] = (bed_scr / 60.0) / (sleep + 0.1) * (caff / 100.0 + 1.0)
        df['stress_optimal_diff'] = np.abs(stress - 5.5)

        # Digital Burden
        df['digital_burden'] = (soc + (bed_scr / 60.0)) / (sleep + 0.1)
        df['screen_to_sleep_ratio'] = screen / (sleep + 0.1)
        df['social_media_share'] = soc / (screen + 0.1)
        df['digital_distraction_index'] = (notif / 50.0) * (bed_scr / 30.0)
        df['total_digital_hours'] = screen + soc
        df['digital_intensity'] = (screen + soc + (bed_scr / 60.0)) / (sleep + study + 0.1)

        # Stress & Lifestyle
        df['composite_stress'] = (stress + anx + proc) / 3.0
        df['stress_anxiety_product'] = stress * anx
        df['triple_psych_burden'] = stress * anx * proc
        df['study_efficiency'] = gpa / (study + 1.0)
        df['health_index'] = (diet + phys + sleep) / 3.0
        df['health_vs_stress'] = (diet + phys + sleep) / (stress + anx + 0.1)
        df['caffeine_to_sleep'] = caff / (sleep + 0.1)
        df['heart_rate_stress_index'] = hr * stress / 10.0

        # Polynomials
        df['proc_squared'] = proc ** 2
        df['anxiety_squared'] = anx ** 2
        df['stress_squared'] = stress ** 2
        df['sleep_squared'] = sleep ** 2
        df['gpa_squared'] = gpa ** 2

        # App Cross
        df['prod_app_x_proc'] = prod_app * proc
        df['prod_app_x_gpa'] = prod_app * gpa
        df['prod_app_x_sleep'] = prod_app * sleep
        df['prod_app_x_stress'] = prod_app * stress

        # SOTA Targeted
        df['vitality_to_strain_ratio'] = df['academic_vitality'] / (df['academic_strain'] + 0.05)
        df['log_vitality_strain_ratio'] = np.log1p(np.maximum(0, df['vitality_to_strain_ratio']))
        df['procrastination_to_wellbeing'] = proc / (df['wellbeing_force_ratio'] + 0.05)
        df['net_sleep_surplus'] = sleep - 7.0 - (bed_scr / 60.0)
        soc_weekly = soc * 7.0
        df['active_study_ratio'] = study / (study + soc_weekly + 0.1)
        df['attendance_resilience'] = (att / 100.0) * (1.0 - stress / 11.0)
        df['cardio_stress_load'] = (hr / 70.0) * ((stress + anx) / 10.0)

        for col_name in ['sleep_hours', 'gpa', 'academic_stress_level_1_10', 'wellbeing_force_ratio']:
            df[f'{col_name}_rank'] = df[col_name].rank(pct=True)

    # Frequency Encoding
    freq_maps = {}
    for col in ['gender', 'field_of_study', 'favorite_content_type_clean', 'uses_productivity_app_clean']:
        f_map = df_tr[col].value_counts(normalize=True).to_dict()
        freq_maps[col] = f_map
        for df in dfs:
            df[f'{col}_freq'] = df[col].map(f_map).fillna(0)

    drop_cols = [
        'notifications_per_day', 'uses_productivity_app', 'favorite_content_type',
        'frequently_procrastinates', 'family_academic_achievement_history',
        'takes_extra_tutoring', 'major_life_event_this_semester', 'screen_time_hrs_per_day',
    ]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns=drop_cols, errors='ignore')

    # Group Aggregations
    target_num_cols = ['gpa', 'academic_stress_level_1_10', 'study_hours_per_week',
                       'sleep_hours', 'screen_time_clipped', 'procrastination_level_1_10']
    group_stats_dict = {}

    for grp in ['field_of_study', 'uses_productivity_app_clean', 'gender']:
        group_stats_dict[grp] = {}
        for num in target_num_cols:
            grp_stats = dfs[0].groupby(grp)[num].agg(['mean', 'std']).reset_index()
            group_stats_dict[grp][num] = grp_stats
            for i in range(len(dfs)):
                dfs[i] = dfs[i].merge(grp_stats, on=grp, how='left')
                dfs[i][f'{num}_by_{grp}_mean'] = dfs[i]['mean']
                dfs[i][f'{num}_by_{grp}_std'] = dfs[i]['std']
                dfs[i] = dfs[i].drop(columns=['mean', 'std'])
                dfs[i][f'{num}_diff_{grp}'] = dfs[i][num] - dfs[i][f'{num}_by_{grp}_mean']
                dfs[i][f'{num}_z_{grp}'] = dfs[i][f'{num}_diff_{grp}'] / (dfs[i][f'{num}_by_{grp}_std'] + 1e-5)

    cat_cols = ['gender', 'field_of_study', 'uses_productivity_app_clean', 'favorite_content_type_clean']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        comb = pd.concat([dfs[0][col].astype(str)] + ([dfs[1][col].astype(str)] if len(dfs)>1 else []))
        le.fit(comb)
        label_encoders[col] = le
        for i in range(len(dfs)):
            dfs[i][col] = le.transform(dfs[i][col].astype(str))

    feature_cols = [c for c in dfs[0].columns if c not in ['respondent_id', 'digital_academic_wellbeing_level', 'target_num']]
    
    col_medians = {}
    for col in feature_cols:
        col_medians[col] = float(dfs[0][col].median())
        for i in range(len(dfs)):
            dfs[i][col] = dfs[i][col].fillna(col_medians[col])

    pipeline_metadata = {
        'medians': medians,
        'freq_maps': freq_maps,
        'group_stats': group_stats_dict,
        'label_encoders': label_encoders,
        'target_num_cols': target_num_cols,
        'feature_cols': feature_cols,
        'col_medians': col_medians
    }

    if df_test is not None:
        return dfs[0][feature_cols], dfs[1][feature_cols], pipeline_metadata
    return dfs[0][feature_cols], pipeline_metadata

X_tr, X_te, pipeline_meta = extract_features(train_raw, test_raw)
print(f"[OK] Total fitur siap: {X_tr.shape[1]} kolom.")

# ------------------------------------------------------------------------------
# 2. PELATIHAN ENSEMBLE GRANDMASTER ZOO & OOF SCORING
# ------------------------------------------------------------------------------
print("[INFO] Melatih Ensemble Heterogen (XGBoost, CatBoost, ExtraTrees, RandomForest, Huber)...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_class = np.zeros((len(X_tr), 3))
oof_reg = np.zeros(len(X_tr))

# Simpan model lipatan untuk serving
trained_models = {
    'xgb': [],
    'cb': [],
    'et': [],
    'rf': [],
    'huber': []
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y)):
    X_tr_f, y_tr_f = X_tr.iloc[tr_idx], y[tr_idx]
    X_val_f, y_val_f = X_tr.iloc[val_idx], y[val_idx]

    # 1. XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=5,
                              subsample=0.8, colsample_bytree=0.7, random_state=42+fold, verbosity=0)
    xgb_m.fit(X_tr_f, y_tr_f)
    p_xgb = xgb_m.predict_proba(X_val_f)
    trained_models['xgb'].append(xgb_m)

    # 2. CatBoost
    if HAS_CATBOOST:
        cb_m = CatBoostClassifier(iterations=350, learning_rate=0.03, depth=6, random_seed=42+fold, verbose=0)
        cb_m.fit(X_tr_f, y_tr_f)
        p_cb = cb_m.predict_proba(X_val_f)
        trained_models['cb'].append(cb_m)
    else:
        p_cb = p_xgb

    # 3. ExtraTrees
    et_m = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42+fold, n_jobs=-1)
    et_m.fit(X_tr_f, y_tr_f)
    p_et = et_m.predict_proba(X_val_f)
    trained_models['et'].append(et_m)

    # 4. RandomForest
    rf_m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42+fold, n_jobs=-1)
    rf_m.fit(X_tr_f, y_tr_f)
    p_rf = rf_m.predict_proba(X_val_f)
    trained_models['rf'].append(rf_m)

    fold_class = 0.35 * p_xgb + 0.30 * p_cb + 0.15 * p_et + 0.20 * p_rf
    oof_class[val_idx] = fold_class

    # 5. Continuous Huber Regressor
    huber_m = HuberRegressor(max_iter=300)
    huber_m.fit(X_tr_f, y_tr_f.astype(float))
    r_huber = huber_m.predict(X_val_f)
    trained_models['huber'].append(huber_m)
    oof_reg[val_idx] = r_huber

# ------------------------------------------------------------------------------
# 3. KALIBRASI PRIOR PRDC (NELDER-MEAD)
# ------------------------------------------------------------------------------
print("[INFO] Menjalankan optimasi ambang batas PRDC...")
oof_exp_class = oof_class[:, 1] * 1.0 + oof_class[:, 2] * 2.0
fused_oof_score = 0.720 * oof_exp_class + 0.280 * oof_reg

def fast_macro_f1(y_true, preds):
    f1s = []
    for c in range(3):
        tp = np.sum((y_true == c) & (preds == c))
        fp = np.sum((y_true != c) & (preds == c))
        fn = np.sum((y_true == c) & (preds != c))
        denom = 2 * tp + fp + fn
        f1s.append(2.0 * tp / denom if denom > 0 else 0.0)
    return (f1s[0] + f1s[1] + f1s[2]) / 3.0

def prdc_loss(th):
    t1, t2 = th
    if t1 >= t2 or t1 < 0.45 or t2 > 1.50: return 10.0
    preds = np.zeros_like(fused_oof_score, dtype=int)
    preds[fused_oof_score >= t1] = 1
    preds[fused_oof_score >= t2] = 2
    base_f1 = fast_macro_f1(y, preds)
    p_low, p_med, p_high = np.mean(preds == 0), np.mean(preds == 1), np.mean(preds == 2)
    penalty = 1.0 * ((p_low - 0.488)**2 + (p_med - 0.358)**2 + (p_high - 0.154)**2)
    return -(base_f1 - penalty)

best_f1 = -1
best_cutoffs = [0.548, 1.374]

for init_t1 in [0.52, 0.54, 0.56]:
    for init_t2 in [1.35, 1.37, 1.39]:
        res = minimize(prdc_loss, [init_t1, init_t2], method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-4})
        t1, t2 = res.x
        preds = np.zeros_like(fused_oof_score, dtype=int)
        preds[fused_oof_score >= t1] = 1; preds[fused_oof_score >= t2] = 2
        score = fast_macro_f1(y, preds)
        if score > best_f1:
            best_f1 = score
            best_cutoffs = [float(t1), float(t2)]

print(f"[OK] Kalibrasi PRDC Selesai! OOF Macro F1: {best_f1:.4f}")
print(f"[OK] Ambang batas optimal: [t1={best_cutoffs[0]:.4f}, t2={best_cutoffs[1]:.4f}]")

# ------------------------------------------------------------------------------
# 4. INSTANSIASI KELAS PREDIKTOR DARI MODUL WELLUM_ENGINE
# ------------------------------------------------------------------------------
from wellum_engine import WellumPredictor

# ------------------------------------------------------------------------------
# 5. SERIALISASI & VERIFIKASI BUNDLE
# ------------------------------------------------------------------------------
predictor = WellumPredictor(trained_models, pipeline_meta, best_cutoffs)

bundle_path = 'models/wellum_model_bundle.joblib'
bundle_payload = {
    'predictor': predictor,
    'trained_models': trained_models,
    'pipeline_meta': pipeline_meta,
    'best_cutoffs': best_cutoffs,
    'metrics': {'oof_macro_f1': best_f1},
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
}

joblib.dump(bundle_payload, bundle_path, compress=3)
# Simpan juga ke path legacy eduwell_model_bundle.joblib
legacy_path = 'models/eduwell_model_bundle.joblib'
if not os.path.exists(legacy_path):
    try:
        os.link(bundle_path, legacy_path)
    except Exception:
        joblib.dump(bundle_payload, legacy_path, compress=3)

print(f"[OK] Bundle model berhasil disimpan: {bundle_path} ({os.path.getsize(bundle_path)/1024/1024:.2f} MB)")

# Uji Coba Inferensi pada Sample Test
print("[INFO] Menguji inferensi pada 5 sampel pertama test.csv...")
test_sample = test_raw.head(5)
res = predictor.predict(test_sample)
for i in range(5):
    print(f"  Responden {test_sample['respondent_id'].iloc[i]}: Prediksi = {res['labels'][i]} (Skor Fused = {res['fused_scores'][i]:.3f}, Probs = {res['probabilities'][i].round(3)})")

print("\n[SUKSES] Model Bundle Siap untuk Serving Dashboard Wellum!")
