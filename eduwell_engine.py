# ==============================================================================
# WELLUM ENGINE: INFERENCE & PRESCRIPTIVE INTELLIGENCE CORE
# Modul Standar untuk Serving Model, Preprocessing & Rekomendasi Preskriptif
# ==============================================================================
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

class WellumPredictor:
    """
    Kelas prediktor mandiri untuk Wellum AI Engine.
    Menerapkan pipeline rekayasa 150+ fitur, fusi multi-model ensemble (GBDT + Bagging + Huber),
    serta kalibrasi prior PRDC (Nelder-Mead).
    """
    def __init__(self, models, pipeline_meta, cutoffs):
        self.models = models
        self.meta = pipeline_meta
        self.cutoffs = cutoffs
        self.inv_map = {0: 'Low', 1: 'Medium', 2: 'High'}

    def preprocess(self, df_input):
        df = df_input.copy()
        
        # 1. Format Dasar & Sanitasi
        df['age'] = df['age'].astype(str).str.extract(r'(\d+)').astype(float)
        df['gender'] = df['gender'].astype(str).str.strip().str.capitalize()
        df['notifications_clean'] = df['notifications_per_day'].replace(-5, np.nan)
        df['screen_time_clipped'] = df['screen_time_hrs_per_day'].clip(upper=24)

        missing_cols = ['caffeine_intake_mg_per_day', 'social_media_hrs_per_day',
                        'uses_productivity_app', 'favorite_content_type',
                        'study_hours_per_week', 'class_attendance_percent']
        for col in missing_cols:
            if col in df.columns:
                df[f'{col}_isna'] = df[col].isnull().astype(int)
            else:
                df[f'{col}_isna'] = 0

        df['n_missing_total'] = sum(df[c].isnull().astype(int) for c in missing_cols if c in df.columns)
        df['productivity_app_num'] = df['uses_productivity_app'].map({'Yes': 1, 'No': 0}).fillna(-1) if 'uses_productivity_app' in df.columns else -1
        df['frequently_procrastinates_num'] = df['frequently_procrastinates'].map({'Yes': 1, 'No': 0}) if 'frequently_procrastinates' in df.columns else 0
        df['uses_productivity_app_clean'] = df['uses_productivity_app'].fillna('Missing') if 'uses_productivity_app' in df.columns else 'Missing'
        df['favorite_content_type_clean'] = df['favorite_content_type'].fillna('Missing') if 'favorite_content_type' in df.columns else 'Missing'

        for col, med in self.meta['medians'].items():
            if col in df.columns:
                df[col] = df[col].fillna(med)
            else:
                df[col] = med

        # 2. Rekayasa Fitur Domain Psiko-Fisiologi & Kronobiologi
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
        late = df['late_assignments_per_semester'] if 'late_assignments_per_semester' in df.columns else 0
        hr = df['resting_heart_rate_bpm']
        prod_app = df['productivity_app_num']

        pos = (sleep / 7.5) * (gpa / 2.5) * (att / 80.0)
        neg = (proc / 5.0) * (bed_scr / 30.0 + 0.1) * (soc / 4.0 + 0.1) * (stress / 5.0)
        df['wellbeing_force_ratio'] = pos / (neg + 0.05)
        df['log_wellbeing_ratio'] = np.log1p(df['wellbeing_force_ratio'])
        df['academic_health'] = gpa * (att / 100.0)
        df['academic_vitality'] = (gpa / 4.0) * (att / 100.0) * (sleep / 8.0)
        df['academic_strain'] = (stress + anx + proc) / (gpa + 0.1)

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

        df['digital_burden'] = (soc + (bed_scr / 60.0)) / (sleep + 0.1)
        df['screen_to_sleep_ratio'] = screen / (sleep + 0.1)
        df['social_media_share'] = soc / (screen + 0.1)
        df['digital_distraction_index'] = (notif / 50.0) * (bed_scr / 30.0)
        df['total_digital_hours'] = screen + soc
        df['digital_intensity'] = (screen + soc + (bed_scr / 60.0)) / (sleep + study + 0.1)

        df['composite_stress'] = (stress + anx + proc) / 3.0
        df['stress_anxiety_product'] = stress * anx
        df['triple_psych_burden'] = stress * anx * proc
        df['study_efficiency'] = gpa / (study + 1.0)
        df['health_index'] = (diet + phys + sleep) / 3.0
        df['health_vs_stress'] = (diet + phys + sleep) / (stress + anx + 0.1)
        df['caffeine_to_sleep'] = caff / (sleep + 0.1)
        df['heart_rate_stress_index'] = hr * stress / 10.0

        df['proc_squared'] = proc ** 2
        df['anxiety_squared'] = anx ** 2
        df['stress_squared'] = stress ** 2
        df['sleep_squared'] = sleep ** 2
        df['gpa_squared'] = gpa ** 2

        df['prod_app_x_proc'] = prod_app * proc
        df['prod_app_x_gpa'] = prod_app * gpa
        df['prod_app_x_sleep'] = prod_app * sleep
        df['prod_app_x_stress'] = prod_app * stress

        df['vitality_to_strain_ratio'] = df['academic_vitality'] / (df['academic_strain'] + 0.05)
        df['log_vitality_strain_ratio'] = np.log1p(np.maximum(0, df['vitality_to_strain_ratio']))
        df['procrastination_to_wellbeing'] = proc / (df['wellbeing_force_ratio'] + 0.05)
        df['net_sleep_surplus'] = sleep - 7.0 - (bed_scr / 60.0)
        soc_weekly = soc * 7.0
        df['active_study_ratio'] = study / (study + soc_weekly + 0.1)
        df['attendance_resilience'] = (att / 100.0) * (1.0 - stress / 11.0)
        df['cardio_stress_load'] = (hr / 70.0) * ((stress + anx) / 10.0)

        for col_name in ['sleep_hours', 'gpa', 'academic_stress_level_1_10', 'wellbeing_force_ratio']:
            df[f'{col_name}_rank'] = 0.5

        # Frequency Maps
        for col, f_map in self.meta['freq_maps'].items():
            if col in df.columns:
                df[f'{col}_freq'] = df[col].map(f_map).fillna(0)
            else:
                df[f'{col}_freq'] = 0

        # Group Aggregations
        for grp, stats_map in self.meta['group_stats'].items():
            for num, grp_df in stats_map.items():
                if grp in df.columns:
                    df = df.merge(grp_df, on=grp, how='left')
                    df[f'{num}_by_{grp}_mean'] = df['mean']
                    df[f'{num}_by_{grp}_std'] = df['std']
                    df = df.drop(columns=['mean', 'std'], errors='ignore')
                else:
                    df[f'{num}_by_{grp}_mean'] = self.meta['col_medians'].get(f'{num}_by_{grp}_mean', 0)
                    df[f'{num}_by_{grp}_std'] = self.meta['col_medians'].get(f'{num}_by_{grp}_std', 1)

                df[f'{num}_diff_{grp}'] = df[num] - df[f'{num}_by_{grp}_mean']
                df[f'{num}_z_{grp}'] = df[f'{num}_diff_{grp}'] / (df[f'{num}_by_{grp}_std'] + 1e-5)

        # Label Encoders
        for col, le in self.meta['label_encoders'].items():
            if col in df.columns:
                known_classes = set(le.classes_)
                df[col] = df[col].astype(str).map(lambda s: s if s in known_classes else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
            else:
                df[col] = 0

        for col in self.meta['feature_cols']:
            if col not in df.columns:
                df[col] = self.meta['col_medians'].get(col, 0.0)
            df[col] = df[col].fillna(self.meta['col_medians'].get(col, 0.0))

        return df[self.meta['feature_cols']]

    def predict(self, df_input):
        X = self.preprocess(df_input)
        n_samples = len(X)

        if 'lgb' in self.models and len(self.models['lgb']) > 0:
            # 4-Seed SOTA Multi-Seed Grandmaster Zoo
            p_lgb = np.mean([m.predict_proba(X) for m in self.models['lgb']], axis=0)
            p_xgb = np.mean([m.predict_proba(X) for m in self.models['xgb']], axis=0)
            p_cb = np.mean([m.predict_proba(X) for m in self.models['cb']], axis=0) if self.models.get('cb') else p_lgb
            p_et = np.mean([m.predict_proba(X) for m in self.models['et']], axis=0)

            # Re-proportioned ensemble (LightGBM 35%, XGB 30%, CatBoost 22%, ExtraTrees 13%)
            p_class = 0.35 * p_lgb + 0.30 * p_xgb + 0.22 * p_cb + 0.13 * p_et

            r_cb = np.mean([m.predict(X) for m in self.models['cb_r']], axis=0) if self.models.get('cb_r') else 0.0
            r_lgb = np.mean([m.predict(X) for m in self.models['lgb_r']], axis=0) if self.models.get('lgb_r') else 0.0
            r_reg = 0.5 * r_cb + 0.5 * r_lgb if self.models.get('cb_r') else r_lgb
        else:
            # Multi-Seed / Baseline Ensemble
            p_xgb = np.mean([m.predict_proba(X) for m in self.models['xgb']], axis=0)
            p_cb = np.mean([m.predict_proba(X) for m in self.models['cb']], axis=0) if self.models.get('cb') else p_xgb
            p_et = np.mean([m.predict_proba(X) for m in self.models['et']], axis=0)
            p_rf = np.mean([m.predict_proba(X) for m in self.models['rf']], axis=0) if self.models.get('rf') else p_xgb
            p_class = 0.35 * p_xgb + 0.30 * p_cb + 0.15 * p_et + 0.20 * p_rf

            if self.models.get('cb_r') and len(self.models['cb_r']) > 0:
                r_reg = np.mean([m.predict(X) for m in self.models['cb_r']], axis=0)
            elif self.models.get('huber') and len(self.models['huber']) > 0:
                r_reg = np.mean([m.predict(X) for m in self.models['huber']], axis=0)
            else:
                r_reg = 0.0

        exp_class = p_class[:, 1] * 1.0 + p_class[:, 2] * 2.0
        fused_score = 0.720 * exp_class + 0.280 * r_reg

        preds = np.zeros(n_samples, dtype=int)
        preds[fused_score >= self.cutoffs[0]] = 1
        preds[fused_score >= self.cutoffs[1]] = 2

        pred_labels = [self.inv_map[p] for p in preds]
        return {
            'predictions': preds,
            'labels': pred_labels,
            'probabilities': p_class,
            'fused_scores': fused_score
        }

# Alias untuk backward compatibility
EduWellPredictor = WellumPredictor

def load_wellum_bundle(model_path=None):
    """Memuat bundle model yang tersimpan (mendukung Wellum dan EduWell)."""
    if model_path is None:
        if os.path.exists('models/wellum_model_bundle.joblib'):
            model_path = 'models/wellum_model_bundle.joblib'
        elif os.path.exists('models/eduwell_model_bundle.joblib'):
            model_path = 'models/eduwell_model_bundle.joblib'
        else:
            model_path = 'models/wellum_model_bundle.joblib'
            
    if not os.path.exists(model_path):
        # Fallback check
        alt_path = 'models/eduwell_model_bundle.joblib' if 'wellum' in model_path else 'models/wellum_model_bundle.joblib'
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model bundle tidak ditemukan di: {model_path}. Silakan jalankan train_and_save_model.py terlebih dahulu.")
    return joblib.load(model_path)

load_eduwell_bundle = load_wellum_bundle

def _safe_float(val, default):
    """Safely convert value to float with fallback for None, NaN, and conversion errors."""
    if val is None or val == '' or pd.isna(val):
        return float(default)
    try:
        f = float(val)
        return float(default) if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return float(default)

def generate_prescriptive_plan(row_data, pred_label, fused_score):
    """
    Menghasilkan rekomendasi intervensi preskriptif terpersonalisasi
    berdasarkan profil perilaku mahasiswa (Pilar 'Build the Future').
    """
    plans = []
    
    # 1. Analisis Tidur & Sirkadian
    sleep = _safe_float(row_data.get('sleep_hours'), 7.0)
    bed_scr = _safe_float(row_data.get('screen_time_before_bed_minutes'), 30.0)
    caff = _safe_float(row_data.get('caffeine_intake_mg_per_day'), 100.0)
    
    if sleep < 6.0:
        plans.append({
            'kategori': '🌙 Ritme Sirkadian & Tidur',
            'status': 'KRITIS',
            'warna': '#E11D48',
            'pesan': f'Defisit tidur parah ({sleep:.1f} jam/malam). Wajib target minimal 7.0 jam untuk regenerasi fungsi kognitif korteks prefrontal.'
        })
    elif sleep < 7.0:
        plans.append({
            'kategori': '🌙 Ritme Sirkadian & Tidur',
            'status': 'PERHATIAN',
            'warna': '#F59E0B',
            'pesan': f'Durasi tidur moderat ({sleep:.1f} jam). Tingkatkan 30–45 menit untuk stabilitas memori jangka panjang.'
        })

    if bed_scr >= 60.0:
        plans.append({
            'kategori': '📱 Paparan Layar Pra-Tidur',
            'status': 'KRITIS',
            'warna': '#E11D48',
            'pesan': f'Paparan gawai pra-tidur sangat tinggi ({bed_scr:.0f} menit). Matikan gawai 45 menit sebelum tidur untuk memulihkan sekresi alami melatonin.'
        })

    # 2. Analisis Beban Psikologis & Stres
    stress = _safe_float(row_data.get('academic_stress_level_1_10'), 5.0)
    anx = _safe_float(row_data.get('pre_exam_anxiety_level_1_10'), 5.0)
    proc = _safe_float(row_data.get('procrastination_level_1_10'), 5.0)

    if stress >= 8.0 or anx >= 8.0:
        plans.append({
            'kategori': '🧠 Kesehatan Mental & Stres',
            'status': 'INTERVENSI SEGERA',
            'warna': '#E11D48',
            'pesan': f'Tingkat stres/kecemasan akademik melampaui batas aman (Skor {max(stress, anx):.0f}/10). Rujuk ke sesi konseling psikolog kampus fakultas.'
        })

    if proc >= 7.0:
        plans.append({
            'kategori': '⏳ Manajemen Prokrastinasi',
            'status': 'PERHATIAN',
            'warna': '#F59E0B',
            'pesan': 'Kecenderungan menunda tugas tinggi. Terapkan teknik Pomodoro 25/5 menit dan pecah tugas besar menjadi deliverable harian kecil.'
        })

    # 3. Analisis Akademik
    att = _safe_float(row_data.get('class_attendance_percent'), 85.0)
    gpa = _safe_float(row_data.get('gpa'), 3.0)

    if att < 75.0:
        plans.append({
            'kategori': '🎓 Ketahanan Akademik',
            'status': 'PERINGATAN AKADEMIK',
            'warna': '#E11D48',
            'pesan': f'Kehadiran kelas berada di bawah ambang minimal ({att:.1f}%). Konsultasikan kendala kehadiran dengan Dosen Pembimbing Akademik.'
        })

    # Rekomendasi Khusus Level High
    if pred_label == 'High':
        plans.append({
            'kategori': '🌟 Peer Mentorship & Leadership',
            'status': 'OPTIMAL',
            'warna': '#059669',
            'pesan': 'Keseimbangan gaya hidup dan performa akademik sangat prima. Direkomendasikan menjadi Peer Mentor untuk mendampingi rekan mahasiswa tingkat pertama.'
        })

    return plans
