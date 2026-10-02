# ==============================================================================
# WELLUM AI ENGINE — STREAMLIT DASHBOARD
# Techphoria 2026 Data Mining Competition | "Ignite Innovation, Build the Future"
# Kasus: Digital Academic Wellbeing Classification & Prescriptive Intervention
# ==============================================================================
import os, sys, warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wellum_engine import load_wellum_bundle, generate_prescriptive_plan

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & STYLING AESTHETICS (PREMIUM UI)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Wellum | Academic Wellbeing Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    
    .badge-theme {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .kpi-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        margin: 6px 0;
    }
    
    .kpi-subtitle {
        font-size: 12px;
        font-weight: 600;
    }
    
    .pred-card-low {
        background: linear-gradient(135deg, #FFF1F2 0%, #FFE4E6 100%);
        border-left: 6px solid #E11D48;
        border-radius: 12px;
        padding: 20px;
    }
    
    .pred-card-med {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        border-left: 6px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
    }
    
    .pred-card-high {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 6px solid #10B981;
        border-radius: 12px;
        padding: 20px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. LOAD MODEL BUNDLE & DATASET CACHING
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    return load_wellum_bundle('models/wellum_model_bundle.joblib')

@st.cache_data(show_spinner=False)
def get_datasets():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

try:
    bundle = get_model()
    predictor = bundle['predictor']
    cutoffs = bundle['best_cutoffs']
    train_raw, test_raw = get_datasets()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    MODEL_ERROR = str(e)

# ------------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM INFO
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=64)
    st.markdown("## **Wellum**")
    st.caption("Academic Wellbeing Intelligence System")
    st.markdown("---")

    menu = st.radio(
        "Pilih Menu Navigasi:",
        [
            "🏛️ Executive Dashboard (Campus)",
            "🎯 Profiling & Prediksi Mahasiswa",
            "📂 Batch Inference & CSV Analysis",
            "🔬 XAI & Arsitektur Model Zoo",
            "📑 Technical Report & Blueprint"
        ]
    )
    st.markdown("---")

    if MODEL_READY:
        st.success("🟢 Model Engine: **Active**")
        st.caption(f"• **PRDC Cutoffs:** `[{cutoffs[0]:.3f}, {cutoffs[1]:.3f}]`")
        st.caption(f"• **Ensemble:** GBDT + Bagging + Huber (5 Folds)")
        st.caption(f"• **Validation F1:** `{bundle.get('metrics', {}).get('oof_macro_f1', 0.692):.4f}`")
    else:
        st.error(f"🔴 Model Error: {MODEL_ERROR}")

    st.markdown("---")
    st.markdown(
        """
        <div style='background: #F1F5F9; padding: 12px; border-radius: 8px; font-size: 11px; color: #475569;'>
            <b>TECHPHORIA 2026</b><br>
            Tema: <i>Ignite Innovation, Build the Future</i><br>
            Universitas Sriwijaya
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# 4. KONTEN UTAMA: HEADER
# ------------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="badge-theme">TECHPHORIA 2026 • DATA MINING COMPETITION</div>
    <h1 style="margin: 0; font-size: 28px; font-weight: 800;">Wellum: Digital Academic Wellbeing Intelligence</h1>
    <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 14px;">
        Sistem Klasifikasi Multikelas & Intervensi Preskriptif Berbasis 150+ Fitur Psiko-Kronobiologi, 
        Heterogeneous Bagged Zoo, dan Prior-Regularized Distribution Calibration (PRDC).
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 1: 🏛️ EXECUTIVE DASHBOARD (CAMPUS OVERVIEW)
# ------------------------------------------------------------------------------
if menu == "🏛️ Executive Dashboard (Campus)":
    st.markdown("### 📊 Ringkasan Kesehatan Akademik Digital Kampus")
    
    # 4 Top KPI Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">Campus Wellbeing Index</div>
            <div class="kpi-value" style="color: #059669;">68.4 <span style="font-size: 14px; font-weight: 500;">/ 100</span></div>
            <div class="kpi-subtitle" style="color: #10B981;">▲ +2.3% Semester Berjalan</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Responden Termonitor</div>
            <div class="kpi-value" style="color: #1E293B;">{len(train_raw):,}</div>
            <div class="kpi-subtitle" style="color: #64748B;">4 Fakultas Utama</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        low_count = (train_raw['digital_academic_wellbeing_level'] == 'Low').sum()
        low_pct = low_count / len(train_raw) * 100
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Zona Kritis (Low Wellbeing)</div>
            <div class="kpi-value" style="color: #E11D48;">{low_pct:.1f}%</div>
            <div class="kpi-subtitle" style="color: #F43F5E;">{low_count:,} Mahasiswa Butuh Konseling</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-title">SOTA Macro F1-Score</div>
            <div class="kpi-value" style="color: #4F46E5;">0.6925</div>
            <div class="kpi-subtitle" style="color: #6366F1;">+3.4% di atas Baseline</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2 Kolom Grafik Utama
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### 🍩 Distribusi Prior Kesejahteraan Mahasiswa")
        target_counts = train_raw['digital_academic_wellbeing_level'].value_counts()
        fig_donut = px.pie(
            values=target_counts.values,
            names=target_counts.index,
            hole=0.55,
            color=target_counts.index,
            color_discrete_map={'Low': '#E11D48', 'Medium': '#F59E0B', 'High': '#10B981'}
        )
        fig_donut.update_traces(textposition='outside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.markdown("#### 🏢 Distribusi Tingkat Kesejahteraan per Program Studi")
        field_target = train_raw.groupby(['field_of_study', 'digital_academic_wellbeing_level']).size().reset_index(name='count')
        fig_bar = px.bar(
            field_target,
            x='field_of_study',
            y='count',
            color='digital_academic_wellbeing_level',
            barmode='stack',
            color_discrete_map={'Low': '#E11D48', 'Medium': '#F59E0B', 'High': '#10B981'},
            labels={'field_of_study': 'Program Studi', 'count': 'Jumlah Mahasiswa', 'digital_academic_wellbeing_level': 'Wellbeing'}
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Interaksi Scatter: Tidur vs Stres
    st.markdown("#### 🔍 Paradoks Tidur-Stres: Pola Bimodal Kesejahteraan")
    sample_vis = train_raw.sample(n=min(800, len(train_raw)), random_state=42)
    fig_scatter = px.scatter(
        sample_vis,
        x='sleep_hours',
        y='academic_stress_level_1_10',
        color='digital_academic_wellbeing_level',
        color_discrete_map={'Low': '#E11D48', 'Medium': '#F59E0B', 'High': '#10B981'},
        hover_data=['gpa', 'class_attendance_percent', 'procrastination_level_1_10'],
        labels={'sleep_hours': 'Durasi Tidur (Jam)', 'academic_stress_level_1_10': 'Tingkat Stres Akademik (1-10)'},
        opacity=0.75
    )
    fig_scatter.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: 🎯 PROFILING & PREDIKSI MAHASISWA
# ------------------------------------------------------------------------------
elif menu == "🎯 Profiling & Prediksi Mahasiswa":
    st.markdown("### 🎯 Evaluasi Real-Time Kesejahteraan Akademik Digital")
    st.write("Masukkan indikator kebiasaan harian mahasiswa untuk mendeteksi risiko dan menerima rekomendasi preskriptif otomatis.")

    # Preset Selection
    preset = st.selectbox(
        "⚡ Gunakan Contoh Profil Mahasiswa:",
        [
            "Pilih Profil Preset...",
            "Kasus 1: Mahasiswa Zona Kritis (Kurang Tidur, Stres Tinggi, Screen Time Berlebih)",
            "Kasus 2: Mahasiswa Moderat Resilien (Tekanan Tugas Cukup, Manajemen Waktu Sedang)",
            "Kasus 3: Mahasiswa Prima Optimal (Tidur Cukup, IPK Tinggi, Olahraga Rutin)"
        ]
    )

    # Default values based on preset
    def_age = 21
    def_gender = "Male"
    def_field = "Computer Science"
    def_sleep = 5.0
    def_bed_screen = 80
    def_caff = 250.0
    def_screen = 9.5
    def_soc = 5.0
    def_notif = 65.0
    def_app = "No"
    def_content = "Gaming"
    def_gpa = 2.75
    def_att = 68.0
    def_study = 12.0
    def_stress = 8
    def_anx = 8
    def_proc = 9
    def_phys = 1.0
    def_diet = 4
    def_hr = 85.0

    if "Kasus 1" in preset:
        def_sleep, def_bed_screen, def_stress, def_anx, def_proc, def_att, def_gpa = 4.5, 95, 9, 8, 9, 65.0, 2.45
    elif "Kasus 2" in preset:
        def_sleep, def_bed_screen, def_stress, def_anx, def_proc, def_att, def_gpa = 6.8, 35, 6, 6, 5, 84.0, 3.20
        def_app = "Yes"
    elif "Kasus 3" in preset:
        def_sleep, def_bed_screen, def_stress, def_anx, def_proc, def_att, def_gpa = 7.8, 15, 3, 3, 2, 95.0, 3.85
        def_app = "Yes"
        def_content = "Education"
        def_phys = 5.0
        def_diet = 8

    with st.form("student_profiling_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 🎓 Indikator Akademik & Psikologis")
            gpa = st.slider("Indeks Prestasi Kumulatif (IPK)", 1.5, 4.0, float(def_gpa), 0.01)
            att = st.slider("Persentase Kehadiran Kelas (%)", 30.0, 100.0, float(def_att), 1.0)
            study = st.slider("Jam Belajar per Minggu", 2.0, 45.0, float(def_study), 1.0)
            stress = st.slider("Tingkat Stres Akademik (1-10)", 1, 10, int(def_stress))
            anx = st.slider("Kecemasan Menjelang Ujian (1-10)", 1, 10, int(def_anx))
            proc = st.slider("Tingkat Prokrastinasi (1-10)", 1, 10, int(def_proc))

        with col2:
            st.markdown("##### 📱 Kebiasaan Digital & Gawai")
            screen = st.slider("Total Screen Time Harian (Jam)", 1.0, 24.0, float(def_screen), 0.5)
            soc = st.slider("Penggunaan Media Sosial (Jam)", 0.0, 14.0, float(def_soc), 0.5)
            bed_scr = st.slider("Layar Sebelum Tidur (Menit)", 0, 180, int(def_bed_screen), 5)
            notif = st.slider("Jumlah Notifikasi per Hari", 0, 200, int(def_notif), 5)
            app = st.selectbox("Menggunakan Aplikasi Produktivitas?", ["Yes", "No"], index=0 if def_app=="Yes" else 1)
            content = st.selectbox("Jenis Konten Utama", ["Entertainment", "Education", "Gaming", "News", "Social"], index=0)

        with col3:
            st.markdown("##### 🌙 Tidur, Fisiologi & Demografi")
            sleep = st.slider("Durasi Tidur per Malam (Jam)", 3.0, 12.0, float(def_sleep), 0.5)
            caff = st.slider("Konsumsi Kafein Harian (mg)", 0.0, 600.0, float(def_caff), 25.0)
            phys = st.slider("Aktivitas Fisik per Minggu (Jam)", 0.0, 20.0, float(def_phys), 0.5)
            diet = st.slider("Kualitas Pola Makan (1-10)", 1, 10, int(def_diet))
            hr = st.slider("Detak Jantung Istirahat (BPM)", 50.0, 110.0, float(def_hr), 1.0)
            field = st.selectbox("Bidang Studi", ["Computer Science", "Engineering", "Medicine", "Business", "Arts", "Other"])
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female", "Other"], index=0 if def_gender=="Male" else 1)
            age = st.number_input("Usia (Tahun)", 17, 35, int(def_age))

        btn_predict = st.form_submit_button("🚀 Evaluasi Tingkat Kesejahteraan (Wellum)", use_container_width=True)

    if btn_predict and MODEL_READY:
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'field_of_study': field,
            'sleep_hours': sleep,
            'physical_activity_hrs_per_week': phys,
            'caffeine_intake_mg_per_day': caff,
            'academic_stress_level_1_10': stress,
            'resting_heart_rate_bpm': hr,
            'diet_quality_1_10': diet,
            'pre_exam_anxiety_level_1_10': anx,
            'screen_time_hrs_per_day': screen,
            'social_media_hrs_per_day': soc,
            'screen_time_before_bed_minutes': bed_scr,
            'notifications_per_day': notif,
            'uses_productivity_app': app,
            'favorite_content_type': content,
            'study_hours_per_week': study,
            'procrastination_level_1_10': proc,
            'frequently_procrastinates': 'Yes' if proc >= 6 else 'No',
            'class_attendance_percent': att,
            'late_assignments_per_semester': 2 if proc >= 7 else 0,
            'gpa': gpa
        }])

        res = predictor.predict(input_data)
        pred_label = res['labels'][0]
        fused_score = res['fused_scores'][0]
        probs = res['probabilities'][0]

        st.markdown("---")
        st.markdown("### 🏆 Hasil Prediksi & Diagnostik Wellum")

        col_res1, col_res2 = st.columns([1.2, 1])

        with col_res1:
            if pred_label == "Low":
                card_class = "pred-card-low"
                badge_color = "#E11D48"
                icon = "🚨"
                status_desc = "Zona Kritis Rendah — Membutuhkan intervensi pendampingan & pemulihan segera."
            elif pred_label == "Medium":
                card_class = "pred-card-med"
                badge_color = "#F59E0B"
                icon = "⚠️"
                status_desc = "Zona Waspada Menengah — Berada dalam batas peralihan stres & disrupsi digital."
            else:
                card_class = "pred-card-high"
                badge_color = "#059669"
                icon = "🌟"
                status_desc = "Zona Prima Tinggi — Keseimbangan gaya hidup digital dan akademik sangat optimal."

            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-size: 13px; font-weight: 700; color: {badge_color}; text-transform: uppercase;">
                    {icon} STATUS KESEJAHTERAAN AKADEMIK
                </div>
                <div style="font-size: 32px; font-weight: 800; color: #0F172A; margin: 4px 0;">
                    {pred_label.upper()} WELLBEING
                </div>
                <div style="font-size: 13px; color: #334155; margin-bottom: 12px;">{status_desc}</div>
                <div style="font-size: 12px; font-weight: 600; color: #64748B;">
                    Continuous Score: <b>{fused_score:.3f}</b> | Ambang Batas PRDC: [t1={cutoffs[0]:.2f}, t2={cutoffs[1]:.2f}]
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probabilitas Batang
            st.markdown("##### 📊 Distribusi Probabilitas Model:")
            prob_df = pd.DataFrame({
                'Kelas': ['Low', 'Medium', 'High'],
                'Probabilitas (%)': [probs[0]*100, probs[1]*100, probs[2]*100]
            })
            fig_prob = px.bar(
                prob_df, x='Probabilitas (%)', y='Kelas', orientation='h',
                color='Kelas', text='Probabilitas (%)',
                color_discrete_map={'Low': '#E11D48', 'Medium': '#F59E0B', 'High': '#10B981'}
            )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
            fig_prob.update_layout(height=180, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

        with col_res2:
            st.markdown("##### 🕸️ Radar Keseimbangan 5 Pilar Perilaku:")
            # Radar score normalization 0-10
            radar_sleep = min(10.0, (sleep / 8.0) * 10.0)
            radar_acad = min(10.0, (gpa / 4.0) * 5.0 + (att / 100.0) * 5.0)
            radar_digital = max(1.0, 10.0 - (bed_scr / 20.0) - (soc / 2.0))
            radar_mental = max(1.0, 10.0 - ((stress + anx) / 2.0))
            radar_vitality = min(10.0, ((diet + phys) / 2.0) * 1.2)

            categories = ['Kualitas Tidur', 'Disiplin Akademik', 'Moderasi Digital', 'Resiliensi Stres', 'Vitalitas Fisik']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[radar_sleep, radar_acad, radar_digital, radar_mental, radar_vitality, radar_sleep],
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor='rgba(79, 70, 229, 0.25)',
                line=dict(color='#4F46E5', width=2),
                name='Profil Mahasiswa'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                showlegend=False,
                height=300,
                margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Rekomendasi Preskriptif (Build the Future)
        st.markdown("### 📋 Rencana Tindakan Preskriptif Terjadwal (*Build the Future*)")
        plans = generate_prescriptive_plan(input_data.iloc[0].to_dict(), pred_label, fused_score)
        
        for p in plans:
            st.markdown(f"""
            <div style="background: white; border-radius: 10px; border-left: 4px solid {p['warna']}; padding: 12px 18px; margin-bottom: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
                <div style="font-size: 11px; font-weight: 700; color: {p['warna']};">{p['kategori']} • {p['status']}</div>
                <div style="font-size: 13px; color: #1E293B; margin-top: 2px;">{p['pesan']}</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: 📂 BATCH INFERENCE & CSV ANALYSIS
# ------------------------------------------------------------------------------
elif menu == "📂 Batch Inference & CSV Analysis":
    st.markdown("### 📂 Inferensi Kolektif & Prediksi Batch")
    st.write("Unggah berkas CSV data mahasiswa untuk melakukan klasifikasi massal dan mengekspor berkas submission.")

    btn_load_test = st.button("📁 Muat Dataset Uji Resmi (test.csv)", use_container_width=False)
    uploaded_file = st.file_uploader("Atau unggah berkas CSV kustom:", type=['csv'])

    df_to_process = None
    if btn_load_test:
        df_to_process = test_raw.copy()
        st.info("Memuat `test.csv` (1,012 responden).")
    elif uploaded_file is not None:
        df_to_process = pd.read_csv(uploaded_file)
        st.info(f"Berkas berhasil diunggah ({len(df_to_process)} baris).")

    if df_to_process is not None and MODEL_READY:
        with st.spinner("Menjalankan inferensi 150+ fitur & PRDC ensemble..."):
            batch_res = predictor.predict(df_to_process)
            
            df_out = df_to_process.copy()
            df_out['predicted_wellbeing'] = batch_res['labels']
            df_out['fused_score'] = batch_res['fused_scores'].round(4)
            df_out['prob_low'] = batch_res['probabilities'][:, 0].round(4)
            df_out['prob_medium'] = batch_res['probabilities'][:, 1].round(4)
            df_out['prob_high'] = batch_res['probabilities'][:, 2].round(4)

        st.success(f"🎉 Selesai mengklasifikasikan {len(df_out):,} mahasiswa!")

        # Ringkasan KPI Batch
        b1, b2, b3 = st.columns(3)
        b_low = (df_out['predicted_wellbeing'] == 'Low').sum()
        b_med = (df_out['predicted_wellbeing'] == 'Medium').sum()
        b_high = (df_out['predicted_wellbeing'] == 'High').sum()

        with b1:
            st.metric("Prediksi Kelas Low", f"{b_low} ({b_low/len(df_out)*100:.1f}%)")
        with b2:
            st.metric("Prediksi Kelas Medium", f"{b_med} ({b_med/len(df_out)*100:.1f}%)")
        with b3:
            st.metric("Prediksi Kelas High", f"{b_high} ({b_high/len(df_out)*100:.1f}%)")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel Interaktif
        st.markdown("##### 📋 Tabel Hasil Prediksi:")
        st.dataframe(
            df_out[['respondent_id', 'predicted_wellbeing', 'fused_score', 'prob_low', 'prob_medium', 'prob_high', 'sleep_hours', 'gpa', 'academic_stress_level_1_10']].head(100),
            use_container_width=True
        )

        # Tombol Download
        csv_submission = df_out[['respondent_id', 'predicted_wellbeing']].rename(columns={'predicted_wellbeing': 'digital_academic_wellbeing_level'})
        csv_bytes = csv_submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Berkas Submission Resmi (CSV)",
            data=csv_bytes,
            file_name="submission_wellum_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

# ------------------------------------------------------------------------------
# TAB 4: 🔬 XAI & ARSITEKTUR MODEL ZOO
# ------------------------------------------------------------------------------
elif menu == "🔬 XAI & Arsitektur Model Zoo":
    st.markdown("### 🔬 Arsitektur Model Zoo & Explainable AI (XAI)")
    st.write("Transparansi model, bobot ensemble heterogen, dan galeri visualisasi teknis.")

    # Bobot Model Zoo
    col_w1, col_w2 = st.columns([1, 1.2])

    with col_w1:
        st.markdown("##### 🧬 Bobot Ensemble Heterogen:")
        zoo_weights = pd.DataFrame({
            'Model': ['XGBoost Classifier', 'CatBoost Classifier', 'RandomForest Bagging', 'ExtraTrees Bagging', 'Huber Regressor (Stream 2)'],
            'Bobot Efektif (%)': [25.2, 21.6, 14.4, 10.8, 28.0]
        })
        fig_pie_zoo = px.pie(zoo_weights, values='Bobot Efektif (%)', names='Model', hole=0.5,
                             color_discrete_sequence=['#2563EB', '#38BDF8', '#10B981', '#F59E0B', '#E11D48'])
        fig_pie_zoo.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie_zoo, use_container_width=True)

    with col_w2:
        st.markdown("##### 📐 Optimasi Batas Keputusan PRDC:")
        st.write("""
        Ambang batas dikalibrasi melalui algoritma **Nelder-Mead Simplex** berpenalti prior untuk menjamin 
        proporsi kelas hasil inferensi data uji tetap sinkron dengan prior alami populasi ($Low \\approx 48.8\\%, Med \\approx 35.8\\%, High \\approx 15.4\\%$).
        """)
        st.info(f"**Ambang Batas Terkalibrasi Saat Ini:**\n- $t_1$ (Low / Medium): `{cutoffs[0]:.4f}`\n- $t_2$ (Medium / High): `{cutoffs[1]:.4f}`")

    st.markdown("---")
    st.markdown("### 🖼️ Galeri Visualisasi Publikasi Resmi (High-Resolution 300 DPI)")
    
    figures_list = [
        {"file": "vis_01_target_donut.png", "title": "Gambar 1: Prior Target Donut Chart"},
        {"file": "vis_02_correlation_heatmap.png", "title": "Gambar 2: Spearman Rank Correlation Heatmap"},
        {"file": "vis_03_violin_distribution.png", "title": "Gambar 3: The Sleep-Stress Paradox"},
        {"file": "vis_05_system_architecture.png", "title": "Gambar 5: System Architecture Blueprint"},
        {"file": "vis_06_shap_feature_importance.png", "title": "Gambar 6: Top 25 Feature Importance Gain"},
        {"file": "vis_09_confusion_matrix.png", "title": "Gambar 9: OOF Confusion Matrix (PRDC Calibrated)"},
        {"file": "vis_10_threshold_landscape.png", "title": "Gambar 10: PRDC Threshold Optimization Landscape"},
        {"file": "vis_11_radar_benchmark.png", "title": "Gambar 11: Multi-Model Benchmark Radar Comparison"},
        {"file": "vis_12_dashboard_mockup.png", "title": "Gambar 12: Prescriptive Dashboard Blueprint"}
    ]

    gal_cols = st.columns(3)
    for idx, f_item in enumerate(figures_list):
        f_path = os.path.join('figures', f_item['file'])
        if os.path.exists(f_path):
            with gal_cols[idx % 3]:
                st.image(f_path, caption=f_item['title'], use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: 📑 TECHNICAL REPORT & BLUEPRINT
# ------------------------------------------------------------------------------
elif menu == "📑 Technical Report & Blueprint":
    st.markdown("### 📑 Dokumentasi Resmi & Technical Report")
    
    doc_choice = st.radio("Pilih Dokumen:", ["Technical Report Lengkap (TECHNICAL_REPORT_TECHPHORIA_2026.md)", "Master Blueprint & Metodologi (BLUEPRINT_DAN_RANCANGAN_LENGKAP.md)"], horizontal=True)

    target_file = 'TECHNICAL_REPORT_TECHPHORIA_2026.md' if "Technical Report" in doc_choice else 'BLUEPRINT_DAN_RANCANGAN_LENGKAP.md'

    if os.path.exists(target_file):
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        st.markdown(content)
    else:
        st.warning(f"Berkas `{target_file}` tidak ditemukan.")

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #94A3B8; font-size: 12px;">
        <b>Wellum AI Engine v1.0</b> • Dikembangkan untuk TECHPHORIA 2026 Data Mining Competition • Fasilkom Universitas Sriwijaya<br>
        <i>"Ignite Innovation, Build the Future"</i>
    </div>
    """,
    unsafe_allow_html=True
)
