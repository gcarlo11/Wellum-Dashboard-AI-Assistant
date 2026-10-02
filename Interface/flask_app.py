# ==============================================================================
# WELLUM AI ENGINE — FLASK DASHBOARD
# Techphoria 2026 Data Mining Competition | "Ignite Innovation, Build the Future"
# Wellum: Dashboard AI Assistant Pemantauan Tingkat Kesejahteraan Akademik Digital
# Mahasiswa sebagai Basis Intervensi Institusional Berbasis Data
# ==============================================================================
import os, sys, json, warnings, uuid
from datetime import datetime

warnings.filterwarnings('ignore')

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import numpy as np
import pandas as pd
from wellum_engine import load_wellum_bundle, generate_prescriptive_plan

# ==============================================================================
# APP INITIALIZATION
# ==============================================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ==============================================================================
# MODEL LOADING
# ==============================================================================
MODEL_PATH = os.path.join(PARENT_DIR, 'models', 'wellum_model_bundle.joblib')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PARENT_DIR, 'models', 'eduwell_model_bundle.joblib')

try:
    bundle = load_wellum_bundle(MODEL_PATH)
    predictor = bundle['predictor']
    cutoffs = bundle['best_cutoffs']
    MODEL_READY = True
    print("[OK] Wellum Model Bundle loaded successfully.")
except Exception as e:
    MODEL_READY = False
    predictor = None
    cutoffs = [0.5, 1.5]
    print(f"[ERROR] Model loading failed: {e}")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_session_path(session_id):
    return os.path.join(DATA_DIR, f'session_{session_id}.json')


def load_session_data(session_id):
    path = get_session_path(session_id)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def save_session_data(session_id, data):
    path = get_session_path(session_id)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_dashboard_analytics(students):
    """Compute aggregate analytics from student list."""
    if not students:
        return {}

    total = len(students)
    labels = [s.get('predicted_label', 'Medium') for s in students]
    low_count = labels.count('Low')
    med_count = labels.count('Medium')
    high_count = labels.count('High')

    def get_num_list(key):
        vals = []
        for s in students:
            v = s.get(key)
            if v is not None and v != '':
                try:
                    fv = float(v)
                    if not (np.isnan(fv) or np.isinf(fv)):
                        vals.append(fv)
                except (ValueError, TypeError):
                    pass
        return vals

    # Gather numeric stats safely
    sleep_vals = get_num_list('sleep_hours')
    gpa_vals = get_num_list('gpa')
    stress_vals = get_num_list('academic_stress_level_1_10')
    screen_vals = get_num_list('screen_time_hrs_per_day')
    attendance_vals = get_num_list('class_attendance_percent')
    proc_vals = get_num_list('procrastination_level_1_10')
    anxiety_vals = get_num_list('pre_exam_anxiety_level_1_10')
    social_vals = get_num_list('social_media_hrs_per_day')
    phys_vals = get_num_list('physical_activity_hrs_per_week')
    diet_vals = get_num_list('diet_quality_1_10')
    bed_screen_vals = get_num_list('screen_time_before_bed_minutes')

    def safe_avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0

    def safe_pct(lst, threshold, direction='below'):
        if not lst:
            return 0
        if direction == 'below':
            return round(sum(1 for v in lst if v < threshold) / len(lst) * 100, 1)
        else:
            return round(sum(1 for v in lst if v > threshold) / len(lst) * 100, 1)

    # Field of study distribution
    field_dist = {}
    for s in students:
        f = str(s.get('field_of_study') or 'Unknown')
        lbl = s.get('predicted_label', 'Medium')
        if f not in field_dist:
            field_dist[f] = {'Low': 0, 'Medium': 0, 'High': 0, 'total': 0}
        field_dist[f][lbl] += 1
        field_dist[f]['total'] += 1

    # Gender distribution
    gender_dist = {}
    for s in students:
        g = str(s.get('gender') or 'Unknown')
        lbl = s.get('predicted_label', 'Medium')
        if g not in gender_dist:
            gender_dist[g] = {'Low': 0, 'Medium': 0, 'High': 0, 'total': 0}
        gender_dist[g][lbl] += 1
        gender_dist[g]['total'] += 1

    # Fused score distribution for histogram
    score_vals = get_num_list('fused_score')
    score_bins = [0] * 10
    for sc in score_vals:
        idx = max(0, min(int(sc * 5), 9))  # ensure valid bin 0-9
        score_bins[idx] += 1

    # Wellbeing Index (composite 0-100)
    wellbeing_index = round(
        (safe_avg(sleep_vals) / 8.0 * 20) +
        (safe_avg(gpa_vals) / 4.0 * 25) +
        ((10.0 - safe_avg(stress_vals)) / 10.0 * 20) +
        (safe_avg(attendance_vals) / 100.0 * 20) +
        ((10.0 - safe_avg(proc_vals)) / 10.0 * 15),
        1
    )

    return {
        'total_students': total,
        'low_count': low_count,
        'medium_count': med_count,
        'high_count': high_count,
        'low_pct': round(low_count / total * 100, 1),
        'medium_pct': round(med_count / total * 100, 1),
        'high_pct': round(high_count / total * 100, 1),
        'wellbeing_index': wellbeing_index,
        'avg_sleep': safe_avg(sleep_vals),
        'avg_gpa': safe_avg(gpa_vals),
        'avg_stress': safe_avg(stress_vals),
        'avg_screen_time': safe_avg(screen_vals),
        'avg_attendance': safe_avg(attendance_vals),
        'avg_procrastination': safe_avg(proc_vals),
        'avg_anxiety': safe_avg(anxiety_vals),
        'avg_social_media': safe_avg(social_vals),
        'avg_physical_activity': safe_avg(phys_vals),
        'avg_diet_quality': safe_avg(diet_vals),
        'avg_bed_screen': safe_avg(bed_screen_vals),
        'pct_sleep_deficit': safe_pct(sleep_vals, 7.0, 'below'),
        'pct_high_stress': safe_pct(stress_vals, 7, 'above'),
        'pct_high_screen': safe_pct(screen_vals, 8, 'above'),
        'pct_low_attendance': safe_pct(attendance_vals, 75, 'below'),
        'field_distribution': field_dist,
        'gender_distribution': gender_dist,
        'score_distribution': score_bins,
        'score_values': [round(s, 4) for s in score_vals],
        'cutoffs': [round(c, 4) for c in cutoffs],
    }


def generate_insights(analytics):
    """Generate textual insights based on analytics."""
    insights = []

    if analytics.get('pct_sleep_deficit', 0) > 40:
        insights.append({
            'severity': 'critical',
            'category': 'Sleep Quality',
            'title': 'Sleep Deficit Crisis',
            'description': f"{analytics['pct_sleep_deficit']}% of students sleep less than 7 hours per night. "
                           f"Average sleep duration is {analytics['avg_sleep']} hours. "
                           f"Chronic sleep deprivation severely impairs prefrontal cortex function, memory consolidation, and academic performance.",
            'recommendation': 'Implement campus-wide sleep hygiene awareness programs. Consider adjusting early morning class schedules and providing circadian rhythm education.'
        })
    elif analytics.get('pct_sleep_deficit', 0) > 20:
        insights.append({
            'severity': 'warning',
            'category': 'Sleep Quality',
            'title': 'Moderate Sleep Deficit',
            'description': f"{analytics['pct_sleep_deficit']}% of students are below the recommended 7-hour sleep threshold. Average: {analytics['avg_sleep']} hours.",
            'recommendation': 'Encourage digital curfews and distribute sleep quality guides through student wellness centers.'
        })

    if analytics.get('pct_high_stress', 0) > 30:
        insights.append({
            'severity': 'critical',
            'category': 'Mental Health',
            'title': 'Elevated Academic Stress',
            'description': f"{analytics['pct_high_stress']}% of students report stress levels above 7/10. "
                           f"Average stress level: {analytics['avg_stress']}/10. "
                           f"Combined with average anxiety of {analytics['avg_anxiety']}/10, this indicates significant psychosocial burden.",
            'recommendation': 'Scale up counseling services capacity. Introduce structured exam preparation workshops and peer support groups for high-risk cohorts.'
        })

    if analytics.get('avg_screen_time', 0) > 7:
        insights.append({
            'severity': 'warning',
            'category': 'Digital Habits',
            'title': 'Excessive Screen Exposure',
            'description': f"Average daily screen time is {analytics['avg_screen_time']} hours, with {analytics['avg_social_media']} hours on social media. "
                           f"Pre-bedtime screen exposure averages {analytics['avg_bed_screen']} minutes, which disrupts melatonin secretion.",
            'recommendation': 'Promote digital wellness modules. Encourage app-based screen time tracking and implement blue-light reduction campaigns.'
        })

    if analytics.get('pct_low_attendance', 0) > 25:
        insights.append({
            'severity': 'critical',
            'category': 'Academic Engagement',
            'title': 'Low Class Attendance',
            'description': f"{analytics['pct_low_attendance']}% of students have attendance below 75%. "
                           f"Average attendance: {analytics['avg_attendance']}%. Low attendance strongly correlates with reduced wellbeing scores.",
            'recommendation': 'Deploy early warning attendance tracking systems. Schedule academic advisor check-ins for students below the 75% threshold.'
        })

    if analytics.get('low_pct', 0) > 40:
        insights.append({
            'severity': 'critical',
            'category': 'Overall Wellbeing',
            'title': 'High Proportion of At-Risk Students',
            'description': f"{analytics['low_pct']}% of the student population ({analytics['low_count']} students) "
                           f"are classified as Low wellbeing. This exceeds the campus safety threshold of 40%.",
            'recommendation': 'Activate comprehensive intervention protocols. Prioritize resources toward Low-classified students for immediate counseling referrals.'
        })

    if analytics.get('high_pct', 0) > 20:
        insights.append({
            'severity': 'positive',
            'category': 'Peer Leadership',
            'title': 'Strong Peer Mentor Pool',
            'description': f"{analytics['high_pct']}% of students ({analytics['high_count']} students) show High wellbeing levels. "
                           f"These students demonstrate optimal balance between digital habits and academic performance.",
            'recommendation': 'Recruit High-wellbeing students as peer mentors and study group facilitators for at-risk cohorts.'
        })

    if not insights:
        insights.append({
            'severity': 'positive',
            'category': 'Overall Assessment',
            'title': 'Student Wellbeing Within Normal Parameters',
            'description': 'Current metrics indicate student wellbeing is within acceptable ranges across all measured dimensions.',
            'recommendation': 'Maintain current support programs and continue monitoring through regular assessment cycles.'
        })

    return insights


# ==============================================================================
# ROUTES
# ==============================================================================
@app.route('/')
def index():
    """Landing / upload page."""
    sessions = []
    if os.path.exists(DATA_DIR):
        for fn in sorted(os.listdir(DATA_DIR), reverse=True):
            if fn.startswith('session_') and fn.endswith('.json'):
                sid = fn.replace('session_', '').replace('.json', '')
                data = load_session_data(sid)
                if data:
                    sessions.append({
                        'id': sid,
                        'filename': data.get('filename', 'Unknown'),
                        'total': data.get('total_students', 0),
                        'created_at': data.get('created_at', ''),
                    })
    return render_template('index.html', model_ready=MODEL_READY, sessions=sessions)


@app.route('/upload', methods=['POST'])
def upload():
    """Handle CSV upload, run inference, save results as JSON."""
    if not MODEL_READY:
        return jsonify({'error': 'Model is not loaded. Please ensure the model bundle exists.'}), 500

    file = request.files.get('csv_file')
    if not file or not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Please upload a valid CSV file (.csv).'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to parse CSV: {str(e)}'}), 400

    if len(df) == 0:
        return jsonify({'error': 'The uploaded CSV file is empty.'}), 400

    # Run inference
    try:
        results = predictor.predict(df)
    except Exception as e:
        return jsonify({'error': f'Inference error: {str(e)}'}), 500

    # Build student records and save session
    try:
        students = []
        for i, row in df.iterrows():
            record = row.to_dict()
            # Clean NaN / inf / pd.NA values for JSON serialization
            for k, v in list(record.items()):
                if pd.isna(v):
                    record[k] = None
                elif isinstance(v, (np.integer, int)):
                    record[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    if np.isnan(v) or np.isinf(v):
                        record[k] = None
                    else:
                        record[k] = round(float(v), 4)

            record['predicted_label'] = results['labels'][i]
            record['predicted_code'] = int(results['predictions'][i])
            record['fused_score'] = round(float(results['fused_scores'][i]), 4)
            record['prob_low'] = round(float(results['probabilities'][i][0]), 4)
            record['prob_medium'] = round(float(results['probabilities'][i][1]), 4)
            record['prob_high'] = round(float(results['probabilities'][i][2]), 4)

            # Generate prescriptive plan
            plans = generate_prescriptive_plan(record, record['predicted_label'], record['fused_score'])
            # Clean plans (remove emoji for professional display)
            clean_plans = []
            for p in plans:
                clean_plans.append({
                    'category': p.get('kategori', '').encode('ascii', 'ignore').decode('ascii').strip(),
                    'status': p.get('status', ''),
                    'color': p.get('warna', '#0066cc'),
                    'message': p.get('pesan', '')
                })
            record['prescriptive_plans'] = clean_plans
            students.append(record)

        # Create session
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]

        session_data = {
            'session_id': session_id,
            'filename': file.filename,
            'created_at': datetime.now().isoformat(),
            'total_students': len(students),
            'cutoffs': [round(float(c), 4) for c in cutoffs],
            'students': students,
        }

        save_session_data(session_id, session_data)
        return jsonify({'session_id': session_id, 'total': len(students)}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    """Main dashboard view."""
    data = load_session_data(session_id)
    if not data:
        return redirect(url_for('index'))

    students = data.get('students', [])
    analytics = compute_dashboard_analytics(students)
    insights = generate_insights(analytics)

    return render_template(
        'dashboard.html',
        session=data,
        analytics=analytics,
        insights=insights,
        analytics_json=json.dumps(analytics),
    )


@app.route('/students/<session_id>')
def students_list(session_id):
    """Student list view with filters."""
    data = load_session_data(session_id)
    if not data:
        return redirect(url_for('index'))
    return render_template('students.html', session=data)


@app.route('/api/students/<session_id>')
def api_get_students(session_id):
    """API: Get all students for a session."""
    data = load_session_data(session_id)
    if not data:
        return jsonify({'error': 'Session not found'}), 404

    filter_label = request.args.get('label', '')
    search = request.args.get('search', '').lower()

    students = data.get('students', [])

    if filter_label and filter_label in ('Low', 'Medium', 'High'):
        students = [s for s in students if s.get('predicted_label') == filter_label]

    if search:
        students = [s for s in students if
                    search in str(s.get('respondent_id', '')).lower() or
                    search in str(s.get('field_of_study', '')).lower() or
                    search in str(s.get('gender', '')).lower()]

    return jsonify({'students': students, 'total': len(students)})


@app.route('/api/student/<session_id>/<respondent_id>', methods=['GET'])
def api_get_student(session_id, respondent_id):
    """API: Get a single student."""
    data = load_session_data(session_id)
    if not data:
        return jsonify({'error': 'Session not found'}), 404

    for s in data.get('students', []):
        if str(s.get('respondent_id', '')) == respondent_id:
            return jsonify(s)
    return jsonify({'error': 'Student not found'}), 404


@app.route('/api/student/<session_id>/<respondent_id>', methods=['PUT'])
def api_update_student(session_id, respondent_id):
    """API: Update student label (manual override)."""
    data = load_session_data(session_id)
    if not data:
        return jsonify({'error': 'Session not found'}), 404

    updates = request.get_json()
    if not updates:
        return jsonify({'error': 'No data provided'}), 400

    for i, s in enumerate(data['students']):
        if str(s.get('respondent_id', '')) == respondent_id:
            # Allow updating predicted_label (manual override)
            if 'predicted_label' in updates:
                new_label = updates['predicted_label']
                if new_label in ('Low', 'Medium', 'High'):
                    data['students'][i]['predicted_label'] = new_label
                    data['students'][i]['manual_override'] = True
                    data['students'][i]['override_timestamp'] = datetime.now().isoformat()

            # Allow updating notes
            if 'notes' in updates:
                data['students'][i]['notes'] = updates['notes']

            save_session_data(session_id, data)
            return jsonify({'success': True, 'student': data['students'][i]})

    return jsonify({'error': 'Student not found'}), 404


@app.route('/api/session/<session_id>/delete', methods=['DELETE'])
def api_delete_session(session_id):
    """API: Delete a session."""
    path = get_session_path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'success': True})
    return jsonify({'error': 'Session not found'}), 404


# ==============================================================================
# RUN
# ==============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Wellum AI Engine - Flask Dashboard")
    print("  TECHPHORIA 2026 | Ignite Innovation, Build the Future")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
