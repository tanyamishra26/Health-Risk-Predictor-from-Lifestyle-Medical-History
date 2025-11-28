# Python script to compute five continuous risk scores and save augmented CSV.
# Updated with realistic noise added per-risk-category.

import pandas as pd
import numpy as np
from pathlib import Path

# --- User-editable path ---
input_path = Path("Diabetes_and_LifeStyle_Dataset .csv")
output_path = Path("augmented_risks_with_noise.csv")

df = pd.read_csv(input_path)
df_orig = df.copy()

# helper functions
def clip01(x):
    return np.minimum(1, np.maximum(0, x))

def norm_series(series, low, high):
    return clip01((series - low) / (high - low))

# Binary helpers for categorical values
def is_current_smoker(s):
    if pd.isna(s): return 0
    s = str(s).strip().lower()
    return 1 if ("current" in s) or (s in ["yes","y","smoker","smoking"]) else 0

def bool_to_bin(v):
    if pd.isna(v): return 0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return 1 if v==1 else 0
    s = str(v).strip().lower()
    return 1 if s in ["1","true","yes","y"] else 0

# Prepare normalized feature columns
age = df.get('Age')
bmi = df.get('bmi')
whr = df.get('waist_to_hip_ratio')
phys_min_wk = df.get('physical_activity_minutes_per_week')
family_dx = df.get('family_history_diabetes')
fg = df.get('glucose_fasting')
hba1c = df.get('hba1c')
tg = df.get('triglycerides')
sbp = df.get('systolic_bp')
dbp = df.get('diastolic_bp')
smoking = df.get('smoking_status')
tc = df.get('cholesterol_total')
hdl = df.get('hdl_cholesterol')
ldl = df.get('ldl_cholesterol')
diet_score = df.get('diet_score')
sleep_h = df.get('sleep_hours_per_day')
screen_time = df.get('screen_time_hours_per_day')
diagnosed_diabetes = df.get('diagnosed_diabetes')

# Normalizations
age_norm_20_70 = norm_series(age, 20, 70)
age_norm_20_80 = norm_series(age, 20, 80)
age_norm_40_80 = norm_series(age, 40, 80)

bmi_norm = norm_series(bmi, 18.5, 35)
whr_norm = norm_series(whr, 0.7, 1.1)

phys_act_norm = clip01(1 - (phys_min_wk.fillna(0) / 300))

fam_dx_bin = df['family_history_diabetes'].apply(bool_to_bin) if 'family_history_diabetes' in df.columns else pd.Series(0, index=df.index)

fg_norm = norm_series(fg, 70, 200)
hba1c_norm = norm_series(hba1c, 4.5, 9.0)
tg_norm = norm_series(tg, 50, 500)

sbp_norm = norm_series(sbp, 90, 170)
dbp_norm = norm_series(dbp, 50, 100)
smoker_bin = df['smoking_status'].apply(is_current_smoker) if 'smoking_status' in df.columns else pd.Series(0, index=df.index)

tc_norm = norm_series(tc, 120, 300)
hdl_inv = clip01((60 - hdl.fillna(60)) / 40)

diet_norm = clip01(1 - (diet_score.fillna(50) / 100))
sleep_norm = clip01(np.abs(7 - sleep_h.fillna(7)) / 6)
sedentary_norm = norm_series(screen_time.fillna(0), 0, 16)

non_hdl = (tc.fillna(np.nan) - hdl.fillna(np.nan))
nonhdl_norm = norm_series(non_hdl, 70, 270)
ldl_norm = norm_series(ldl, 50, 240)
tg_norm_for_chol = norm_series(tg, 30, 500)

# Weighted scoring helper
def weighted_score(components, weights):
    comp_df = pd.DataFrame(components)
    w = pd.Series(weights)
    present_mask = ~comp_df.isna()
    comp_df_filled = comp_df.fillna(0)
    w_arr = w.reindex(comp_df_filled.columns).fillna(0).values
    effective_w_sum = (present_mask.values * w_arr).sum(axis=1)
    effective_w_sum_safe = np.where(effective_w_sum==0, 1.0, effective_w_sum)
    raw = comp_df_filled.values.dot(w_arr)
    score = raw / effective_w_sum_safe
    return clip01(score) * 100.0

# ------------------------------
# Compute all 5 risk scores
# ------------------------------

# 1) Diabetes
diabetes_components = {
    'age': age_norm_20_70,
    'bmi': bmi_norm,
    'whr': whr_norm,
    'phys_act': phys_act_norm,
    'fam_dx': fam_dx_bin,
    'fg': fg_norm,
    'hba1c': hba1c_norm,
    'tg': tg_norm
}
diabetes_weights = {'age':0.12, 'bmi':0.18, 'whr':0.12, 'phys_act':0.12, 'fam_dx':0.12, 'fg':0.18, 'hba1c':0.12, 'tg':0.04}
df['Diabetes_risk_score_custom'] = weighted_score(diabetes_components, diabetes_weights)

# 2) Hypertension
hypertension_components = {
    'sbp': sbp_norm,
    'dbp': dbp_norm,
    'age': age_norm_20_80,
    'bmi': bmi_norm,
    'smoker': smoker_bin,
    'phys_act': phys_act_norm
}
hypertension_weights = {'sbp':0.35, 'dbp':0.20, 'age':0.20, 'bmi':0.15, 'smoker':0.07, 'phys_act':0.03}
df['Hypertension_risk_score_custom'] = weighted_score(hypertension_components, hypertension_weights)

# 3) Heart disease (ASCVD-like)
diag_bin = df['diagnosed_diabetes'].apply(bool_to_bin) if 'diagnosed_diabetes' in df.columns else pd.Series(0, index=df.index)
diabetes_bin_for_ascvd = diag_bin.where(diag_bin==1, df['Diabetes_risk_score_custom'].fillna(0) > 50).astype(int)

ascvd_components = {
    'age': age_norm_40_80,
    'sbp': sbp_norm,
    'tc': tc_norm,
    'hdl_inv': hdl_inv,
    'smoker': smoker_bin,
    'diabetes_bin': diabetes_bin_for_ascvd
}
ascvd_weights = {'age':0.30, 'sbp':0.22, 'tc':0.16, 'hdl_inv':0.12, 'smoker':0.12, 'diabetes_bin':0.08}
df['HeartDisease_risk_score_custom'] = weighted_score(ascvd_components, ascvd_weights)

# 4) Obesity
obesity_components = {
    'bmi': bmi_norm,
    'whr': whr_norm,
    'phys_act': phys_act_norm,
    'diet': diet_norm,
    'sleep': sleep_norm,
    'sedentary': sedentary_norm
}
obesity_weights = {'bmi':0.35, 'whr':0.25, 'phys_act':0.15, 'diet':0.12, 'sleep':0.08, 'sedentary':0.05}
df['Obesity_risk_score_custom'] = weighted_score(obesity_components, obesity_weights)

# 5) Cholesterol
chol_components = {
    'nonhdl': nonhdl_norm,
    'ldl': ldl_norm,
    'hdl_inv': hdl_inv,
    'tg': tg_norm_for_chol
}
chol_weights = {'nonhdl':0.35, 'ldl':0.30, 'hdl_inv':0.20, 'tg':0.15}
df['Cholesterol_risk_score_custom'] = weighted_score(chol_components, chol_weights)

# ------------------------------
# Add DIFFERENT REALISTIC NOISE
# ------------------------------

rng = np.random.default_rng(seed=42)

def add_noise(series, sigma):
    noise = rng.normal(loc=0, scale=sigma, size=len(series))
    return clip01((series + noise) / 100.0) * 100.0

df['Diabetes_risk_score_custom']      = add_noise(df['Diabetes_risk_score_custom'], sigma=2.0)
df['Hypertension_risk_score_custom']  = add_noise(df['Hypertension_risk_score_custom'], sigma=2.5)
df['HeartDisease_risk_score_custom']  = add_noise(df['HeartDisease_risk_score_custom'], sigma=3.5)
df['Obesity_risk_score_custom']       = add_noise(df['Obesity_risk_score_custom'], sigma=1.5)
df['Cholesterol_risk_score_custom']   = add_noise(df['Cholesterol_risk_score_custom'], sigma=1.0)

# Save output
df.to_csv(output_path, index=False)

print("Dataset saved successfully with realistic noise:", output_path)
