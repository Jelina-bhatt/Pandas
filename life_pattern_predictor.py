"""
Life Pattern Predictor – The Personal Data Chronometer
Single-file project using pandas + scikit-learn.

Features:
- Synthetic dataset generator (daily personal metrics)
- Exploratory Data Analysis (summary + plots)
- Feature engineering: datetime features + rolling stats
- Train a regression model (predict life_score 0..100)
- Train a classifier (predict life_pattern category)
- Model evaluation, saving, and single-sample predict function
- Usage examples at the bottom

Requirements (install via pip):
pip install pandas numpy scikit-learn matplotlib joblib seaborn

Run:
python life_pattern_predictor.py
"""

import os
import sys
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump, load

# -----------------------------
# 1) Synthetic dataset generator
# -----------------------------
def generate_synthetic_daily_data(start_date='2023-01-01', days=730, seed=42):
    """
    Generates synthetic personal daily metrics for `days` days starting from start_date.
    Each row represents a day with multiple features and a derived target `life_score` (0-100)
    and `life_pattern` (categorical).
    """
    random.seed(seed)
    np.random.seed(seed)

    start = pd.to_datetime(start_date)
    rows = []
    for i in range(days):
        date = start + pd.Timedelta(days=i)
        # Base rhythms + noise (simulate realistic daily variance)
        weekday = date.weekday()  # 0 Mon ... 6 Sun
        # sleep_hours: weekday slightly less, weekend more
        sleep_hours = max(3.0, min(10.0, np.random.normal(7 + (0.5 if weekday >=5 else -0.2), 0.9)))
        # steps: weekends lower or higher depending on persona — simulate variability
        steps = int(max(500, np.random.normal(7000 + (1000 if weekday >=5 else 0), 2500)))
        # screen_time hours: more on weekdays or variable
        screen_time = max(1.0, min(16.0, np.random.normal(5 + (1.0 if weekday <5 else 2.0), 2.0)))
        # mood: 1-10 scale
        mood = min(10, max(1, np.random.normal(6 + (0.5 if sleep_hours > 7 else -0.5), 1.4)))
        # productivity: 0-10
        productivity = min(10, max(0, np.random.normal(6 + (0.6 if steps > 6000 else -0.7) + (0.4 if sleep_hours>7 else -0.5), 1.8)))
        # social_minutes: phone calls / outings (0-240)
        social_minutes = int(max(0, np.random.normal(30 + (20 if weekday>=5 else -5), 40)))
        # caffeine_intake cups
        caffeine = max(0, min(8, int(np.random.poisson(1.8 + (0.5 if weekday <5 else 0.2)))))
        # exercise minutes
        exercise = int(max(0, np.random.normal(25 + (10 if steps>8000 else 0), 20)))

        # Derive a life_score (0-100) as weighted sum + noise
        # Positive contributors: sleep_hours, mood, productivity, steps, exercise, social
        # Negative contributors: screen_time, caffeine
        score = (
            (sleep_hours - 4) * 4.5 +
            mood * 6.0 +
            productivity * 5.5 +
            (min(15000, steps)/15000)*10 +
            (exercise/120)*8 +
            (np.tanh(social_minutes/120) * 6)
            - (screen_time - 2)*1.8
            - caffeine*1.5
        )
        # Noise and clip
        score = score + np.random.normal(0, 4.5)
        life_score = int(max(0, min(100, round(score, 0))))

        # Categorize life pattern
        if life_score >= 75:
            pattern = 'productive'
        elif life_score >= 55:
            pattern = 'balanced'
        elif life_score >= 35:
            pattern = 'stressed'
        else:
            pattern = 'struggling'

        rows.append({
            'date': date,
            'sleep_hours': round(sleep_hours,2),
            'steps': steps,
            'screen_time': round(screen_time,2),
            'mood': round(mood,2),
            'productivity': round(productivity,2),
            'social_minutes': social_minutes,
            'caffeine': caffeine,
            'exercise_minutes': exercise,
            'life_score': life_score,
            'life_pattern': pattern
        })

    df = pd.DataFrame(rows)
    df.set_index('date', inplace=False)
    return df

# -----------------------------
# 2) EDA & visualization helpers
# -----------------------------
def quick_eda(df, show_plots=True, save_plots=False, plots_dir='plots'):
    print("=== Quick EDA ===")
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nDescribe:\n", df.describe().T)
    print("\nCategory counts:\n", df['life_pattern'].value_counts())

    if show_plots:
        if save_plots and not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plt.figure(figsize=(10,4))
        df['life_score'].rolling(14, min_periods=1).mean().plot(title='14-day rolling life_score')
        plt.xlabel('Date')
        if save_plots: plt.savefig(os.path.join(plots_dir, 'rolling_life_score.png'))
        plt.show()

        # correlation heatmap
        plt.figure(figsize=(10,8))
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Feature Correlation')
        if save_plots: plt.savefig(os.path.join(plots_dir, 'corr_heatmap.png'))
        plt.show()

        # distributions
        numeric = ['sleep_hours','steps','screen_time','mood','productivity','exercise_minutes','life_score']
        df[numeric].hist(bins=20, figsize=(12,8))
        if save_plots: plt.savefig(os.path.join(plots_dir, 'histograms.png'))
        plt.show()

# -----------------------------
# 3) Feature engineering
# -----------------------------
def feature_engineering(df):
    """Create time-based features and rolling statistics. Returns feature DataFrame and label series."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    # Datetime features
    df['day_of_week'] = df.index.dayofweek            # 0-6
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['is_weekend'] = (df['day_of_week'] >=5).astype(int)

    # Rolling features (7 and 14-day windows)
    for col in ['sleep_hours','steps','screen_time','mood','productivity','exercise_minutes','social_minutes','caffeine']:
        df[f'{col}_7_mean'] = df[col].rolling(7, min_periods=1).mean()
        df[f'{col}_14_mean'] = df[col].rolling(14, min_periods=1).mean()
        df[f'{col}_7_std'] = df[col].rolling(7, min_periods=1).std().fillna(0)
    
    # Lag features (yesterday)
    for col in ['sleep_hours','steps','screen_time','mood','productivity']:
        df[f'{col}_lag1'] = df[col].shift(1).fillna(method='bfill')

    # Drop rows with NA if any (should be minimal after bfill)
    df.dropna(inplace=True)

    # Target
    y_reg = df['life_score']
    y_clf = df['life_pattern']

    # Drop direct target columns from X
    X = df.drop(columns=['life_score','life_pattern'])
    return X, y_reg, y_clf

# -----------------------------
# 4) Train models
# -----------------------------
def train_models(X, y_reg, y_clf, random_state=42):
    # Simple numeric preprocessing
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Encode classifier labels
    le = LabelEncoder()
    y_clf_enc = le.fit_transform(y_clf)

    # Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_scaled, y_reg, y_clf_enc, test_size=0.2, random_state=random_state, shuffle=True
    )

    # Regression model
    reg = RandomForestRegressor(n_estimators=200, random_state=random_state)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    rmse = mean_squared_error(y_reg_test, y_reg_pred, squared=False)

    # Classification model
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(X_train, y_clf_train)
    y_clf_pred = clf.predict(X_test)
    acc = accuracy_score(y_clf_test, y_clf_pred)
    clf_report = classification_report(y_clf_test, y_clf_pred, target_names=le.classes_)

    # Cross-val quick checks
    reg_cv = cross_val_score(reg, X_scaled, y_reg, cv=5, scoring='neg_mean_absolute_error')
    clf_cv = cross_val_score(clf, X_scaled, y_clf_enc, cv=5, scoring='accuracy')

    # Pack models + metadata
    models = {
        'regressor': reg,
        'classifier': clf,
        'scaler': scaler,
        'label_encoder': le
    }

    metrics = {
        'regression_mae': mae,
        'regression_rmse': rmse,
        'regression_cv_mae_mean': -reg_cv.mean(),
        'classification_accuracy': acc,
        'classification_cv_acc_mean': clf_cv.mean(),
        'classification_report': clf_report,
        'confusion_matrix': confusion_matrix(y_clf_test, y_clf_pred)
    }

    # Print summary
    print("=== Training Summary ===")
    print(f"Regression MAE: {mae:.3f}, RMSE: {rmse:.3f}, CV MAE (5-fold): {-reg_cv.mean():.3f}")
    print(f"Classification Accuracy: {acc:.3f}, CV Acc (5-fold): {clf_cv.mean():.3f}")
    print("\nClassification Report:\n", clf_report)
    print("\nConfusion Matrix (rows true, cols pred):\n", metrics['confusion_matrix'])

    return models, metrics, (X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test)

# -----------------------------
# 5) Save / Load utilities
# -----------------------------
def save_models(models, folder='saved_models'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    dump(models['regressor'], os.path.join(folder, 'regressor.joblib'))
    dump(models['classifier'], os.path.join(folder, 'classifier.joblib'))
    dump(models['scaler'], os.path.join(folder, 'scaler.joblib'))
    dump(models['label_encoder'], os.path.join(folder, 'label_encoder.joblib'))
    print(f"Models saved to folder: {folder}")

def load_models(folder='saved_models'):
    reg = load(os.path.join(folder, 'regressor.joblib'))
    clf = load(os.path.join(folder, 'classifier.joblib'))
    scaler = load(os.path.join(folder, 'scaler.joblib'))
    le = load(os.path.join(folder, 'label_encoder.joblib'))
    return {'regressor': reg, 'classifier': clf, 'scaler': scaler, 'label_encoder': le}

# -----------------------------
# 6) Single-sample prediction helper
# -----------------------------
def predict_single_day(models, sample_row, feature_columns):
    """
    sample_row: dict or pandas Series containing same feature columns used for X (before scaling).
    feature_columns: list of X columns in same order.
    """
    # Build DataFrame
    x = pd.DataFrame([sample_row], columns=feature_columns)
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    x_scaled = x.copy()
    # scale numeric with loaded scaler (assumes scaler fit on same numeric order)
    x_scaled[numeric_cols] = models['scaler'].transform(x[numeric_cols])

    reg_pred = models['regressor'].predict(x_scaled)[0]
    clf_pred_enc = models['classifier'].predict(x_scaled)[0]
    clf_pred_label = models['label_encoder'].inverse_transform([int(clf_pred_enc)])[0]
    return {'predicted_life_score': float(np.clip(reg_pred,0,100)), 'predicted_pattern': clf_pred_label}

# -----------------------------
# 7) Putting all together: run pipeline
# -----------------------------
def run_full_pipeline(save_models_flag=True, show_eda=True):
    print("Generating synthetic dataset...")
    df = generate_synthetic_daily_data(start_date='2024-01-01', days=700, seed=123)
    # EDA
    if show_eda:
        quick_eda(df, show_plots=True, save_plots=False)

    # Feature engineering
    X, y_reg, y_clf = feature_engineering(df)

    # Train
    models, metrics, splits = train_models(X, y_reg, y_clf)

    # Save
    if save_models_flag:
        save_models(models)

    return models, metrics, X, y_reg, y_clf

# -----------------------------
# 8) Demo usage: create a sample and predict
# -----------------------------
def demo_predict_example(models, X):
    # Take the last row of X as a real-looking sample
    sample = X.tail(1).iloc[0].to_dict()
    print("\nSample input (last day features snippet):")
    for k,v in list(sample.items())[:10]:
        print(f"  {k}: {v}")
    pred = predict_single_day(models, sample, X.columns.tolist())
    print("\nPrediction for sample:")
    print(f" Predicted life_score: {pred['predicted_life_score']:.2f}")
    print(f" Predicted life_pattern: {pred['predicted_pattern']}")

# -----------------------------
# 9) Optional: interactive CLI single-sample creation
# -----------------------------
def build_sample_from_user_input(X_columns):
    """
    Minimal CLI to build a sample row. If running in non-interactive environment, skip.
    """
    print("\nLet's create a custom sample for prediction. Press Enter to use defaults.")
    sample = {}
    # Provide defaults based on medians from columns naming convention
    defaults = {
        'sleep_hours': 7.0,
        'steps': 6000,
        'screen_time': 5.0,
        'mood': 6.0,
        'productivity': 6.0,
        'social_minutes': 30,
        'caffeine': 1,
        'exercise_minutes': 20,
        'day_of_week': 2,
        'day_of_month': 15,
        'month': 6,
        'is_weekend': 0
    }
    for col in X_columns:
        if col in ['date']: 
            continue
        if col in defaults:
            inp = input(f"{col} (default {defaults[col]}): ").strip()
            sample[col] = float(inp) if inp else defaults[col]
        else:
            # for engineered columns like _7_mean etc., fallback to default or 0
            sample[col] = defaults.get(col.split('_')[0], 0)
    return sample

# -----------------------------
# 10) Main entry
# -----------------------------
if __name__ == "__main__":
    # Run pipeline
    models, metrics, X, y_reg, y_clf = run_full_pipeline(save_models_flag=True, show_eda=True)

    # Demo prediction
    demo_predict_example(models, X)

    # Optionally allow user to create a custom sample and predict
    if sys.stdin and sys.stdin.isatty():
        want = input("\nDo you want to create a custom sample and predict? (y/N): ").strip().lower()
        if want == 'y':
            sample = build_sample_from_user_input(X.columns.tolist())
            pred = predict_single_day(models, sample, X.columns.tolist())
            print("\nCustom prediction result:")
            print(f"  Predicted life_score: {pred['predicted_life_score']:.2f}")
            print(f"  Predicted life_pattern: {pred['predicted_pattern']}")
    else:
        print("\nNon-interactive environment detected — skipping custom input.")

    print("\nAll done. Models and scaler/label encoder stored in ./saved_models/")

