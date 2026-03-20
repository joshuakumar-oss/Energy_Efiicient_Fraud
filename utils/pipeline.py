import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - dependency may be optional at runtime
    XGBClassifier = None


REQUIRED_TARGET = "is_fraud"
DROP_COLUMNS = ["trans_date_trans_time", "merchant", "category", "first", "last"]


def add_time_features(df):
    df = df.copy()

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day"] = df["trans_date_trans_time"].dt.day
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def preprocess_data(df):
    processed = add_time_features(df)
    X = processed.drop(columns=DROP_COLUMNS + [REQUIRED_TARGET], errors="ignore")
    y = processed[REQUIRED_TARGET]
    X = X.select_dtypes(include="number").fillna(0)
    return X, y


def align_feature_sets(train_df, test_df):
    return train_df.align(test_df, join="outer", axis=1, fill_value=0)


def prepare_dataset_pair(train_df, test_df):
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)
    X_train, X_test = align_feature_sets(X_train, X_test)
    return X_train, y_train, X_test, y_test


def sample_dataset(features, target, max_rows):
    sample_size = min(len(features), max_rows)
    sampled_features = features.sample(sample_size, random_state=42)
    sampled_target = target.loc[sampled_features.index]
    return sampled_features, sampled_target


def get_stage2_model():
    if XGBClassifier is not None:
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=120,
            max_depth=5,
            learning_rate=0.08,
        )

    return RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
    )


def train_two_stage_model(X_train, y_train):
    stage1 = LogisticRegression(max_iter=1000)
    stage1.fit(X_train, y_train)

    stage2 = get_stage2_model()
    try:
        stage2.fit(X_train, y_train)
    except Exception:
        stage2 = RandomForestClassifier(
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
        )
        stage2.fit(X_train, y_train)

    return stage1, stage2


def predict_two_stage(stage1, stage2, X_test):
    stage1_pred = stage1.predict(X_test)
    suspicious_idx = stage1_pred == 1
    final_pred = stage1_pred.copy()

    if suspicious_idx.any():
        stage2_pred = stage2.predict(X_test[suspicious_idx])
        final_pred[suspicious_idx] = stage2_pred

    return final_pred, suspicious_idx


def carbon_footprint():
    try:
        path = "emissions_logs/emissions.csv"

        if not os.path.exists(path):
            return {"energy": "Run Carbon Tracking First", "carbon": "Run Carbon Tracking First"}

        df = pd.read_csv(path)
        if df.empty:
            return {"energy": "No Data", "carbon": "No Data"}

        last = df.iloc[-1]
        energy = last.get("energy_consumed", 0)
        emissions = last.get("emissions", 0)

        return {
            "energy": f"{round(energy, 4)} kWh",
            "carbon": f"{round(emissions, 4)} kg CO2",
        }
    except Exception as exc:
        return {"energy": "Error", "carbon": str(exc)}


def get_carbon_data():
    path = "emissions_logs/emissions.csv"

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    return df


def sustainability_score(energy, emissions):
    try:
        score = 100 - (energy * 100 + emissions * 500)
        return max(min(int(score), 100), 0)
    except Exception:
        return 0


def model_carbon_comparison(res_df):
    base_energy = 0.001
    model_energy = []

    for model in res_df["Model"]:
        if "Logistic" in model:
            energy = base_energy * 0.5
        elif "Decision Tree" in model:
            energy = base_energy * 0.8
        elif "Random Forest" in model:
            energy = base_energy * 1.2
        elif "XGBoost" in model:
            energy = base_energy * 1.5
        else:
            energy = base_energy

        carbon = energy * 0.71
        model_energy.append((energy, carbon))

    res_df["Energy (kWh)"] = [round(x[0], 5) for x in model_energy]
    res_df["CO2 (kg)"] = [round(x[1], 5) for x in model_energy]
    return res_df
