import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REAL_CSV_PATH = DATA_DIR / "bedtime_screentime_sleep_debt.csv"
SYNTHETIC_CSV_PATH = DATA_DIR / "synthetic_bedtime_screentime.csv"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "meta.json"

CAT_FEATURES = [
    "gender",
    "occupation_type",
    "chronotype",
    "primary_bedtime_app",
]
NUMERIC_FEATURES = [
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
    "sleep_latency_min",
]
FEATURE_COLS = CAT_FEATURES + NUMERIC_FEATURES
TARGET = "sleep_debt_category"
INTEGER_FEATURES = {
    "age",
    "bedtime_phone_minutes",
    "screen_brightness_pct",
    "blue_light_filter_active",
    "caffeine_post_5pm_mg",
    "physical_activity_min",
}

LABELS = {
    "gender": "Gênero",
    "occupation_type": "Ocupação",
    "chronotype": "Cronotipo",
    "primary_bedtime_app": "App principal antes de dormir",
    "age": "Idade",
    "bedtime_phone_minutes": "Minutos no celular antes de dormir",
    "screen_brightness_pct": "Brilho da tela (%)",
    "blue_light_filter_active": "Filtro de luz azul ativo",
    "caffeine_post_5pm_mg": "Cafeína após 17h (mg)",
    "physical_activity_min": "Atividade física (min)",
    "sleep_latency_min": "Latência do sono (min)",
}

VALUE_LABELS = {
    "Female": "Feminino",
    "Male": "Masculino",
    "Non-Binary": "Não binário",
    "Corporate 9-to-5": "Corporativo 9 às 5",
    "Remote Tech": "Tecnologia remota",
    "Healthcare / Shift Worker": "Saúde / trabalho em turnos",
    "Student": "Estudante",
    "Freelance / Creative": "Freelance / criativo",
    "Intermediate": "Intermediário",
    "Night Owl": "Coruja (noturno)",
    "Morning Lark": "Cotovia (matutino)",
    "TikTok / Reels": "TikTok / Reels",
    "YouTube": "YouTube",
    "Instagram / Reddit": "Instagram / Reddit",
    "Messaging / Chat": "Mensagens",
    "News / Reading": "Notícias / leitura",
    "Streaming (Netflix/Hulu)": "Streaming (Netflix/Hulu)",
    "Optimal Recovery": "Recuperação ideal",
    "Mild Deficit": "Déficit leve",
    "Moderate Debt": "Dívida moderada",
    "Severe Sleep Debt": "Dívida severa de sono",
}


def display_label(value: str) -> str:
    return VALUE_LABELS.get(value, value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_training_frame(n_rows: int = 8500, seed: int = 42) -> pd.DataFrame:
    """Synthetic stand-in for the Kaggle CSV, using categories seen in the notebook."""
    rng = np.random.default_rng(seed)
    gender = rng.choice(
        ["Female", "Male", "Non-Binary"], n_rows, p=[0.51, 0.46, 0.03]
    )
    occupation = rng.choice(
        [
            "Corporate 9-to-5",
            "Remote Tech",
            "Healthcare / Shift Worker",
            "Student",
        ],
        n_rows,
        p=[0.33, 0.25, 0.21, 0.21],
    )
    chronotype = rng.choice(
        ["Intermediate", "Night Owl", "Morning Lark"],
        n_rows,
        p=[0.46, 0.29, 0.25],
    )
    app = rng.choice(
        ["TikTok / Reels", "YouTube", "Instagram / Reddit", "Messaging", "Reading"],
        n_rows,
        p=[0.26, 0.23, 0.22, 0.17, 0.12],
    )
    age = np.clip(rng.normal(32, 11, n_rows), 18, 65).round().astype(int)
    phone = np.clip(rng.lognormal(3.85, 0.55, n_rows), 1, 180).round().astype(int)
    brightness = np.clip(rng.normal(55, 18, n_rows), 10, 100).round().astype(int)
    blue = rng.choice([0, 1], n_rows, p=[0.532, 0.468])
    caffeine = np.clip(rng.exponential(28, n_rows), 0, 250).round().astype(int)
    activity = np.clip(rng.normal(36, 22, n_rows), 0, 112).round().astype(int)
    latency = np.clip(8 + 0.38 * phone + rng.normal(0, 10, n_rows), 6, 123.3)
    latency = np.round(latency, 1)

    app_weight = {
        "TikTok / Reels": 1.25,
        "YouTube": 0.75,
        "Instagram / Reddit": 0.95,
        "Messaging": 0.35,
        "Reading": 0.05,
    }
    chrono_weight = {"Night Owl": 1.15, "Intermediate": 0.25, "Morning Lark": -0.7}
    occupation_weight = {
        "Healthcare / Shift Worker": 0.45,
        "Student": 0.3,
        "Corporate 9-to-5": 0.15,
        "Remote Tech": 0.0,
    }
    risk = (
        (phone - 40) / 35
        + ((brightness - 50) / 70) * np.where(blue == 1, 0.4, 1.2)
        + caffeine / 160
        + (latency - 20) / 28
        - activity / 140
        + np.array([app_weight[value] for value in app])
        + np.array([chrono_weight[value] for value in chronotype])
        + np.array([occupation_weight[value] for value in occupation])
        + rng.normal(0, 0.45, n_rows)
    )
    low, mid, high = np.quantile(risk, [0.16, 0.38, 0.915])
    category = np.where(
        risk <= low,
        "Optimal Recovery",
        np.where(
            risk <= mid,
            "Mild Deficit",
            np.where(risk <= high, "Moderate Debt", "Severe Sleep Debt"),
        ),
    )
    return pd.DataFrame(
        {
            "user_id": [f"USR-{index:05d}" for index in range(1, n_rows + 1)],
            "age": age,
            "gender": gender,
            "occupation_type": occupation,
            "chronotype": chronotype,
            "bedtime_phone_minutes": phone,
            "primary_bedtime_app": app,
            "screen_brightness_pct": brightness,
            "blue_light_filter_active": blue,
            "caffeine_post_5pm_mg": caffeine,
            "physical_activity_min": activity,
            "sleep_latency_min": latency,
            TARGET: category,
        }
    )


def _read_training_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in FEATURE_COLS + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError("O CSV não tem as colunas do modelo: " + ", ".join(missing))
    return frame


def ensure_dataset() -> tuple[pd.DataFrame, str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if REAL_CSV_PATH.exists():
        return _read_training_frame(REAL_CSV_PATH), "csv", REAL_CSV_PATH
    if not SYNTHETIC_CSV_PATH.exists():
        generate_training_frame().to_csv(SYNTHETIC_CSV_PATH, index=False)
    return _read_training_frame(SYNTHETIC_CSV_PATH), "synthetic", SYNTHETIC_CSV_PATH


def build_schema(frame: pd.DataFrame) -> list[dict]:
    features = []
    for column in FEATURE_COLS:
        if column in CAT_FEATURES:
            options = sorted(frame[column].astype(str).unique().tolist())
            features.append(
                {
                    "name": column,
                    "label": LABELS[column],
                    "type": "categorical",
                    "options": [
                        {"value": value, "label": display_label(value)}
                        for value in options
                    ],
                }
            )
            continue
        if column == "blue_light_filter_active":
            features.append(
                {
                    "name": column,
                    "label": LABELS[column],
                    "type": "boolean",
                    "default": int(round(float(frame[column].median()))),
                }
            )
            continue
        series = frame[column]
        integer = column in INTEGER_FEATURES
        default = float(series.median())
        features.append(
            {
                "name": column,
                "label": LABELS[column],
                "type": "integer" if integer else "number",
                "min": int(series.min()) if integer else float(series.min()),
                "max": int(series.max()) if integer else float(series.max()),
                "default": int(round(default)) if integer else round(default, 1),
                "step": 1 if integer else 0.1,
            }
        )
    return features


def train_bundle(frame: pd.DataFrame, source: str) -> dict:
    model_df = frame.copy()
    encoders = {}
    for column in CAT_FEATURES:
        encoder = LabelEncoder()
        model_df[column] = encoder.fit_transform(model_df[column].astype(str))
        encoders[column] = encoder

    target_encoder = LabelEncoder()
    model_df[TARGET] = target_encoder.fit_transform(model_df[TARGET].astype(str))

    features = model_df[FEATURE_COLS]
    target = model_df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        target_names=target_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    importance = sorted(
        (
            {
                "name": name,
                "label": LABELS[name],
                "importance": round(float(value), 4),
            }
            for name, value in zip(FEATURE_COLS, model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "model": model,
        "encoders": encoders,
        "target_encoder": target_encoder,
        "feature_cols": FEATURE_COLS,
        "schema": build_schema(frame),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
            "report": report,
            "feature_importance": importance,
            "n_rows": int(len(frame)),
            "data_source": source,
            "classes": [
                {"value": label, "label": display_label(label)}
                for label in target_encoder.classes_
            ],
        },
    }


class ModelService:
    def __init__(self) -> None:
        self.bundle = None

    def load(self) -> dict:
        frame, source, csv_path = ensure_dataset()
        digest = _file_sha256(csv_path)
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        if MODEL_PATH.exists() and META_PATH.exists():
            meta = json.loads(META_PATH.read_text())
            if meta.get("sha256") == digest and meta.get("source") == source:
                self.bundle = joblib.load(MODEL_PATH)
                return self.bundle
        self.bundle = train_bundle(frame, source)
        joblib.dump(self.bundle, MODEL_PATH)
        META_PATH.write_text(json.dumps({"sha256": digest, "source": source}))
        return self.bundle

    def schema(self) -> dict:
        bundle = self.bundle or self.load()
        return {
            "target": TARGET,
            "target_label": "Categoria de dívida de sono",
            "data_source": bundle["metrics"]["data_source"],
            "features": bundle["schema"],
        }

    def metrics(self) -> dict:
        bundle = self.bundle or self.load()
        return bundle["metrics"]

    def predict(self, payload: dict) -> dict:
        bundle = self.bundle or self.load()
        schema = {feature["name"]: feature for feature in bundle["schema"]}
        missing = [name for name in FEATURE_COLS if name not in payload]
        unknown = [name for name in payload if name not in schema]
        if missing or unknown:
            raise ValueError(
                json.dumps(
                    {"missing": missing, "unknown": unknown},
                    ensure_ascii=False,
                )
            )

        row = {}
        errors = []
        for name in FEATURE_COLS:
            spec = schema[name]
            value = payload[name]
            if spec["type"] == "categorical":
                allowed = {option["value"] for option in spec["options"]}
                if value not in allowed:
                    errors.append(f"{name} deve ser um de: {sorted(allowed)}")
                    continue
                row[name] = int(bundle["encoders"][name].transform([str(value)])[0])
                continue
            if spec["type"] == "boolean":
                if value not in (0, 1, True, False):
                    errors.append(f"{name} deve ser 0 ou 1")
                    continue
                row[name] = int(bool(value))
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"{name} deve ser numérico")
                continue
            if number < spec["min"] or number > spec["max"]:
                errors.append(
                    f"{name} deve estar entre {spec['min']} e {spec['max']}"
                )
                continue
            row[name] = int(number) if spec["type"] == "integer" else number

        if errors:
            raise ValueError(json.dumps({"errors": errors}, ensure_ascii=False))

        frame = pd.DataFrame([row], columns=FEATURE_COLS)
        model = bundle["model"]
        encoded = int(model.predict(frame)[0])
        category = bundle["target_encoder"].inverse_transform([encoded])[0]
        probabilities = model.predict_proba(frame)[0]
        classes = bundle["target_encoder"].classes_
        ranked = sorted(
            (
                {
                    "value": label,
                    "label": display_label(label),
                    "probability": round(float(probability), 4),
                }
                for label, probability in zip(classes, probabilities)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )
        return {
            "sleep_debt_category": category,
            "label": display_label(category),
            "probabilities": ranked,
        }


service = ModelService()
