import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SalaryPredictor:

    def __init__(self):
        self._model = None
        self._mlb = None
        self._skill_cols = []
        self.models_dir = Path("models/")

    def train(self, df: pd.DataFrame) -> dict:
        logger.info("Training salary predictor...")

        df = df.copy()
        df = df.dropna(subset=["salary_min"])
        df = df[df["salary_min"] > 100000]
        df = df[df["salary_min"] < 50000000]

        if len(df) < 10:
            raise ValueError(f"Not enough salary data: {len(df)} rows")

        # Encode skills
        self._mlb = MultiLabelBinarizer()
        skill_matrix = self._mlb.fit_transform(df["skills"])
        self._skill_cols = [
            f"skill_{s.replace(' ', '_').lower()}"
            for s in self._mlb.classes_
        ]
        skill_df = pd.DataFrame(skill_matrix, columns=self._skill_cols, index=df.index)

        X = pd.concat([
            df[["experience_min"]].fillna(0).rename(columns={"experience_min": "experience"}),
            df[["city", "employment_type"]].fillna("unknown"),
            skill_df,
        ], axis=1)

        y = np.log1p(df["salary_min"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        categorical = ["city", "employment_type"]
        numeric = ["experience"] + self._skill_cols

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ])

        self._model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
            )),
        ])

        self._model.fit(X_train, y_train)

        y_pred = self._model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

        logger.info("Salary model trained: R2=%.3f MAE=%.0f", r2, mae)
        return {"r2": round(r2, 3), "mae": round(mae, 0)}

    def predict(self, skills: list, experience: float = 2.0,
                city: str = "Bengaluru", employment_type: str = "full-time") -> dict:

        if self._model is None or self._mlb is None:
            raise RuntimeError("Model not trained yet.")

        skill_vec = self._mlb.transform([skills])
        skill_df = pd.DataFrame(skill_vec, columns=self._skill_cols)

        X = pd.concat([
            pd.DataFrame({
                "experience": [experience],
                "city": [city],
                "employment_type": [employment_type],
            }),
            skill_df,
        ], axis=1)

        log_pred = self._model.predict(X)[0]
        median = float(np.expm1(log_pred))

        return {
            "predicted_median": round(median, -3),
            "lower_bound": round(median * 0.80, -3),
            "upper_bound": round(median * 1.20, -3),
        }

    def save(self):
        self.models_dir.mkdir(exist_ok=True)
        path = self.models_dir / "salary_predictor.pkl"
        joblib.dump({
            "model": self._model,
            "mlb": self._mlb,
            "skill_cols": self._skill_cols,
        }, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls):
        instance = cls()
        path = instance.models_dir / "salary_predictor.pkl"
        data = joblib.load(path)
        instance._model = data["model"]
        instance._mlb = data["mlb"]
        instance._skill_cols = data["skill_cols"]
        return instance