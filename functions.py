import numpy as np
from numpy import ndarray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import chi2_contingency, mannwhitneyu
import pandas as pd
from typing import List, Dict, Union
from category_encoders import TargetEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
)


def create_numerical_pipeline(
    imputer: TransformerMixin, scaler: TransformerMixin
) -> Pipeline:
    """Creates a pipeline for processing numerical data."""
    return Pipeline([("imputer", imputer), ("scaler", scaler)])


def create_categorical_pipeline(encoder: TransformerMixin) -> Pipeline:
    """Creates a pipeline for processing categorical data."""
    return Pipeline([("encoder", encoder)])


def create_preprocessor(
    numerical_cols: list[str],
    categorical_cols: list[str],
    passthrough_cols: list[str],
    imputer: TransformerMixin,
    scaler: TransformerMixin,
    encoder: TransformerMixin,
) -> ColumnTransformer:
    """Creates a preprocessor for both numerical and categorical data."""

    return ColumnTransformer(
        [
            ("num", create_numerical_pipeline(imputer, scaler), numerical_cols),
            ("cat", create_categorical_pipeline(encoder), categorical_cols),
            ("pass", "passthrough", passthrough_cols),
        ]
    )


def build_pipeline(
    model: BaseEstimator,
    numerical_cols: list[str],
    categorical_cols: list[str],
    passthrough_cols: list[str],
    imputer: TransformerMixin = None,
    scaler: TransformerMixin = None,
    encoder: TransformerMixin = None,
    sampler: Union[TransformerMixin, str] = "passthrough",
) -> ImbPipeline:
    """Builds a complete pipeline including preprocessing, SMOTE (optional), and a model."""

    # Defining default transformers
    default_imputer = SimpleImputer(strategy="mean")
    default_scaler = StandardScaler()
    default_encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore")

    preprocessor = create_preprocessor(
        numerical_cols,
        categorical_cols,
        passthrough_cols,
        imputer=imputer or default_imputer,
        scaler=scaler or default_scaler,
        encoder=encoder or default_encoder,
    )

    return ImbPipeline(
        [
            ("preprocessor", preprocessor),
            ("sampler", sampler),
            ("classifier", model),
        ]
    )


def custom_cross_validation(
    models: list[dict], X_train: pd.DataFrame, y_train: pd.Series, cv_strategy
) -> pd.DataFrame:
    results = []
    """
    Performs cross-validation on a list of machine learning models.
    It evaluates each model's performance using precision,
    recall, ROC AUC,and precision-recall AUC metrics.
    Returns a DataFrame summarizing these metrics for each model.
    """
    for model in models:
        best_estimator = model["Best_Estimator"]
        model_name = model["Model"]

        precision_scores, recall_scores, roc_auc_scores, pr_auc_scores = [], [], [], []

        for train_idx, test_idx in cv_strategy.split(X_train, y_train):
            X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

            best_estimator.fit(X_train_fold, y_train_fold)
            y_pred_fold = best_estimator.predict(X_test_fold)
            y_pred_probs_fold = best_estimator.predict_proba(X_test_fold)[:, 1]

            precision_scores.append(
                precision_score(y_test_fold, y_pred_fold, pos_label=1)
            )
            recall_scores.append(recall_score(y_test_fold, y_pred_fold, pos_label=1))
            roc_auc_scores.append(roc_auc_score(y_test_fold, y_pred_probs_fold))

            precision_fold, recall_fold, _ = precision_recall_curve(
                y_test_fold, y_pred_probs_fold
            )
            pr_auc_scores.append(auc(recall_fold, precision_fold))

        model_result = {
            "Model": model_name,
            "Precision": np.round(np.mean(precision_scores), 3),
            "Precision_Std": np.round(np.std(precision_scores), 3),
            "Recall": np.round(np.mean(recall_scores), 3),
            "Recall_Std": np.round(np.std(recall_scores), 3),
            "ROC_AUC": np.round(np.mean(roc_auc_scores), 3),
            "Precision_Recall_AUC": np.round(np.mean(pr_auc_scores), 3),
        }

        results.append(model_result)

    return pd.DataFrame(results)


def make_common_preprocessing(
    input_imputers: list[TransformerMixin] = None,
    input_encoders: list[TransformerMixin] = None,
    input_scalers: list[TransformerMixin] = None,
) -> dict:
    """Generates a dictionary of common preprocessing options for grid search."""

    default_imputers = [SimpleImputer(strategy="mean"), KNNImputer()]
    default_encoders = [TargetEncoder(), OneHotEncoder(drop="if_binary")]
    default_scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    imputers = input_imputers or default_imputers
    encoders = input_encoders or default_encoders
    scalers = input_scalers or default_scalers

    sampling_techniques = [
        SMOTE(random_state=42),
        RandomUnderSampler(random_state=42),
        "passthrough",
    ]
    return {
        "preprocessor__num__imputer": imputers,
        "preprocessor__num__scaler": scalers,
        "preprocessor__cat__encoder": encoders,
        "sampler": sampling_techniques,
    }


def random_search_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict,
    numerical_cols: list[str],
    categorical_cols: list[str],
    passthrough_cols: list[str],
    param_grid: dict,
    scoring: str,
    cv: int,
) -> list:
    """
    Conducts a randomized search cross-validation for a set of models.
    It optimizes each model's hyperparameters based on the provided training data,
    parameter grid, and scoring metric. Returns a list of dictionaries containing
    the best score, parameters, and estimator for each model.
    """
    results = []
    for model_name, model in models.items():
        pipeline = build_pipeline(
            model, numerical_cols, categorical_cols, passthrough_cols
        )
        params_grid = param_grid.get(model_name, {})
        random_search = RandomizedSearchCV(
            pipeline,
            params_grid,
            n_iter=60,
            cv=cv,
            scoring=scoring,
            random_state=42,
            verbose=False,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)
        results.append(
            {
                "Model": model_name,
                "Best_Score": random_search.best_score_,
                "Best_Params": random_search.best_params_,
                "Best_Estimator": random_search.best_estimator_,
            }
        )
    return results


def multiple_test_chi2(
    df: pd.DataFrame, features: List[str], target: str
) -> Dict[str, List[float]]:
    """Performs chi-square tests for a list of features against a target."""

    results = {}
    for feature in features:
        contingency_table = pd.crosstab(df[feature], df[target])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results[feature] = [p_value, chi2]
    return results


def mutiple_test_mann_whitney(
    df: pd.DataFrame, features: List[str], target: str
) -> Dict[str, float]:
    """Performs Mann-Whitney U tests for a list of features against a target."""
    results = {}
    for feature in features:
        df = df.dropna(subset=[feature])
        group1 = df[df[target] == 0][feature]
        group2 = df[df[target] == 1][feature]
        stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
        results[feature] = p_value
    return results


def confusion_matrix_df(y_test: ndarray, y_preds: ndarray) -> pd.DataFrame:
    conf_matrix = pd.DataFrame(
        confusion_matrix(y_test, y_preds),
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
    return conf_matrix


def extract_feature_importances(model_pipeline: Pipeline) -> pd.DataFrame:
    feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = model_pipeline.named_steps["classifier"].feature_importances_
    feature_importances = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    return feature_importances
