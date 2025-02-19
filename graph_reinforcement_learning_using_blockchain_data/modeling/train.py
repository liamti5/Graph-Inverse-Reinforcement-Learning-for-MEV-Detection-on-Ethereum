import mlflow
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()

app = typer.Typer()

class RandomForestTrainer:
    def grid_search(self, features_to_scale: list):
        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), features_to_scale)
            ],
            remainder='passthrough'
        )

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        param_distributions = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt", None],
        }

        grid_search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=param_distributions,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=2,
            random_state=42,
        )
        return grid_search

    def train(self, X_train, X_test, y_train, y_test, grid_search, experiment_name):
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            best_model = grid_search.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            train_precision = precision_score(y_train, y_train_pred)
            test_precision = precision_score(y_test, y_test_pred)

            train_recall = recall_score(y_train, y_train_pred)
            test_recall = recall_score(y_test, y_test_pred)

            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            train_roc_auc = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
            test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

            mlflow.log_param("n_iter", grid_search.n_iter)
            for param, value in grid_search.best_params_.items():
                mlflow.log_param(param, value)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("test_f1", test_f1)
            mlflow.log_metric("train_roc_auc", train_roc_auc)
            mlflow.log_metric("test_roc_auc", test_roc_auc)

            cm = confusion_matrix(y_test, y_test_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
            plt.close()

            report = classification_report(y_test, y_test_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv("classification_report.csv", index=True)
            mlflow.log_artifact("classification_report.csv")

            mlflow.sklearn.log_model(best_model, "random_forest_model")

            return best_model


@app.command()
def main():
    print("main not implemented")


if __name__ == "__main__":
    app()
