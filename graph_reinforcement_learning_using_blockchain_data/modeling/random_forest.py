import mlflow
import pandas as pd
import seaborn as sns
import typer
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from graph_reinforcement_learning_using_blockchain_data import config

mlflow.set_tracking_uri(uri=config.MLFLOW_TRACKING_URI)

from graph_reinforcement_learning_using_blockchain_data import config

config.load_dotenv()

app = typer.Typer()


class RandomForestTrainer:
    def grid_search(self, features_to_scale: list):
        preprocessor = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), features_to_scale)], remainder="passthrough"
        )

        full_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=2)),
            ]
        )

        param_distributions = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt", None],
        }

        randomized_search = RandomizedSearchCV(
            estimator=full_pipeline,
            param_distributions=param_distributions,
            n_iter=20,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=2),
            n_jobs=-1,
            verbose=2,
            random_state=2,
        )
        return randomized_search

    def train(self, X_train, X_test, y_train, y_test, grid_search, experiment_name):
        mlflow.sklearn.autolog()
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            best_model = grid_search.fit(X_train, y_train)

            y_test_pred = best_model.predict(X_test)

            test_accuracy, test_precision, test_recall, test_f1 = self.test_metrics(
                y_test, y_test_pred
            )
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1", test_f1)

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

            report_df = self.classification_report(y_test, y_test_pred)
            report_df.to_csv("classification_report.csv", index=True)
            mlflow.log_artifact("classification_report.csv")

        return best_model

    def test_metrics(self, y_test, y_test_pred):
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        return test_accuracy, test_precision, test_recall, test_f1

    def classification_report(self, y_test, y_test_pred):
        report = classification_report(y_test, y_test_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        return report_df


@app.command()
def main():
    print("main not implemented")


if __name__ == "__main__":
    app()
