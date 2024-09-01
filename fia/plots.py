import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc
)


class PerformancePlotter:
    def __init__(self):
        # You could initialize other attributes here if needed
        pass

    def plot_auc_curve(self, y_true, y_probs, ax=None):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        should_display = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            should_display = True
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        if should_display:
            plt.show()

    def plot_precision_recall_curve(self, y_true, y_probs, ax=None):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        should_display = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            should_display = True
        ax.plot(recalls, precisions, label="Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="best")
        ax.set_title("Precision-Recall Curve")
        if should_display:
            plt.show()

    def plot_precision_recall_f1_vs_threshold(self, y_true, y_probs, ax=None):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        should_display = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            should_display = True
        ax.plot(thresholds, f1_scores[:-1], "r-", label="F1-score")
        ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Precision/Recall")
        ax.legend(loc="best")
        ax.set_title("Precision and Recall vs. Threshold")
        ax.grid(True)
        if should_display:
            plt.show()

    def plot_metrics(self, y_true, y_probs):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        self.plot_auc_curve(y_true, y_probs, ax=axs[0])
        self.plot_precision_recall_curve(y_true, y_probs, ax=axs[1])
        self.plot_precision_recall_f1_vs_threshold(y_true, y_probs, ax=axs[2])
        plt.tight_layout()
        plt.show()


class EdaPlotter:
    def __init__(self) -> None:
        pass

    def plot_skewness(self, df):
        # Filter numerical features in the DataFrame
        numerical_features = df.select_dtypes(include=["number"])

        # Calculate skewness of each numerical feature
        skew_values = numerical_features.skew()

        # Create a plot of skewness values
        plt.figure(figsize=(10, 5))
        skew_values.plot(kind="bar")
        plt.title("Skewness of Numerical Features")
        plt.xlabel("Features")
        plt.ylabel("Skewness Value")
        plt.axhline(y=0, color="r", linestyle="-")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        # Return skewness values for reference
        return skew_values

    def plot_numerical_features(self, dataframe):
        df_numerical = dataframe.select_dtypes(include=["int64", "float64"])
        num_cols = len(df_numerical.columns)
        num_rows = (num_cols // 2) + (num_cols % 2)

        fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows))

        for i, feature in enumerate(df_numerical.columns):
            row = i // 2
            col = i % 2

            ax = axes[row, col]
            ax.hist(dataframe[feature].dropna(), bins=30, edgecolor="black")
            ax.set_title(f"Distribution of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")

        # Hide empty subplots if the number of features is odd
        if num_cols % 2 != 0:
            axes[-1, -1].axis("off")

        fig.tight_layout()
        plt.show()

    def plot_categorical_features(self, df):
        categorical_features = df.select_dtypes(include=["object", "category"])

        cat_cols = len(categorical_features.columns)
        cat_rows = (cat_cols // 2) + (cat_cols % 2)

        fig, axes = plt.subplots(cat_rows, 2, figsize=(15, 4 * cat_rows))

        for i, feature in enumerate(categorical_features.columns):
            row = i // 2
            col = i % 2
            ax = axes[row, col]

            value_counts = df[feature].value_counts()
            ax.bar(
                value_counts.index,
                value_counts.values,
                color="skyblue",
                edgecolor="black",
            )
            ax.set_title(f"Countplot of {feature}")
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            ax.tick_params(
                axis="x", rotation=45
            )  # Rotate x-axis labels for better readability if necessary

        # Hide empty subplots if the number of features is odd
        if cat_cols % 2 != 0:
            axes[-1, -1].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_missing_values_proportion(
        self, df: pd.DataFrame, cols_missing_neg1: list[str]
    ):
        # Replace -1 with NaN in the specified columns
        df[cols_missing_neg1] = df[cols_missing_neg1].replace(-1, np.nan)

        # Calculate the percentage of missing values by feature
        null_X = df.isna().sum() / len(df) * 100

        # Plot the missing values
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = (
            null_X.loc[null_X > 0]
            .sort_values()
            .plot(kind="bar", title="Percentage of Missing Values", ax=ax)
        )

        # Annotate the bars with the percentage of missing values
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
                color="red",
            )

        ax.set_ylabel("Missing %")
        ax.set_xlabel("Feature")

        # Remove gridlines from the x-axis
        ax.xaxis.grid(False)

        plt.show()
