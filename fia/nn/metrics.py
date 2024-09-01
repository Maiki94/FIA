from torchmetrics import (
    Accuracy,
    Recall,
    Precision,
    F1Score,
    AUROC,
    MetricCollection,
)


def get_metrics(num_classes: int):
    accuracy_metrics = get_accuracy_metrics(num_classes=num_classes)
    precision_metrics = get_precision_metrics(num_classes=num_classes)
    f1_score_metrics = get_f1_score_metrics(num_classes=num_classes)
    recall_metrics = get_recall_metrics(num_classes=num_classes)
    auc_metrics = get_auc_metrics(num_classes=num_classes)

    return MetricCollection(
        {
            **accuracy_metrics,
            **recall_metrics,
            **precision_metrics,
            **f1_score_metrics,
            **auc_metrics,
        }
    )


def get_recall_metrics(num_classes: int):
    return {

        "recall_weighted": Recall(
            num_classes=num_classes, average="weighted", task="binary"
        ),
    }


def get_precision_metrics(num_classes: int):
    return {

        "precision_weighted": Precision(
            num_classes=num_classes, average="weighted", task="binary"
        ),
    }


def get_f1_score_metrics(num_classes: int):
    return {

        "f1_score_weighted": F1Score(
            num_classes=num_classes, average="weighted", task="binary"
        ),
    }


def get_accuracy_metrics(num_classes: int):
    return {
        "accuracy_macro": Accuracy(
            num_classes=num_classes, average="macro", task="binary"
        ),
    }


def get_auc_metrics(num_classes: int):
    return {"auroc": AUROC(num_classes=num_classes, task="binary")}
