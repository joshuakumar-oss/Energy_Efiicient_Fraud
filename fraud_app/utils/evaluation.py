from sklearn.metrics import confusion_matrix, classification_report


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_classification_metrics(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)