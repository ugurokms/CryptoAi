from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Sklearn_Classifier():
    def __init__(self, sklearn_classifier) -> None:
        self.model = sklearn_classifier

    def train_test(self, X_train, y_train, X_test, y_test):
        self.model = self.model.fit(X=X_train, y=y_train)
        y_pred = self.model.predict(X_test)
        f1_s = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
        print(f"F1-score macro: {f1_s}")
        accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=False)/y_test.shape[0]
        print(f"Accuracy: {accuracy}")
        disp = ConfusionMatrixDisplay(confusion_matrix(y_true=y_test, y_pred=y_pred))
        disp.plot()