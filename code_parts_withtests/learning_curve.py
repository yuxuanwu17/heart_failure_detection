import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # %%

    heart_data = pd.read_csv("/home/yuxuan/kaggle/heart_failure_clinical_records_dataset.csv")

    X = heart_data.iloc[:, 0:11]
    y = heart_data['DEATH_EVENT']

    selected_feature = ['serum_creatinine', 'age', 'ejection_fraction', 'creatinine_phosphokinase']
    X_processed = X[selected_feature]
    X_processed = StandardScaler().fit_transform(X_processed)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=2)

    model = SVC(kernel='rbf', random_state=1, C=10, gamma=0.01)
    model.fit(X_train, y_train)

    loo = LeaveOneOut()
    train_size, train_acc, test_acc = learning_curve(model, X_train, y_train, cv=loo)
    learn_df = pd.DataFrame({"Train_size": train_size, "Train_Accuracy": train_acc.mean(axis=1),
                             "Test_Accuracy": test_acc.mean(axis=1)}).melt(id_vars="Train_size")
    sns.lineplot(x="Train_size", y="value", data=learn_df, hue="variable")
    plt.title("Learning Curve")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == '__main__':
    main()
