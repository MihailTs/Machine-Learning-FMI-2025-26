import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def main():
    df_diabetes = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week04\\ForHome\\Data\\diabetes_clean.csv"
    )
    df_diabetes = df_diabetes[(df_diabetes.bmi != 0) & (df_diabetes.glucose != 0)]

    y = df_diabetes['diabetes']
    X = df_diabetes.drop('diabetes', axis=1)
    X = X[X.bmi != 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    reg = LogisticRegression()
    knn = KNeighborsClassifier()
    ridge = RidgeClassifier()

    search_space_reg = {
        'fit_intercept': [True, False],
        'tol': [1e-6, 1e-5, 1e-4, 1e-2, 1e-1],
    }

    search_space_knn = {
        'n_neighbors': [5, 10, 15, 20, 50, 100],
        'metric': ['manhattan', 'minkowski']
    }

    search_space_ridge = {
        'alpha': [1, 2, 5, 10],
        'max_iter': [50, 100, 500, 1000]
    }

    scoring = ['accuracy', 'f1', 'roc_auc']

    models = [reg, knn, ridge]
    search_spaces = [search_space_reg, search_space_knn, search_space_ridge]

    feature_sets = [['glucose', 'pregnancies'], 
                    ['glucose'],
                    ['glucose', 'diastolic'],
                    ['glucose', 'triceps'],
                    ['glucose', 'insulin'],
                    ['bmi', 'dpf'],
                    ['glucose', 'age'],
                    ['glucose', 'pregnancies', 'diastolic'],
                    ['glucose', 'pregnancies', 'triceps'],
                    ['glucose', 'pregnancies', 'insulin'],
                    ['glucose', 'insulin', 'diastolic'],
                    ['glucose', 'bmi', 'diastolic'],
                    ['glucose', 'pregnancies', 'diastolic', 'insulin'],
                    ['glucose', 'pregnancies', 'diastolic', 'insulin', 'age'],
                    ]

    for model, search_space in zip(models, search_spaces):
        rows = []
        for feature_set in feature_sets:
            GS = GridSearchCV(estimator=model,
                            param_grid=search_space,
                            scoring=scoring,
                            refit='f1',
                            cv=5,
                            verbose=4)
            GS.fit(X_train[feature_set], y_train)
            best_model = GS.best_estimator_

            plt.figure(figsize=(4, 4))
            ConfusionMatrixDisplay.from_estimator(
                best_model, X_test[feature_set], y_test,
                cmap='Blues', colorbar=False
            )
            plt.title(f"{type(model).__name__} - {feature_set}")
            plt.savefig(f"{type(model).__name__.replace(' ', '_')}_{feature_set}")

            best_idx = GS.best_index_
            best_scores = {}
            for metric in GS.scoring:
                key = f'mean_test_{metric}'
                best_scores[metric] = GS.cv_results_[key][best_idx]

            row = {'features': str(feature_set)}
            row.update(GS.best_params_)
            row.update(best_scores)
            rows.append(row)

        df = pd.DataFrame(rows)
        with pd.ExcelWriter('model_report_diabetes.xlsx', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=type(model).__name__, index=False)

    majority_class = y_train.value_counts().idxmax()
    y_pred = np.full(shape=y_test.shape, fill_value=majority_class)

    accuracy = sum([1 if i == j else 0 for i, j in zip(y_test, y_pred)]) / len(y_test)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    print("Baseline model metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    knn = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
    knn.fit(X_train, y_train)

    y_score = knn.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'KNN (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for best model')
    plt.legend(loc='lower right')
    plt.savefig('Best_model_ROC')

if __name__ == "__main__":
    main()