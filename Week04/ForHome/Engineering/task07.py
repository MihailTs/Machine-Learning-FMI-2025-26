import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from ml_lib.linear_model import LogisticRegression, Ridge
from ml_lib.model_selection import train_test_split
from ml_lib.metrics import f1_score, accuracy_score

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

    best_features = ['glucose', 'bmi', 'diastolic']
    reg.fit(X_train=X_train[best_features], y_train=y_train)
    y_pred = reg.predict(X_test=X_test[best_features])
    acuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('Logistic regression')
    print(f'features: {best_features}')
    print(f'accuracy: {acuracy}')
    print(f'f1: {f1}')
    
    # from task 6
    rdg = Ridge()

    best_features = ['pregnancies', 'diastolic', 'insulin', 'age', 'bmi', 'dpf', 'triceps']
    y = df_diabetes['glucose']
    X = df_diabetes.drop('glucose', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    rdg.alpha = 1
    rdg.fit(X_train=X_train[best_features], y_train=y_train)
    r2 = rdg.score(X_test[best_features], y_test)
    print('\nRidge regression')
    print(f'features: {best_features}')
    print(f'r2: {r2}')

if __name__ == "__main__":
    main()