from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    df_churn = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week02\\ForHome\\Data\\telecom_churn_clean.csv")
    X = df_churn[['account_length', 'customer_service_calls']]
    y = df_churn['churn']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=20,
                                                        stratify=y)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train, y_train)

    X_new = np.array([[30.0, 17.5], [107.0, 24.1], [213.0, 10.9]])

    predictions = knn.predict(X_new)
    print(f'{predictions=}')

if __name__ == "__main__":
    main()