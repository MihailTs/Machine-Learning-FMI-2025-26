import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import ml_lib.neighbors as nb
from ml_lib.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    df_churn = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week02\\ForHome\\Data\\telecom_churn_clean.csv")
    y = df_churn['churn']
    X_train, X_test, y_train, y_test = train_test_split(df_churn,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=21,
                                                        stratify=y)
    n_neighbors = range(60)
    accuracy_scores = []
    for i in range(1, 61):
        knn = nb.KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)
        accuracy_scores.append(knn.score(y_test, prediction))

    plot = sns.lineplot(x=n_neighbors, y=accuracy_scores, color='blue')
    plt.title('Accuracy vs Number of Neighbors for the top model')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()