import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():
    df_churn = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week02\\ForHome\\Data\\telecom_churn_clean.csv") 
    y = df_churn['churn']
    X_train, X_test, y_train, y_test = train_test_split(df_churn, y, test_size=0.2, random_state=21, stratify=y)
    pair_plot = sns.pairplot(X_train, hue='churn')
    pair_plot.figure.savefig('pairplot.png')
    

if __name__ == "__main__":
    main()