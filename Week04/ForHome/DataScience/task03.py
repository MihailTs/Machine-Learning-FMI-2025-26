import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def main():
    df_diabetes = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week04\\ForHome\\Data\\diabetes_clean.csv"
    )

    numeric_summary = df_diabetes.describe(percentiles=[0.25, 0.5, 0.75]).T

    numeric_summary['missing'] = df_diabetes.isna().sum()
    numeric_summary['unique'] = df_diabetes.nunique()

    numeric_summary.to_excel("data_audit3.xlsx")

    y = df_diabetes['diabetes']
    X = df_diabetes.drop('diabetes', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    X_train_with_target = X_train.copy()
    X_train_with_target['diabetes'] = y_train
    pair_plot = sns.pairplot(X_train_with_target, hue='diabetes')
    pair_plot.figure.savefig('pairplot.png')

    with pd.ExcelWriter('data_audit_diabetes.xlsx',
                    mode='a') as writer:  
        for column in df_diabetes.columns:
            df = df_diabetes[column].value_counts().reset_index()
            df.columns = ['Values', 'Count']
            df = df.sort_values('Values')
            df.to_excel(writer, sheet_name=column, index=False)

if __name__ == "__main__":
    main()