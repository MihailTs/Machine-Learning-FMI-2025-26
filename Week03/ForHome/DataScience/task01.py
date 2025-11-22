import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def main():
    df_advert = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week03\\ForHome\\Data\\advertising_and_sales_clean.csv"
    )

    dummies = pd.get_dummies(df_advert['influencer'], prefix='influencer', drop_first=True, dtype=int)
    df_advert = pd.concat([df_advert, dummies], axis=1)
    df_advert = df_advert.drop('influencer',axis=1)

    numeric_summary = df_advert.describe(percentiles=[0.25, 0.5, 0.75]).T

    numeric_summary['missing'] = df_advert.isna().sum()
    numeric_summary['unique'] = df_advert.nunique()

    # numeric_summary.to_excel("data_audit2.xlsx")

    # with pd.ExcelWriter('data_audit.xlsx',
    #                 mode='a') as writer:  
    #     for column in df_advert.columns:
    #         df = df_advert[column].value_counts().reset_index()
    #         df.columns = ['Values', 'Count']
    #         df = df.sort_values('Values')
    #         df.to_excel(writer, sheet_name=column, index=False)

    y = df_advert['sales']
    X = df_advert.drop('sales', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    X_train_with_target = X_train.copy()
    X_train_with_target['sales'] = y_train
    X_train_with_target['sales_category'] = pd.qcut(y_train, q=3, labels=['Low', 'Medium', 'High'])

    pair_plot = sns.pairplot(X_train_with_target, hue='sales_category')
    pair_plot.figure.savefig('pairplot.png')

if __name__ == "__main__":
    main()
