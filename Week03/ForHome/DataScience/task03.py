import itertools
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ml_lib.metrics import r2_score


def experiment(X_train, y_train, X_test, y_test, reg):
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = np.square(np.subtract(y_test, y_pred)).mean()
    rmse = np.sqrt(np.square(np.subtract(y_test, y_pred)).mean())
    r_2 = reg.score(X_test, y_test)
    r_2_adj = 1 - (1 - r_2) * (len(y_test) - 1) / (len(y_test) -
                                                   X_test.shape[1] - 1)
    print(X_train.columns)
    print(f'mse: {mse}, rmse: {rmse}, R2: {r_2}, R2_adj: {r_2_adj}\n')

    # if X_train.shape[1] == 1:
    #     if X_train.columns[0] != 'tv':
    #         return
    #     x_col = X_train.columns[0]

    #     sorted_idx = np.argsort(X_test[x_col])
    #     X_test_sorted = X_test.iloc[sorted_idx]
    #     y_pred_sorted = y_pred[sorted_idx]

    #     plt.figure(figsize=(8, 5))
    #     sns.scatterplot(x=X_train[x_col],
    #                     y=y_train,
    #                     color='blue',
    #                     label='Train data')
    #     sns.scatterplot(x=X_test[x_col],
    #                     y=y_test,
    #                     color='green',
    #                     label='Test data')
    #     sns.lineplot(x=X_test_sorted[x_col],
    #                  y=y_pred_sorted,
    #                  color='red',
    #                  linewidth=2,
    #                  label='Regression line')
    #     plt.xlabel(x_col)
    #     plt.ylabel('Target')
    #     plt.title(f'Linear Regression: {x_col} vs Target')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.savefig('tv_regression.png', dpi=300)

    #     # residuals plot
    #     residuals = y_test - y_pred
    #     plt.figure(figsize=(8,5))
    #     sns.scatterplot(x=y_pred, y=residuals, color='purple')
    #     plt.axhline(0, color='red', linestyle='--')
    #     plt.xlabel('Predicted Values')
    #     plt.ylabel('Residuals')
    #     plt.title('Residual Plot')
    #     plt.grid(True)
    #     plt.savefig('tv_regression_residuals.png', dpi=300)

def main():
    df_advert = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week03\\ForHome\\Data\\advertising_and_sales_clean.csv"
    )

    y = df_advert['sales']
    df_advert = df_advert.drop('sales', axis=1)
    dummies = pd.get_dummies(df_advert['influencer'],
                             prefix='influencer',
                             drop_first=True,
                             dtype=int)
    df_advert = pd.concat([df_advert, dummies], axis=1)
    df_advert = df_advert.drop('influencer', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_advert,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=21)

    # baseline model
    bsln = np.full(len(y_test), y_test.mean())
    mse = np.square(np.subtract(y_test, bsln)).mean()
    rmse = np.sqrt(np.square(np.subtract(y_test, bsln)).mean())
    r_2 = r2_score(bsln, y_test)
    r_2_adj = 1 - (1 - r_2) * (len(y_test) - 1) / (len(y_test) -
                                                   X_test.shape[1] - 1)
    print('BASELINE:')
    print(f'mse: {mse}, rmse: {rmse}, R2: {r_2}, R2_adj: {r_2_adj}\n')

    reg = LinearRegression()

    for column in X_train.columns:
        print(column)
        experiment(X_train=X_train[[column]],
                   y_train=y_train,
                   X_test=X_test[[column]],
                   y_test=y_test,
                   reg=reg)

    # X_train, X_test, y_train, y_test = train_test_split(df_advert,
    #                                                     y,
    #                                                     test_size=0.3,
    #                                                     random_state=21)
    # for column1, column2 in itertools.product(X_train.columns, X_train.columns):
    #     print(column1, column2)
    #     experiment(X_train=X_train[[column1, column2]],
    #                y_train=y_train,
    #                X_test=X_test[[column1, column2]],
    #                y_test=y_test,
    #                reg=reg)

    # X_train, X_test, y_train, y_test = train_test_split(df_advert,
    #                                                     y,
    #                                                     test_size=0.20,
    #                                                     random_state=21)
    # experiment(X_train=X_train[['tv']],
    #                y_train=y_train,
    #                X_test=X_test[['tv']],
    #                y_test=y_test,
    #                reg=reg)

    # all features
    # experiment(X_train=X_train,
    #            y_train=y_train,
    #            X_test=X_test,
    #            y_test=y_test,
    #            reg=reg)

    # experiment(X_train=X_train[['tv','social_media','radio']],
    #            y_train=y_train,
    #            X_test=X_test[['tv','social_media','radio']],
    #            y_test=y_test,
    #            reg=reg)

    # experiment(X_train=X_train[['tv','social_media','radio','influencer_Mega']],
    #            y_train=y_train,
    #            X_test=X_test[['tv','social_media','radio','influencer_Mega']],
    #            y_test=y_test,
    #            reg=reg)

    # experiment(X_train=X_train[['tv','social_media','radio','influencer_Micro']],
    #            y_train=y_train,
    #            X_test=X_test[['tv','social_media','radio','influencer_Micro']],
    #            y_test=y_test,
    #            reg=reg)

    # experiment(X_train=X_train[['tv','social_media','influencer_Mega']],
    #            y_train=y_train,
    #            X_test=X_test[['tv','social_media','influencer_Mega']],
    #            y_test=y_test,
    #            reg=reg)


if __name__ == "__main__":
    main()
