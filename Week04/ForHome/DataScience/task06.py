import sys
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from ml_lib.metrics import r2_score

def main():
    df_diabetes = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week04\\ForHome\\Data\\diabetes_clean.csv"
    )
    df_diabetes = df_diabetes[(df_diabetes.bmi != 0) & (df_diabetes.glucose != 0)]

    y = df_diabetes['glucose']
    X = df_diabetes.drop('glucose', axis=1)
    X = X[X.bmi != 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

    feature_sets = [['pregnancies', 'insulin'], 
                    ['diastolic'],
                    ['triceps'],
                    ['insulin', 'age'],
                    ['age', 'bmi'],
                    ['triceps', 'insulin'],
                    ['triceps', 'diastolic'],
                    ['bmi', 'triceps', 'age'],
                    ['bmi', 'insulin'],
                    ['insulin', 'diastolic'],
                    ['bmi', 'diastolic', 'insulin', 'age'],
                    ['pregnancies', 'diastolic', 'insulin', 'age', 'bmi'],
                    ['pregnancies', 'diastolic', 'insulin', 'age', 'bmi', 'dpf', 'triceps']
                    ]
    
    alphas = [1, 5, 10, 50]

    rows = []
    results = []
    for a in alphas:
        reg = Ridge(alpha=a)
        for feature_set in feature_sets:
            reg.fit(X_train[feature_set], y_train)
            
            y_pred = reg.predict(X_test[feature_set])
            mse = np.square(np.subtract(y_test, y_pred)).mean()
            rmse = np.sqrt(np.square(np.subtract(y_test, y_pred)).mean())
            r2_test = reg.score(X_test[feature_set], y_test)
            r_2_adj = 1 - (1 - r2_test) * (len(y_test) - 1) / (len(y_test) -
                                                        X_test[feature_set].shape[1] - 1)
            row = {'features': str(feature_set), 'alpha': a, 'mse': mse, 'rmse': rmse, 'r2': r2_test, 'r2adj': r_2_adj}
            rows.append(row)

            r2_train = reg.score(X_train[feature_set], y_train)
            exp_name = f"Ridge_alpha={a}_{str(feature_set)}"
            results.append([exp_name, r2_test, r2_train])

    df = pd.DataFrame(rows)
    with pd.ExcelWriter('model_report_glucose.xlsx', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name='Glucose prediction', index=False)

    # after assesing the best model with best_features and alpha = 1
    best_features = ['pregnancies', 'diastolic', 'insulin', 'age', 'bmi', 'dpf', 'triceps']
    alpha = 1

    best_reg = Ridge(alpha=alpha)
    best_reg.fit(X_train[best_features], y_train)

    y_pred_best = best_reg.predict(X_test[best_features])

    residuals = y_test - y_pred_best
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='blue', bins=20)
    plt.title('Residual Distribution (Best Ridge Model)')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('residual_distribution.png')

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_pred_best, y=residuals, color='green', alpha=0.6)
    plt.axhline(0, color='red', linestyle='-', linewidth=2)
    plt.title('Residuals vs Predicted Values (Best Ridge Model)')
    plt.xlabel('Predicted Glucose')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig('residuals_vs_predicted.png')

    # coefficients importance table
    coef_df = pd.DataFrame({
        'features': best_features,
        'coefficients': best_reg.coef_
    })

    coef_df['abs_coefficients'] = coef_df['coefficients'].abs()
    coef_df = coef_df.sort_values(by='abs_coefficients', ascending=False).reset_index(drop=True)
    coef_df['relative_importance'] = 100 * coef_df['abs_coefficients'] / coef_df['abs_coefficients'].sum()

    with pd.ExcelWriter('model_report_glucose.xlsx', mode='a', if_sheet_exists='replace') as writer:
        coef_df.to_excel(writer, sheet_name='Feature importance', index=False)

    # model bar chart
    experiment_results_T = experiment_results.transpose()
    df = pd.DataFrame({
        "Model": experiment_results_T[0],
        "Test R2": experiment_results_T[1],
        "Train R2": experiment_results_T[2]
    })

    df_melted = df.melt(id_vars="Model", 
                        value_vars=["Test R2", "Train R2"],
                        var_name="Dataset", 
                        value_name="R2 Score")

    plt.figure(figsize=(20, max(6, len(df) * 0.4)))
    sns.set_style("whitegrid")
    sns.barplot(data=df_melted, 
                x="R2 Score", 
                y="Model", 
                hue="Dataset")
    plt.title("Train and Test R² Scores by Model", fontsize=14, weight='bold')
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout(pad=2)
    plt.legend(loc="lower right")
    plt.savefig("bar_chart.png", dpi=150, bbox_inches='tight')
    plt.close()

    experiment_results = np.array(results, dtype=object)
    experiment_results_T = experiment_results.transpose()
    df = pd.DataFrame({
        "Model": experiment_results_T[0],
        "Test R2": experiment_results_T[1],
        "Train R2": experiment_results_T[2]
    })

    df_melted = df.melt(id_vars="Model", 
                        value_vars=["Test R2", "Train R2"],
                        var_name="Dataset", 
                        value_name="R2 Score")

    plt.figure(figsize=(20, max(6, len(df) * 0.4)))
    sns.set_style("whitegrid")
    sns.barplot(data=df_melted, 
                x="R2 Score", 
                y="Model", 
                hue="Dataset")
    plt.title("Train and Test R² Scores by Model", fontsize=14, weight='bold')
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.tight_layout(pad=2)
    plt.legend(loc="lower right")
    plt.savefig("bar_chart_glucose_regression.png", dpi=150, bbox_inches='tight')
    plt.close()

    # baseline model
    y_pred = np.full(fill_value=df_diabetes['glucose'].mean(), shape=(len(y_test),))
    mse = np.square(np.subtract(y_test, y_pred)).mean()
    rmse = np.sqrt(np.square(np.subtract(y_test, y_pred)).mean())
    r2 = r2_score(y_test, y_pred)
    r2_adj = 1 - (1-r2_score(y_test, y_pred)) * (len(y_test)-1)/(len(y_test)-X.shape[1]-1)
    
    print(f'mse: {mse}')
    print(f'rmse: {rmse}')
    print(f'r2: {r2}')
    print(f'r2_adj: {r2_adj}')

if __name__ == "__main__":
    main()