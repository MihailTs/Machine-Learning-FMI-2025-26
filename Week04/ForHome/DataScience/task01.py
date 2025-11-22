import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_validate

def experiment(clf, scoring, X, y, cv, exp_name=""):
    exp = f'{type(clf).__name__},cross_val={cv},alpha={clf.alpha},{str(list(X.columns)).replace('\'', '').replace(' ','')}'
    results = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_train_score=True)
    print(f"Mean R2: {results['test_r2'].mean():.4f}")
    print(f"Mean MSE: {-results['test_mse'].mean():.4f}\n")

    # Compute averages
    mean_train_r2 = results["train_r2"].mean()
    mean_test_r2 = results["test_r2"].mean()
    mean_train_mse = -results["train_mse"].mean()
    mean_test_mse = -results["test_mse"].mean()

    # Organize results for plotting
    metrics = ["R²", "R²", "MSE", "MSE"]
    dataset = ["Train", "Test", "Train", "Test"]
    values = [mean_train_r2, mean_test_r2, mean_train_mse, mean_test_mse]

    df_plot = pd.DataFrame({
        "Metric": metrics,
        "Dataset": dataset,
        "Value": values
    })

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    sns.barplot(data=df_plot[df_plot["Metric"] == "R²"], x="Value", y="Dataset", ax=axes[0], palette="viridis")
    axes[0].set_title(f"{exp_name} - R² Scores")

    sns.barplot(data=df_plot[df_plot["Metric"] == "MSE"], x="Value", y="Dataset", ax=axes[1], palette="viridis")
    axes[1].set_title(f"{exp_name} - MSE Scores")

    plt.tight_layout()
    plt.savefig(f"{exp_name}_split_bar_chart.png", dpi=120)
    plt.close()

    return np.array([exp, mean_test_r2, mean_train_r2], dtype=object)

def main():
    df_advert = pd.read_csv(
        "D:\\UNIVERSITY\\ML\\Week04\\ForHome\\Data\\advertising_and_sales_clean.csv"
    )

    # Separate target and features
    y = df_advert['sales']
    df_advert = df_advert.drop('sales', axis=1)

    dummies = pd.get_dummies(df_advert['influencer'],
                             prefix='influencer',
                             drop_first=True,
                             dtype=int)
    df_advert = pd.concat([df_advert.drop('influencer', axis=1), dummies], axis=1)
    
    clf = Ridge(alpha=1.0)

    scoring = {
        'r2': 'r2',
        'mse': make_scorer(mean_squared_error, greater_is_better=False)
    }

    for column in df_advert.columns:
        res = experiment(clf, scoring, df_advert[[column]], y, cv=10)
        experiment_results = res
        
    clf = Ridge(alpha=10.0)
    res = experiment(clf, scoring, df_advert[['tv']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv']], y, cv=5)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])

    clf = Ridge(alpha=5.0)
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=5)
    experiment_results = np.vstack([experiment_results, res])

    clf = Lasso(alpha=1.0)
    for column in df_advert.columns:
        res = experiment(clf, scoring, df_advert[[column]], y, cv=10)
        experiment_results = np.vstack([experiment_results, res])

    clf = Lasso(alpha=5.0)
    res = experiment(clf, scoring, df_advert[['tv']], y, cv=5)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=5)
    experiment_results = np.vstack([experiment_results, res])
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])

    clf = Lasso(alpha=10.0)
    res = experiment(clf, scoring, df_advert[['tv', 'radio', 'social_media', 'influencer_Mega', 'influencer_Micro', 'influencer_Nano']], y, cv=10)
    experiment_results = np.vstack([experiment_results, res])

    print(experiment_results)

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
    # Create horizontal bar plot with separate colors for Train/Test
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

if __name__ == "__main__":
    main()
