from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def plot_feature_distribution(data, feature, target):
    """
    Plots the distribution of a numerical feature for both target=True and target=False.

    Parameters:
    - data: DataFrame containing the data.
    - feature: The name of the feature (column) to plot.
    - target: The name of the target column (binary values for True/False).
    """
    # Set Seaborn style (optional)
    sns.set_style("darkgrid")

    # Filter the data based on the target values
    data_true = data[data[target] == True]
    data_false = data[data[target] == False]

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot KDE for target=True
    sns.kdeplot(data_true[feature], color='blue',
                label='target=True', fill=True)

    # Plot KDE for target=False
    sns.kdeplot(data_false[feature], color='red',
                label='target=False', fill=True)

    # Title with skewness values for both target categories
    plt.title(f"{feature} | Skewness (True): {round(data_true[feature].skew(
    ), 2)} | Skewness (False): {round(data_false[feature].skew(), 2)}")
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming `data` is your DataFrame, and 'feature_name' is the feature to be plotted
# plot_feature_distribution(data, 'hematocrit_gpdl_adm', target)


def check_feature_significance(data, feature, target):
    """
    Check the statistical significance of a feature with respect to a binary target.

    Parameters:
    - data: DataFrame containing the data.
    - feature: The name of the feature (column) to check.
    - target: The name of the target column (binary values for True/False).

    Returns:
    - A dictionary with results for point-biserial correlation, t-test, logistic regression, and Pearson correlation.
    """
    # Ensure feature and target columns are numeric or binary (for tests)
    data = data[[feature, target]].dropna()

    # Point-Biserial Correlation
    point_biserial_corr, p_value_corr = pointbiserialr(
        data[feature], data[target])

    # T-test
    data_true = data[data[target] == True][feature]
    data_false = data[data[target] == False][feature]
    t_stat, p_value_ttest = ttest_ind(data_true, data_false, equal_var=False)

    # Logistic Regression (with balanced class weight to handle unbalanced target)
    X = data[[feature]]
    y = data[target]

    # Split into training and testing sets (we'll fit the model on the training set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # Initialize logistic regression with balanced class weights
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)

    # Model results
    # exponentiated coefficient gives the odds ratio
    odds_ratio = np.exp(model.coef_[0][0])
    p_value_log_reg = model.pvalues_[0] if hasattr(model, 'pvalues_') else None

    # Pearson Correlation
    pearson_corr, p_value_pearson = pearsonr(data[feature], data[target])

    # Collecting results
    results = {
        "Point-Biserial Correlation": point_biserial_corr,
        "p-value (Point-Biserial)": p_value_corr,
        "T-test Statistic": t_stat,
        "p-value (T-test)": p_value_ttest,
        "Odds Ratio (Logistic Regression)": odds_ratio,
        "p-value (Logistic Regression)": p_value_log_reg,
        "Pearson Correlation": pearson_corr,
        "p-value (Pearson)": p_value_pearson
    }

    # Print results
    print(f"Point-Biserial Correlation: {point_biserial_corr} "
          f"(p-value: {p_value_corr}) - "
          f"{'Statistically Significant' if p_value_corr < 0.05 else 'Not Statistically Significant'}")

    print(f"T-test Statistic: {t_stat} "
          f"(p-value: {p_value_ttest}) - "
          f"{'Statistically Significant' if p_value_ttest < 0.05 else 'Not Statistically Significant'}")

    print(f"Odds Ratio (Logistic Regression): {odds_ratio} "
          f"(p-value: {p_value_log_reg}) - "
          f"{'Statistically Significant' if p_value_log_reg is not None and p_value_log_reg < 0.05 else 'Not Statistically Significant'}")

    print(f"Pearson Correlation: {pearson_corr} "
          f"(p-value: {p_value_pearson}) - "
          f"{'Statistically Significant' if p_value_pearson < 0.05 else 'Not Statistically Significant'}")

    return results
# results = check_feature_significance(data, 'hematocrit_gpdl_adm', target)


def feature_inspection(data, f, target):
    filtered_data = data.dropna(subset=[f])
    plot_feature_distribution(filtered_data, f, target)
    return check_feature_significance(filtered_data, f, target)
