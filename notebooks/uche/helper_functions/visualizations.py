import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency, pearsonr

import numpy as np
from scipy.stats import chi2_contingency


def heatmap_numerical(df, tolerance=0.0):
    df_num = df.select_dtypes(include=['number'])
    corr_matrix = df_num.corr()
    filtered_corr = corr_matrix.abs()[(
        corr_matrix.abs() > tolerance) & (corr_matrix != 1)]
    plt.figure(figsize=(15, 10))
    sns.heatmap(filtered_corr, cmap='Reds')
    plt.show()


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k-1)*(r-1)/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt((phi2corr / min((kcorr-1), (rcorr-1))))


def heatmap_categorical_cols(df, return_association_matrix=False):
    df = df.copy()
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    categorical_cols = sorted(categorical_cols.tolist())

    association_matrix = pd.DataFrame(
        index=categorical_cols, columns=categorical_cols)

    for col1 in categorical_cols:
        for col2 in categorical_cols:
            if col1 != col2:  # Avoid calculating association of a column with itself
                contingency = pd.crosstab(df[col1], df[col2])
                association_matrix.loc[col1, col2] = cramers_v(
                    contingency.values)
            else:
                # Set the diagonal to 1 for better visualization
                association_matrix.loc[col1, col2] = 1

    plt.figure(figsize=(20, 20))  # Adjust figure size as needed
    # annot=False is recommended with many columns
    sns.heatmap(association_matrix.astype(float), annot=False, cmap="YlGnBu")
    plt.title("Heatmap of Cramer's V (Pairwise)")
    plt.show()
    return association_matrix if return_association_matrix else None


def heatmap_categorical_numerical_cols(df, return_association_matrix=False):
    df = df.copy()
    df_cat = df.select_dtypes(exclude=['number'])
    df_num = df.select_dtypes(include=['number'])

    # Encode categorical features numerically
    df_cat_encoded = df_cat.apply(lambda col: pd.factorize(col)[0])

    # Compute point biserial correlation for binary categorical features
    corr_matrix = pd.DataFrame(index=df_cat.columns, columns=df_num.columns)

    for cat_col in df_cat.columns:
        for num_col in df_num.columns:
            if df_cat_encoded[cat_col].nunique() == 2:  # Binary categorical
                corr, _ = pointbiserialr(
                    df_cat_encoded[cat_col], df_num[num_col])
            else:  # Categorical with multiple classes
                contingency_table = pd.crosstab(
                    df_cat_encoded[cat_col], df_num[num_col])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                # Cramér's V approximation
                corr = np.sqrt(chi2 / (df_cat_encoded.shape[0] + chi2))

            corr_matrix.loc[cat_col, num_col] = corr

    # Convert to float and plot heatmap
    corr_matrix = corr_matrix.astype(float)
    nrow, ncol = corr_matrix.shape

    plt.figure(figsize=(ncol/1.5, nrow/2.5))
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Correlation Heatmap: Numerical vs. Categorical Features")
    plt.show()
    return corr_matrix if return_association_matrix else None


def heatmap_all_correlations(df):
    df = df.copy()
    df_cat = df.select_dtypes(exclude=['number'])  # Categorical features
    df_num = df.select_dtypes(include=['number'])  # Numerical features

    # Encode categorical features numerically
    df_cat_encoded = df_cat.apply(lambda col: pd.factorize(col)[0])

    # Create an empty correlation matrix
    corr_matrix = pd.DataFrame(index=df.columns, columns=df.columns)

    # Calculate categorical vs numerical correlations
    for cat_col in df_cat.columns:
        for num_col in df_num.columns:
            if df_cat_encoded[cat_col].nunique() == 2:  # Binary categorical
                corr, _ = pointbiserialr(
                    df_cat_encoded[cat_col], df_num[num_col])
            else:  # Categorical with multiple classes
                contingency_table = pd.crosstab(
                    df_cat_encoded[cat_col], df_num[num_col])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                # Cramér's V approximation
                corr = np.sqrt(chi2 / (df_cat_encoded.shape[0] + chi2))
            corr_matrix.loc[cat_col, num_col] = corr
            corr_matrix.loc[num_col, cat_col] = corr

    # Calculate numerical vs numerical correlations
    for num_col_1 in df_num.columns:
        for num_col_2 in df_num.columns:
            if num_col_1 != num_col_2:
                corr, _ = pearsonr(df_num[num_col_1], df_num[num_col_2])
                corr_matrix.loc[num_col_1, num_col_2] = corr
                corr_matrix.loc[num_col_2, num_col_1] = corr

    # Calculate categorical vs categorical correlations using Cramér's V
    for cat_col_1 in df_cat.columns:
        for cat_col_2 in df_cat.columns:
            if cat_col_1 != cat_col_2:
                contingency_table = pd.crosstab(
                    df_cat_encoded[cat_col_1], df_cat_encoded[cat_col_2])
                corr = cramers_v(contingency_table)
                corr_matrix.loc[cat_col_1, cat_col_2] = corr
                corr_matrix.loc[cat_col_2, cat_col_1] = corr

    # Convert to float and return the correlation matrix
    corr_matrix = corr_matrix.astype(float)
    corr_matrix = corr_matrix.astype(float)
    nrow, ncol = corr_matrix.shape

    plt.figure(figsize=(ncol/1.5, nrow/2.5))  # Adjust figure size as needed
    # annot=False is recommended with many columns
    sns.heatmap(corr_matrix.astype(float), annot=False, cmap="coolwarm")
    plt.show()
    return corr_matrix


def group_correlated_features_simple(association_matrix, threshold=0.3):
    """

    """
    corr_matrix = association_matrix.astype(float).abs()
    correlated_groups = {}
    group_number = 1
    assigned_features = set()  # Keep track of assigned features

    for i in range(len(corr_matrix.columns)):
        feature = corr_matrix.columns[i]
        if feature not in assigned_features:
            group = [feature]
            for j in range(i + 1, len(corr_matrix.columns)):  # Avoid duplicates
                other_feature = corr_matrix.columns[j]
                if abs(corr_matrix.iloc[i, j]) > threshold:  # Directly use iloc
                    group.append(other_feature)
                    assigned_features.add(other_feature)
            if group:  # Only add non-empty groups
                correlated_groups[group_number] = group
                group_number += 1
            assigned_features.add(feature)  # Mark current feature as assigned
    return correlated_groups


def numerical_analysis(num_col, df, target):
    """
    Analyzes a numerical feature in relation to a binary target variable.

    - Plots the KDE distributions for target classes (normalized).
    - Tests significance using t-test (if normal) or Mann-Whitney U test.
    """
    # Separate the data based on the target variable
    data_0 = df[df[target] == 0][num_col].dropna()
    data_1 = df[df[target] == 1][num_col].dropna()

    # Determine whether to use a t-test or Mann-Whitney U test
    _, p_normal_0 = stats.shapiro(data_0) if len(data_0) > 3 else (None, 1)
    _, p_normal_1 = stats.shapiro(data_1) if len(data_1) > 3 else (None, 1)

    if p_normal_0 > 0.05 and p_normal_1 > 0.05:  # Both distributions are normal
        stat, p = stats.ttest_ind(
            data_0, data_1, equal_var=False)  # Welch's t-test
        test_used = "T-test"
    else:  # At least one is not normal
        stat, p = stats.mannwhitneyu(data_0, data_1, alternative='two-sided')
        test_used = "Mann-Whitney U test"

    # Interpret significance
    if p < 0.05:
        result = f"{num_col} significantly influences the outcome (p={p:.3f}, {
            test_used})."
        result_color = "maroon"
    else:
        result = f"No significant influence of {
            num_col} on the outcome (p={p:.3f}, {test_used})."
        result_color = "black"

    # Plot the distributions using KDE (Kernel Density Estimation)
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data_0, label="0 (Negative Outcome)",
                color='#00BFFF', fill=True, linewidth=2)
    sns.kdeplot(data_1, label="1 (Positive Outcome)",
                color='#DC143C', fill=True, linewidth=2)

    # Title and labels
    plt.title(f"Distribution of {num_col} by {target} Outcome")
    plt.xlabel(num_col)
    plt.ylabel("Density (Scaled)")

    # Add significance text on the right-hand side, below legend
    plt.text(1.05, 0.5, result, ha="left", va="center", fontsize=10, color=result_color,
             transform=plt.gca().transAxes, wrap=True)

    # Move the legend outside
    plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Show the plot
    plt.tight_layout()
    plt.show()


def categorical_analysis(cat_col, df, target, color=['#1f77b4', '#ff7f0e'], return_result_only=False):
    unique_values = df[cat_col].nunique()

    if unique_values == 1:
        if return_result_only:
            return None
        # Special case when there is only one unique value in the category
        unique_value = df[cat_col].unique()[0]
        target_counts = df[target].value_counts(normalize=True)

        # Plot target distribution for the single category
        target_counts.plot(kind='bar', color=color, figsize=(6, 4))
        plt.title(
            f'Distribution of {target} within {cat_col} = {unique_value}')
        plt.xlabel(target)
        plt.ylabel('Proportion')
        plt.xticks(ticks=[0, 1], labels=[
                   '0 (Negative)', '1 (Positive)'], rotation=0)
        plt.ylim(0, 1)

        # Display category info
        result = f"Only one category ({unique_value}) present in {cat_col}.\nShowing target distribution."
        plt.text(1.05, 0.5, result, ha='left', va='center', fontsize=10, color='black',
                 transform=plt.gca().transAxes, wrap=True)

        plt.show()
        return

    # Perform chi-square test if more than one unique value exists
    contingency_table = pd.crosstab(df[cat_col], df[target])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    cramer_v = np.sqrt(
        chi2 / (df.shape[0] * (min(contingency_table.shape) - 1)))

    if p < 0.05:
        result = f"The {cat_col} category significantly influences the outcome with Cramer_v of: {cramer_v:0.2f} and a p-factor of {p:0.2f}."
        result_color = 'maroon'
    else:
        result = f"No significant influence of {cat_col} category on the outcome. The p-factor is {p:0.2f}."
        result_color = 'black'

    if return_result_only:
        return cramer_v if p < 0.05 else None

    # Calculate proportions
    proportions = df.groupby([cat_col, target]).size().unstack().fillna(0)
    proportions = proportions.div(proportions.sum(axis=1), axis=0)
    proportions = proportions.sort_values(by=[0])

    # Plot proportions
    proportions.plot(kind='bar', stacked=True, color=color, figsize=(10, 6))
    plt.text(1.05, 0.5, result, ha='left', va='center', fontsize=10, color=result_color,
             transform=plt.gca().transAxes, wrap=True)
    plt.title(
        f'Relative Proportions of {target} Outcome per {cat_col.capitalize()} Category')
    plt.xlabel(f'{cat_col.capitalize()}')
    plt.legend(title=target, labels=[
               '0 (Negative)', '1 (Positive)'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
