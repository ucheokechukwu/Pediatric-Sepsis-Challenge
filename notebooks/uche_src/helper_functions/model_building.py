import inspect

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn.metrics import classification_report, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from helper_functions.get_name import get_name

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

target = "inhospital_mortality"
DATA_PATH = "../../"
n_components = 3


def get_preprocessor(data, numerical_cols=None, categorical_cols=None):

    if numerical_cols is None:
        numerical_cols = data.select_dtypes(
            include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = data.select_dtypes(
            include=['object', 'category']).columns.tolist()

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def get_train_test_data(data, test_size=0.2):
    # Split data into training and test sets
    X = data.copy()
    y = pd.read_csv(f"{DATA_PATH}cleaned_dataset.csv")[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Verify stratification
    print("Class distribution in training data:")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution in test data:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def evaluate_methods(data, preprocessor, pca, classifier, model_name=None, pca_after_smote=True):
    """
    Evaluates different methods of handling imbalanced data with class weighting.

    Parameters:
        preprocessor: Data preprocessing pipeline or transformer.
        pca: PCA object for dimensionality reduction.
        classifier: Base classifier (e.g., RandomForestClassifier or SVC).

    Returns:
        results: Dictionary with precision and recall for class 0 and 1.
    """

    X_train, X_test, y_train, y_test = get_train_test_data(data)
    # Oversampling and hybrid methods to try
    methods = {
        "None (Class Weight Only)": None,
        "SMOTE": SMOTE(random_state=42),
        "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
        "ADASYN": ADASYN(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42)
    }

    results = {}

    for method_name, oversampler in methods.items():
        print(f"Evaluating: {method_name}")

        # Create pipeline
        steps = [('preprocessor', preprocessor)]

        if pca_after_smote:
            if oversampler is not None:
                steps.append(('oversample', oversampler))
            if pca is not None:
                steps.extend([('pca', pca), ('classifier', classifier)])
            else:
                steps.extend([('classifier', classifier)])
        else:
            if pca is not None:
                steps.append(('pca', pca))
            if oversampler is not None:
                steps.extend([('oversample', oversampler),
                             ('classifier', classifier)])
            else:
                steps.extend([('classifier', classifier)])

        pipeline = ImbPipeline(steps)

        # Fit pipeline
        pipeline.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipeline.predict(X_test)

        # Calculate precision and recall for both classes
        precision_class_0 = precision_score(y_test, y_pred, pos_label=0)
        precision_class_1 = precision_score(y_test, y_pred, pos_label=1)
        recall_class_0 = recall_score(y_test, y_pred, pos_label=0)
        recall_class_1 = recall_score(y_test, y_pred, pos_label=1)

        # Store results
        results[method_name] = {
            "precision_0": precision_class_0,
            "precision_1": precision_class_1,
            "recall_0": recall_class_0,
            "recall_1": recall_class_1
        }

    # Plot results
    plot_results(results, model_name)

    return results


def plot_results(results, model_name):
    """
    Plots precision and recall for each method.

    Parameters:
        results: Dictionary with precision and recall values.
    """
    methods = list(results.keys())
    precision_0 = [results[method]["precision_0"] for method in methods]
    precision_1 = [results[method]["precision_1"] for method in methods]
    recall_0 = [results[method]["recall_0"] for method in methods]
    recall_1 = [results[method]["recall_1"] for method in methods]

    x = np.arange(len(methods))
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, precision_0, width,
           label='Precision Class 0', color='blue')
    ax.bar(x - 0.5*width, precision_1, width,
           label='Precision Class 1', color='cyan')
    ax.bar(x + 0.5*width, recall_0, width,
           label='Recall Class 0', color='green')
    ax.bar(x + 1.5*width, recall_1, width,
           label='Recall Class 1', color='lime')

    # Add labels and legend
    ax.set_xlabel('Methods')
    ax.set_ylabel('Scores')
    ax.set_title(
        f'{model_name}: Precision and Recall for Different Class Balancing Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def plot_comparison_results(results):
    """
    Plots a comparison of precision and recall metrics for Random Forest, SVC, and XGBoost models.

    Parameters:
        results: a list of tuples of the form (model_name, model_results)
        results_rf: Dictionary of results for Random Forest.
        results_svc: Dictionary of results for SVC.
        results_xgb: Dictionary of results for XGBoost.
    """
    # Combine results into a DataFrame

    models = {k: v for (k, v) in results}

    combined_results = []
    for model_name, results in models.items():
        for method, metrics in results.items():
            combined_results.append({
                'Model': model_name,
                'Method': method,
                'Precision_0': metrics['precision_0'],
                'Precision_1': metrics['precision_1'],
                'Recall_0': metrics['recall_0'],
                'Recall_1': metrics['recall_1']
            })

    df = pd.DataFrame(combined_results)

    # Create Plotly bar chart
    fig = go.Figure()

    # Add bars for each metric
    for metric in ['Precision_0', 'Precision_1', 'Recall_0', 'Recall_1']:
        for model in df['Model'].unique():
            filtered_df = df[df['Model'] == model]
            fig.add_trace(go.Bar(
                x=filtered_df['Method'],
                y=filtered_df[metric],
                name=f"{metric} ({model})",
                text=[f"{val:.2f}" for val in filtered_df[metric]],
                textposition='auto'
            ))

    # Customize layout
    fig.update_layout(
        title='Comparison of Models on Imbalanced Data',
        xaxis=dict(title='Oversampling Method'),
        yaxis=dict(title='Score'),
        barmode='group',  # Grouped bar chart
        legend_title='Metrics and Models',
        height=600,
        width=1000
    )

    # Add interactive toggle buttons
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="All Metrics",
                         method="update",
                         args=[{"visible": [True] * len(fig.data)},
                               {"title": "All Metrics"}]),
                    dict(label="Precision Class 0",
                         method="update",
                         args=[{"visible": [i % 4 == 0 for i in range(len(fig.data))]},
                               {"title": "Precision Class 0"}]),
                    dict(label="Precision Class 1",
                         method="update",
                         args=[{"visible": [i % 4 == 1 for i in range(len(fig.data))]},
                               {"title": "Precision Class 1"}]),
                    dict(label="Recall Class 0",
                         method="update",
                         args=[{"visible": [i % 4 == 2 for i in range(len(fig.data))]},
                               {"title": "Recall Class 0"}]),
                    dict(label="Recall Class 1",
                         method="update",
                         args=[{"visible": [i % 4 == 3 for i in range(len(fig.data))]},
                               {"title": "Recall Class 1"}])
                ],
                direction="down",
                showactive=True
            )
        ]
    )
    # Dynamically set visibility for all traces except "Recall Class 1"
    # fig.update_traces(visible=[i % 4 == 3 for i in range(len(fig.data))])
    # Show plot
    fig.show()
