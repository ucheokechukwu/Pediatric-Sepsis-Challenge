import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

from rich import print, pretty
pretty.install()
DATA_PATH = "../../"


def get_cleaned_data(data_path=DATA_PATH):
    """input is the DATA_PATH which is a constant
    collects the cleaned_data from the given path
    and returns the data after filling missing values
    and the lists of categorical and numerical columns"""

    # loading cleaned data

    data = pd.read_csv(f"{data_path}cleaned_dataset.csv")
    target = "inhospital_mortality"  # save target column
    data.drop(columns=[target], inplace=True)

    # fill in missing values using 'most frequent' for categorical
    # and 'median' for numerical values

    print(f"Missing values before imputation: {data.isna().sum().sum()}")
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(exclude=['number']).columns

    # Create imputers
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # Impute missing values for numerical and categorical columns
    if not numerical_cols.empty:
        data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])
    if not categorical_cols.empty:
        data[categorical_cols] = cat_imputer.fit_transform(
            data[categorical_cols])
    print("Filled out missing values.", end=" ")
    print(f"Missing values after imputation: {data.isna().sum().sum()}")

    # return values are: data after cleaning and filling missing values, the list of numerical columns
    return data, numerical_cols, categorical_cols


def pie_chart(sorted_contrib, dim, threshold=0.5):
    # Use the top_features_to_50_percent function to select features
    selected_features, _ = top_features_to_50_percent(
        sorted_contrib, threshold)

    # Create labels: selected features retain their labels, others are grouped as "Others"
    top_contrib = sorted_contrib.loc[selected_features]
    other_contrib = sorted_contrib.drop(selected_features).sum()

    labels = list(top_contrib.index) + ["Others"]
    values = list(top_contrib.values) + [other_contrib]

    # Pie chart for contributions
    plt.figure(figsize=(10, 10))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140,
            # Hide labels if too many
            labeldistance=None if len(labels) > 10 else 1.1)
    plt.title(f"Contributions to {dim} (Top {threshold*100:.0f}%)")
    plt.axis('equal')
    plt.legend(labels, loc="upper right", bbox_to_anchor=(
        1.1, 1))  # Legend for visibility
    plt.show()


def top_features_to_50_percent(sorted_contrib, threshold=0.5):
    cumulative_sum = 0
    selected_features = []

    for feature, value in sorted_contrib.items():
        if cumulative_sum + value > threshold:
            break
        cumulative_sum += value
        selected_features.append(feature)

    return selected_features, cumulative_sum


def diverging_bar_plot(sorted_contrib, dim):
    # Separate positive and negative contributions
    pos_contrib = sorted_contrib[sorted_contrib > 0]
    neg_contrib = sorted_contrib[sorted_contrib < 0]

    # Combine contributions for the plot
    combined_contrib = pd.concat([neg_contrib, pos_contrib]).sort_values()

    # Create a figure
    fig = go.Figure()

    # Add negative contributions
    fig.add_trace(go.Bar(
        y=combined_contrib.index,
        x=combined_contrib[combined_contrib < 0],
        orientation='h',
        name='Negative Contributions',
        marker=dict(color='red')
    ))

    # Add positive contributions
    fig.add_trace(go.Bar(
        y=combined_contrib.index,
        x=combined_contrib[combined_contrib > 0],
        orientation='h',
        name='Positive Contributions',
        marker=dict(color='blue')
    ))

    # Add vertical lines at 0, 0.2, and -0.2
    fig.add_vline(x=0, line=dict(color='black', width=2, dash='dash'))
    fig.add_vline(x=0.2, line=dict(color='gray', width=2, dash='dash'))
    fig.add_vline(x=-0.2, line=dict(color='gray', width=2, dash='dash'))

    # Update layout to include axis labels and title
    fig.update_layout(
        title=f"Contributions to {dim}",
        xaxis_title="Contribution Value",
        yaxis_title="Features",
        barmode='stack',
        template='plotly_white',
        autosize=True,
        showlegend=False,
        height=800
    )

    # Show the plot
    fig.show()


def pca_3D_visualization(pca_data):
    """Visualize the PCA components in 3 dimensions
    """
    import plotly.express as px
    import pandas as pd
    target = "inhospital_mortality"
    labels = pd.read_csv("../../cleaned_dataset.csv")[target]

    # Assuming `pca_data` is a NumPy array with your PCA components (n_samples x 3)
    # and `labels` contains your binary class labels

    # Convert to a Pandas DataFrame for easier handling
    df = pd.DataFrame({
        'PC1': pca_data[:, 0],  # Principal Component 1
        'PC2': pca_data[:, 1],  # Principal Component 2
        'PC3': pca_data[:, 2],  # Principal Component 3
        'Label': labels         # Class labels
    })

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Label',          # Color points by class labels
        color_continuous_scale='Viridis',  # Choose a color scale
        symbol='Label',         # Optional: Different markers for classes
        title='Interactive 3D PCA Visualization'
    )

    # Customize the layout (optional)
    fig.update_layout(
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)  # Reduce margins for better view
    )

    # Show the figure
    fig.show()


"""perform PCA on numerical columns to reduce to 3 dimensions"""

N_COMPONENTS = 3

# Perform PCA on numerical columns


def get_pca(data, numerical_cols=None, categorical_cols=None, n_components=3, visualize=True):
    # Validate inputs
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

    # Define full pipeline including PCA
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components))
    ])

    # Fit and transform data using the pipeline
    pca_result = pipeline.fit_transform(data)

    # Extract PCA model and explained variance
    pca = pipeline.named_steps['pca']

    # Create a DataFrame for PCA components
    feature_names = (
        numerical_cols.tolist() +
        list(preprocessor.named_transformers_[
             'cat'].get_feature_names_out(categorical_cols))
    )
    pca_components_df = pd.DataFrame(pca.components_.T, index=feature_names)

    # Sort contributions for each principal component
    sorted_pca_contributions = {
        f"Principal Component {pc+1}": pca_components_df[pc].sort_values(ascending=False)
        for pc in range(pca_components_df.shape[1])
    }

    # Visualize sorted contributions
    if visualize:
        for pc, sorted_contrib in sorted_pca_contributions.items():
            print(f"\n{pc} (Sorted Contributions):")
            print(sorted_contrib)
            diverging_bar_plot(sorted_contrib, pc)
        pca_3D_visualization(pca_result)

    return pca_result, pca_components_df, pca
