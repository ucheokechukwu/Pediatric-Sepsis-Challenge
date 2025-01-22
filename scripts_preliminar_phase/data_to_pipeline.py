import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


target = "inhospital_mortality"
DATA_PATH = ""
n_components = 3


def get_cleaned_data(data_path=DATA_PATH):
    """input is the DATA_PATH which is a constant
    collects the cleaned_data from the given path
    and returns the data after filling missing values
    and the lists of categorical and numerical columns"""

    # loading cleaned data

    data = pd.read_csv(f"{data_path}cleaned_data.csv")
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

    # return values are: data after cleaning and filling missing values, the list of numerical columns
    return data, numerical_cols, categorical_cols


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
    y = pd.read_csv(f"{DATA_PATH}cleaned_data.csv")[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Verify stratification
    print("Class distribution in training data:")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution in test data:")
    print(y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def model_training(data, numerical_cols, categorical_cols):

    # get features and labels
    X_train, _, y_train, _ = get_train_test_data(data)

    # get pipeline components
    preprocessor = get_preprocessor(data, numerical_cols, categorical_cols)
    pca = PCA(n_components=n_components)
    classifier = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=2,
            min_samples_split=2,
            random_state=42),
        learning_rate=0.01,
        n_estimators=50,
        random_state=42,
        algorithm='SAMME')

    # Create pipeline
    oversampler = SMOTEENN(random_state=42)
    steps = [('preprocessor', preprocessor)]
    steps.append(('oversample', oversampler))
    steps.extend([('pca', pca), ('classifier', classifier)])
    pipeline = ImbPipeline(steps)

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    return pipeline


def model_training_main(cleaned_data=None, verbose=True):
    # prepare data
    print("preparing data now...") if verbose else ""
    data, numerical_cols, categorical_cols = get_cleaned_data()
    print("preparing data completed...") if verbose else ""

    # training model
    print("training model now...") if verbose else ""
    pipeline = model_training(data, numerical_cols, categorical_cols)
    print(pipeline)
    print("\n\n\ntraining model complete...") if verbose else ""
    return pipeline, None


def main():
    model_training_main()


if __name__ == "__main__":
    main()
