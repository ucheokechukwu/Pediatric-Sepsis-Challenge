#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.

import sys
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import mne
import pandas as pd

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

#
################################################################################


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.


def train_challenge_model(data_folder, model_folder, verbose=False):
    input_file = f"{data_folder}/SyntheticData_training.csv"
    cleaned_data = data_cleaning_main(input_file, verbose)
    prediction_model, imputer = model_training_main(cleaned_data, verbose)

    # remove the cleaned_data.csv file
    os.remove(f"{DATA_PATH}cleaned_data.csv")

    # Save the models.
    save_challenge_model(model_folder, imputer, prediction_model)

    if verbose >= 1:
        print('Done!')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.


def load_challenge_model(model_folder, verbose):
    print('Loading the model...')
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)


def run_challenge_model(model, data_folder, verbose):
    imputer = model['imputer']
    prediction_model = model['prediction_model']

    # Load data.
    patient_ids, data, label, features = load_challenge_data(data_folder)

    data = pd.get_dummies(data)

    # Impute missing data.
    data_imputed = imputer.transform(data)

    # Apply model to data.
    prediction_binary = prediction_model.predict(data_imputed)[:]
    prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]

    return patient_ids, prediction_binary, prediction_probability


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Clean the dataset


def clean_data(data):
    # Drop unnecessary columns
    data.drop(columns=['studyid_adm', 'lengthadm'], inplace=True)

    # Change datatype of 'inhospital_mortality'
    data['inhospital_mortality'] = data['inhospital_mortality'].astype(
        'object')

    # Remove intervention columns
    intervention_columns = data.columns[data.columns.str.contains(
        'admitabx_adm_')]
    data.drop(columns=intervention_columns, inplace=True)

    # Define age groups
    data['age_group'] = pd.cut(
        data['agecalc_adm'],
        bins=[0, 2, 5, 11, 23, 35, 47, 59, 65],
        labels=[
            'Newborn (0-2m)',
            'Infant (3-5m)',
            'Baby (6-11m)',
            'Toddler (12-23m)',
            'Preschool-2 (24-35m)',
            'Preschool-3 (36-47m)',
            'Preschool-4 (48-59m)',
            '5 years (60m)'
        ],
        right=True
    )

    # Function to fill missing values by median of age groups
    def fill_missing_by_median(data, column_name, age_group_column):
        medians = data.groupby(age_group_column)[column_name].median()
        data[column_name] = data.apply(
            lambda row: medians[row[age_group_column]] if pd.isnull(
                row[column_name]) else row[column_name],
            axis=1
        )

    # Apply missing value filling for specific columns
    fill_missing_by_median(data, 'height_cm_adm', 'age_group')
    fill_missing_by_median(data, 'weight_kg_adm', 'age_group')
    fill_missing_by_median(data, 'muac_mm_adm', 'age_group')
    fill_missing_by_median(data, 'hr_bpm_adm', 'age_group')
    fill_missing_by_median(data, 'rr_brpm_app_adm', 'age_group')
    fill_missing_by_median(data, 'sysbp_mmhg_adm', 'age_group')
    fill_missing_by_median(data, 'diasbp_mmhg_adm', 'age_group')
    fill_missing_by_median(data, 'hematocrit_gpdl_adm', 'age_group')

    # Fill missing values for oxygen saturation with the mean
    data['spo2site2_pc_oxi_adm'].fillna(
        data['spo2site2_pc_oxi_adm'].mean(), inplace=True)

    # Drop additional columns with high NaN values and low relevance
    columns_to_drop = [
        'spo2other_adm', 'vaccmeaslessource_adm', 'vaccpneumocsource_adm',
        'vaccdptsource_adm', 'nonexclbreastfed_adm', 'lactate2_mmolpl_adm'
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # Fill categorical columns with mode
    categorical_columns = ['temp_c_adm', 'spo2site1_pc_oxi_adm', 'spo2onoxy_adm', 'respdistress_adm', 'caprefill_adm', 'bcseye_adm', 'vaccdpt_adm', 'bcsverbal_adm', 'vaccmeasles_adm', 'priorweekantimal_adm', 'priorhosp_adm', 'prioryearwheeze_adm', 'exclbreastfed_adm', 'totalbreastfed_adm', 'deliveryloc_adm', 'travelmethod_adm', 'traveldist_adm', 'badhealthduration_adm', 'caregiver_adm_new',
                           'caregiverage_adm', 'momalive_adm', 'momagefirstpregknown_adm', 'momhiv_adm', 'householdsize_adm', 'alivechildren_adm', 'deadchildren_adm', 'watersource_adm', 'waterpure_adm', 'cookloc_adm', 'lightfuel_adm', 'bednet_adm', 'hivstatus_adm', 'malariastatuspos_adm', 'momageknown_adm', 'momage_adm', 'hctpretransfusion_adm', 'lactate_mmolpl_adm', 'glucose_mmolpl_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm']
    for column in categorical_columns:
        if column in data.columns:
            data[column].fillna(data[column].mode()[0], inplace=True)

    # Handle specific columns with proportional filling
    column_name = 'momagefirstpreg_adm'
    category_counts = data[column_name].value_counts(normalize=True)
    num_nan = data[column_name].isna().sum()
    nan_distribution = (category_counts * num_nan).round().astype(int)
    values_to_fill = np.hstack([
        [category] * count for category, count in nan_distribution.items()
    ])
    np.random.shuffle(values_to_fill)
    data.loc[data[column_name].isna(), column_name] = values_to_fill

    return data

# Save cleaned dataset


def save_data(data, output_path):
    data.to_csv(output_path, index=False)

# Main script execution


def data_cleaning_main(input_file, verbose=False):

    output_file = "cleaned_data.csv"
    print('Loading the Challenge data...') if verbose >= 1 else None
    data = load_data(input_file)
    cleaned_data = clean_data(data)
    print('Cleaning and feature extraction is complete...') if verbose >= 1 else None
    save_data(cleaned_data, output_file)

# Code for model training


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


def model_training_main(cleaned_data=None, verbose=False):
    # prepare data
    print("preparing data now...") if verbose else None
    data, numerical_cols, categorical_cols = get_cleaned_data()
    print("preparing data completed...") if verbose else None

    # training model
    print("training model now...") if verbose else None
    pipeline = model_training(data, numerical_cols, categorical_cols)
    print(pipeline)
    print("\n\n\ntraining model complete...") if verbose else None
    return pipeline, None


#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, prediction_model):
    d = {'imputer': imputer, 'prediction_model': prediction_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
