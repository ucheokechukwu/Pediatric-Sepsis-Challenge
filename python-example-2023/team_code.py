#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries and functions. You can change or remove them.

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd
import mne
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import sys
import os
import warnings
from helper_code import *
warnings.simplefilter(action='ignore', category=FutureWarning)


TARGET = "inhospital_mortality"
CLEANED_DATA_PATH = ""
N_COMPONENTS = 3

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
    imputer = data_cleaning_main(input_file, verbose)
    prediction_model = model_training_main(verbose)

    # remove the cleaned_data.csv file
    os.remove(f"{CLEANED_DATA_PATH}cleaned_data.csv")

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_challenge_model(model_folder, imputer, prediction_model)

    print("Model and imputer have been trained and saved. Ready to run") if verbose else None


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

    # Impute missing data.
    data_imputed = imputer.transform(data)

    # Apply model to data.
    prediction_binary = prediction_model.predict(data_imputed)[:]
    prediction_probability = prediction_model.predict_proba(data_imputed)[:, 1]

    print("Successfully passed test data to trained model and received prediction values.") if verbose else None
    return patient_ids, prediction_binary, prediction_probability


################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

###########################################
# Data cleaning and feature extraction
###########################################

# Load the raw dataset
def load_raw_data(file_path):
    return pd.read_csv(file_path)

# Clean the dataset


class DataImputer:
    def __init__(self):
        self.medians_by_age_group = {}
        self.mean_values = {}
        self.mode_values = {}
        self.nan_distributions = {}
        self.age_bins = [0, 2, 5, 11, 23, 35, 47, 59, 65]
        self.age_labels = [
            'Newborn (0-2m)',
            'Infant (3-5m)',
            'Baby (6-11m)',
            'Toddler (12-23m)',
            'Preschool-2 (24-35m)',
            'Preschool-3 (36-47m)',
            'Preschool-4 (48-59m)',
            '5 years (60m)'
        ]
        self.columns_to_drop = []

    def fit(self, train_data):
        data = train_data.copy()

        # Drop unnecessary columns
        data.drop(columns=['studyid_adm', 'lengthadm'], inplace=True)

        # Remove intervention columns
        intervention_columns = data.columns[data.columns.str.contains(
            'admitabx_adm_')]
        data.drop(columns=intervention_columns, inplace=True)

        # Define age groups
        data['age_group'] = pd.cut(
            data['agecalc_adm'],
            bins=self.age_bins,
            labels=self.age_labels,
            right=True
        )

        # Compute medians for numerical columns by age group
        columns_to_fill_by_median = [
            'height_cm_adm', 'weight_kg_adm', 'muac_mm_adm', 'hr_bpm_adm',
            'rr_brpm_app_adm', 'sysbp_mmhg_adm', 'diasbp_mmhg_adm', 'hematocrit_gpdl_adm'
        ]
        for column in columns_to_fill_by_median:
            self.medians_by_age_group[column] = data.groupby('age_group')[
                column].median()

        # Compute mean for oxygen saturation
        self.mean_values['spo2site2_pc_oxi_adm'] = data['spo2site2_pc_oxi_adm'].mean()

        # Drop additional columns with high NaN values and low relevance
        self.columns_to_drop = [
            'spo2other_adm', 'vaccmeaslessource_adm', 'vaccpneumocsource_adm',
            'vaccdptsource_adm', 'nonexclbreastfed_adm', 'lactate2_mmolpl_adm'
        ]
        data.drop(columns=self.columns_to_drop, inplace=True)

        # Store mode for categorical columns
        categorical_columns = [
            'birthattend_adm',
            'temp_c_adm', 'spo2site1_pc_oxi_adm', 'spo2onoxy_adm', 'respdistress_adm', 'caprefill_adm',
            'bcseye_adm', 'vaccdpt_adm', 'bcsverbal_adm', 'vaccmeasles_adm', 'priorweekantimal_adm',
            'priorhosp_adm', 'prioryearwheeze_adm', 'exclbreastfed_adm', 'totalbreastfed_adm',
            'deliveryloc_adm', 'travelmethod_adm', 'traveldist_adm', 'badhealthduration_adm',
            'caregiver_adm_new', 'caregiverage_adm', 'momalive_adm', 'momagefirstpregknown_adm',
            'momhiv_adm', 'householdsize_adm', 'alivechildren_adm', 'deadchildren_adm', 'watersource_adm',
            'waterpure_adm', 'cookloc_adm', 'lightfuel_adm', 'bednet_adm', 'hivstatus_adm',
            'malariastatuspos_adm', 'momageknown_adm', 'momage_adm', 'hctpretransfusion_adm',
            'lactate_mmolpl_adm', 'glucose_mmolpl_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm'
        ]
        for column in categorical_columns:
            if column in data.columns:
                self.mode_values[column] = data[column].mode()[0]

        # Handle proportional filling for specific columns
        column_name = 'momagefirstpreg_adm'
        category_counts = data[column_name].value_counts(normalize=True)
        num_nan = data[column_name].isna().sum()
        nan_distribution = (category_counts * num_nan).round().astype(int)
        values_to_fill = np.hstack([
            [category] * count for category, count in nan_distribution.items()
        ])
        np.random.shuffle(values_to_fill)
        self.nan_distributions[column_name] = values_to_fill

        return self

    def transform(self, test_data):
        data = test_data.copy()

        # Drop unnecessary columns
        data.drop(columns=['studyid_adm', 'lengthadm'],
                  errors='ignore', inplace=True)

        # Remove intervention columns
        intervention_columns = data.columns[data.columns.str.contains(
            'admitabx_adm_')]
        data.drop(columns=intervention_columns, errors='ignore', inplace=True)

        # Define age groups
        data['age_group'] = pd.cut(
            data['agecalc_adm'],
            bins=self.age_bins,
            labels=self.age_labels,
            right=True
        )

        # Fill missing values by median of age groups
        for column, medians in self.medians_by_age_group.items():
            data[column] = data.apply(
                lambda row: medians[row['age_group']] if pd.isnull(
                    row[column]) else row[column],
                axis=1
            )

        # Fill missing values for oxygen saturation with the mean
        if 'spo2site2_pc_oxi_adm' in data.columns:
            data['spo2site2_pc_oxi_adm'].fillna(
                self.mean_values['spo2site2_pc_oxi_adm'], inplace=True)

        # Drop additional columns with high NaN values and low relevance
        data.drop(columns=self.columns_to_drop, inplace=True)

        # Fill categorical columns with mode
        for column, mode_value in self.mode_values.items():
            if column in data.columns:
                data[column].fillna(mode_value, inplace=True)

        # Handle specific columns with proportional filling
        column_name = 'momagefirstpreg_adm'

        if column_name in data.columns and column_name in self.nan_distributions:
            nan_indices = data[data[column_name].isna()].index
            values_to_fill = self.nan_distributions[column_name][:len(
                nan_indices)]
            data.loc[nan_indices, column_name] = values_to_fill

        return data

# Save cleaned dataset


def save_cleaned_data(data, output_path):
    data.to_csv(output_path, index=False)

# Main script execution


def data_cleaning_main(input_file, verbose=False):
    data = load_raw_data(input_file)
    print('Challenge data loaded.') if verbose else None

    imputer = DataImputer()
    imputer.fit(data)
    cleaned_data = imputer.transform(data)
    print('Imputed fitted and data transformed.') if verbose else None

    output_file = "cleaned_data.csv"
    save_cleaned_data(cleaned_data, output_file)
    return imputer


###########################################
# Model training
###########################################

class ModelPipeline:
    def __init__(self, data_path=CLEANED_DATA_PATH, target_column=TARGET, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.numerical_cols = None
        self.categorical_cols = None

    def load_and_prepare_data(self):
        """Loads the data and identifies numerical and categorical columns."""
        data = pd.read_csv(f"{self.data_path}cleaned_data.csv")
        self.data = data.drop(columns=[self.target_column])
        self.numerical_cols = self.data.select_dtypes(
            include=['number']).columns
        self.categorical_cols = self.data.select_dtypes(
            exclude=['number']).columns

    def get_preprocessor(self):
        """Creates and returns a preprocessing pipeline."""
        return ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_cols)
            ],
            remainder='drop'
        )

    def get_train_test_data(self):
        """Splits the data into training and testing sets."""
        X = self.data.copy()
        y = pd.read_csv(f"{self.data_path}cleaned_data.csv")[
            self.target_column]
        return train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

    def train_model(self, n_components=N_COMPONENTS):
        """Trains the model pipeline and returns the trained pipeline."""
        if self.data is None:
            raise ValueError(
                "Data is not loaded. Call load_and_prepare_data first.")

        # Split data
        X_train, X_test, y_train, y_test = self.get_train_test_data()

        # Define pipeline components
        preprocessor = self.get_preprocessor()
        oversampler = SMOTEENN(random_state=self.random_state)
        pca = PCA(n_components=n_components) if n_components else None
        classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                min_samples_split=2,
                random_state=self.random_state
            ),
            learning_rate=0.01,
            n_estimators=50,
            random_state=self.random_state,
            algorithm='SAMME'
        )

        # Build pipeline
        steps = [('preprocessor', preprocessor), ('oversample', oversampler)]
        if pca:
            steps.append(('pca', pca))
        steps.append(('classifier', classifier))
        pipeline = ImbPipeline(steps)

        # Train pipeline
        pipeline.fit(X_train, y_train)

        return pipeline

    def run(self, verbose=False):
        """Main entry point for preparing data and training the model."""
        self.load_and_prepare_data()
        print("Data preparation completed.") if verbose else None
        pipeline = self.train_model()
        print("Model training completed.") if verbose else None

        return pipeline


def model_training_main(verbose=False):
    model_pipeline = ModelPipeline()
    pipeline = model_pipeline.run(verbose)
    return pipeline

################################################################################
#
# End Optional functions
#
################################################################################

# Save your trained model.


def save_challenge_model(model_folder, imputer, prediction_model):
    d = {'imputer': imputer, 'prediction_model': prediction_model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
