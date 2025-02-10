import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pointbiserialr, spearmanr, chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# ORDINAL_MAPPING = {
#     'age_group': {
#         'Newborn (0-2m)' : 0,
#         'Infant (3-5m)' : 1,
#         'Baby (6-11m)' : 2,
#         "Toddler (12-23m)": 3,
#         "Preschool-2 (24-35m)": 4,
#         "Preschool-3 (36-47m)": 5,
#         "Preschool-4 (48-59m)": 6,
#         "5 years (60m)": 7
#     }
# }

ORDINAL_MAPPING = {
    'oxygenavail_adm': {
        "Oxygen not available": 0,
        "Oxygen available but not enough": 1,
        "Oxygen available and not being used": 2,
        "Oxygen available and being used": 3
    },
    'bcsmotor_adm': {
        "No response or inappropriate response": 2,
        "Withdraws limb from painful stimulus": 1,
        "Localizes painful stimulus": 0
    },
    'bcsverbal_adm': {
        "No vocal response to pain": 2,
        "Moan or abnormal cry with pain": 1,
        "Cries appropriately with pain, or, if verbal, speaks": 0
    },
    'vaccpneumoc_adm': {
        "0 doses": 3,
        "1 dose": 2,
        "2 doses": 1,
        "3 doses": 0,
        "Unknown": -1
    },
    'vaccdpt_adm': {
        "0 doses": 3,
        "1 dose": 2,
        "2 doses": 1,
        "3 doses": 0,
        "Unknown": -1
    },
    'priorhosp_adm': {
        "> 1 year": 4,
        "1 month - 1 year": 3,
        "7 days - 1 month": 2,
        "< 7 days": 1,
        "Never": 0,
        "Dont know": -1
    },
    'prioryearwheeze_adm': {
        "Not applicable (< 6m old)": -2,
        "No": 0,
        "Yes": 1,
        "Dont know": -1
    },
    'prioryearcough_adm': {
        "Not applicable (child < 6m old)": -2,
        "No": 0,
        "Yes": 1,
        "Dont know": -1
    },
    # 'exclbreastfed_adm': {
    #     "never exclusively breastfed": 7,
    #     "1 month": 6,
    #     "2 months": 5,
    #     "3 months": 4,
    #     "4 months": 3,
    #     "5 months": 2,
    #     "6 months": 1,
    #     "More than 6 months": 0,
    #     "Unknown": -1
    # },
    # 'totalbreastfed_adm': {
    #     "0-3 months": 3,
    #     "4-6 months": 2,
    #     "7-12 months": 1,
    #     ">12 months": 0,
    #     "Unknown": -1
    # },
    'traveldist_adm': {
        "< 30 minutes": 0,
        "30 minutes - 1 hour": 1,
        "1 - 2 hours": 2,
        "2 - 3 hours": 3,
        "3 - 4 hours": 4,
        "4 - 8 hours": 5,
        "> 8 hours": 6
    },
    'caregiverage_adm': {
        "N/A (in care)": -2,
        "< = 18 years old": 0,
        "> 18 years old": 1,
        "> 50 years old": 2
    },
    'momedu_adm': {
        "No school": 4,
        "< = P3": 3,
        "P4-P7": 2,
        "S1-S6": 1,
        "Post secondary (including post S4 technical school)": 0,
        "Dont know": -1
    },
    # 'tobacco_adm': {
    #     "Never": 0,
    #     "Less than monthly": 1,
    #     "Monthly": 2,
    #     "Weekly": 3,
    #     "Daily": 4
    # },
    'bednet_adm': {
        "Never": 2,
        "Sometimes": 1,
        "Always": 0
    },
    'feedingstatus_adm': {
        "Not feeding at all": 2,
        "Feeding poorly": 1,
        "Feeding well": 0
    },
    'birthattend_adm': {
        "Other": 4,
        "Relative": 3,
        "Traditional birth attendant": 2,
        "Nurse / midwife": 1,
        "Doctor / Clinical officer": 0
    },
    'travelmethod_adm': {
        "Other": 0,
        "Walking": 1,
        "Motorcycle": 2,
        "Taxi/special hire": 3,
        "Private vehicle": 4,
        "Ambulance": 5
    },
    'watersource_adm': {
        "Open source (unprotected, stagnant water, dam)": 0,
        "Slow running water": 1,
        "Fast running water": 2,
        "Bore hole": 3,
        "Protected spring": 4,
        "Municipal water": 5
    },
    'caregiver_adm_new': {
        "Other": 3,
        "Grandparent": 2,
        "Father": 1,
        "Mother": 0
    }
}

THREE_VALUE_ORDINAL_MAPPING = {
    'vaccmeasles_adm': {"Yes": 1, "No": 0, "Unknown": -1},
    'priorweekabx_adm': {"No": 0, "Dont know": -1, "Yes": 1},
    'priorweekantimal_adm': {"No": 0, "Yes": 1, "Dont know": -1},
    'diarrheaoften_adm': {"No": 0, "Yes": 1, "Dont know": -1},
    'tbcontact_adm': {"No": 0, "Yes": 1, "Dont know": -1},
    'nonexclbreastfed_adm': {"Formula+": 0, "Other+": 1, "N/A - exclusive breastfeeding": -1},
    'momhiv_adm': {"Negative": 0, "Unknown": -1, "Positive": 1}
}

NOMINAL_FEATURES = ['deliveryloc_adm', 'caregivermarried_adm', 'cookloc_adm',
                    'exclbreastfed_adm', 'totalbreastfed_adm', 'tobacco_adm',
                    'lightfuel_adm', 'badhealthduration_adm']

BINARY_FEATURES = ['admitabx_adm___1',  'admitabx_adm___10',  'admitabx_adm___11',  'admitabx_adm___12',
                   'admitabx_adm___13',  'admitabx_adm___14',  'admitabx_adm___15',  'admitabx_adm___16',
                   'admitabx_adm___17',  'admitabx_adm___18',  'admitabx_adm___19',  'admitabx_adm___2',
                   'admitabx_adm___20',  'admitabx_adm___21',  'admitabx_adm___3',  'admitabx_adm___4',
                   'admitabx_adm___5',  'admitabx_adm___6',  'admitabx_adm___7',  'admitabx_adm___8',
                   'admitabx_adm___9',  'bcgscar_adm',  'bcseye_adm',  'birthdetail_adm___1',
                   'birthdetail_adm___2',  'birthdetail_adm___3',  'birthdetail_adm___4',  'birthdetail_adm___5',
                   'birthdetail_adm___6',  'caprefill_adm',  'comorbidity_adm___1',  'comorbidity_adm___10',
                   'comorbidity_adm___11',  'comorbidity_adm___12',  'comorbidity_adm___2',  'comorbidity_adm___3',
                   'comorbidity_adm___4',  'comorbidity_adm___5',  'comorbidity_adm___6',  'comorbidity_adm___7',
                   'comorbidity_adm___8',  'comorbidity_adm___9',  'cookfuel_adm___1',  'cookfuel_adm___2',
                   'cookfuel_adm___3',  'cookfuel_adm___4',  'cookfuel_adm___5',  'cookfuel_adm___6',
                   'cookfuel_adm___7',  'cookfuel_adm___8',  'duedateknown_adm',  'hctpretransfusion_adm',
                   'hivstatus_adm',  'malariastatuspos_adm',  'momagefirstpregknown_adm',  'momageknown_adm',
                   'momalive_adm',  'respdistress_adm',  'sex_adm',  'spo2onoxy_adm',  'symptoms_adm___1',
                   'symptoms_adm___10',  'symptoms_adm___11',  'symptoms_adm___12',  'symptoms_adm___13',
                   'symptoms_adm___14',  'symptoms_adm___15',  'symptoms_adm___16',  'symptoms_adm___17',
                   'symptoms_adm___18',  'symptoms_adm___2',  'symptoms_adm___3',  'symptoms_adm___4',
                   'symptoms_adm___5',  'symptoms_adm___6',  'symptoms_adm___7',  'symptoms_adm___8',
                   'symptoms_adm___9',  'vaccdptsource_adm',  'vaccmeaslessource_adm',  'vaccpneumocsource_adm',
                   'waterpure_adm']

NUMERIC_FEATURES = ['height_cm_adm', 'agecalc_adm', 'weight_kg_adm', 'muac_mm_adm',
                    'hr_bpm_adm', 'rr_brpm_app_adm', 'sysbp_mmhg_adm', 'diasbp_mmhg_adm', 'temp_c_adm',
                    'spo2site1_pc_oxi_adm', 'spo2site2_pc_oxi_adm', 'spo2other_adm', 'hematocrit_gpdl_adm',
                    'lactate_mmolpl_adm', 'lactate2_mmolpl_adm', 'glucose_mmolpl_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm',
                    'momage_adm', 'momagefirstpreg_adm', 'householdsize_adm', 'alivechildren_adm', 'deadchildren_adm']


class FeatureTransformer:
    def __init__(self):
        # numerical feature columns
        self.numerical_features = NUMERIC_FEATURES
        self.scaler = StandardScaler()
        self.lower_percentile = None  # To store 2nd percentile for trimming
        self.upper_percentile = None  # To store 98th percentile for trimming

        # ordinal mapping
        self.ordinal_mapping = ORDINAL_MAPPING

        # Mapping for 3-value ordinal features (Yes/No/Unknown)
        self.three_value_ordinal_mapping = THREE_VALUE_ORDINAL_MAPPING

        # Nominal Features for One-Hot Encoding
        self.nominal_features = NOMINAL_FEATURES

        # binary features
        self.binary_features = BINARY_FEATURES

    def fit(self, X: pd.DataFrame, y=None):
        # Fit the scaler to the numerical data
        self.scaler.fit(X[self.numerical_features])

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X = X.copy()
        # Scale numerical features using the saved scaler parameters
        X[self.numerical_features] = self.scaler.transform(
            X[self.numerical_features])

        # placeholder code to maybe re-categorize tobacco, exclbreastfeeding, totalbreastfeeding

        # Apply ordinal mappings
        for col, mapping in self.ordinal_mapping.items():
            if col in X.columns:
                # If any missing values, set to -1
                X[col] = X[col].map(mapping).fillna(-1)

        # Apply 3-value ordinal mappings
        for col, mapping in self.three_value_ordinal_mapping.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(-1)

        # Apply One-Hot Encoding for nominal features
        X = pd.get_dummies(X, columns=self.nominal_features, drop_first=True)

        # Handle binary features
        for col in self.binary_features:
            unique_values = X[col].dropna().unique()
            # If the column has only one unique value, treat it as binary (1 for presence)
            if len(unique_values) == 1:
                X[col] = 1
            else:
                # Otherwise, treat it as a binary feature
                X[col] = X[col].apply(lambda x: 1 if x ==
                                      unique_values[0] else 0)

        return X


def scale_ordinal_features(X, ordinal_columns):
    """
    Scales the ordinal features in the range of -2 to +2.

    Args:
        X (pd.DataFrame): The input feature set.
        ordinal_columns (list): List of columns representing ordinal features.

    Returns:
        pd.DataFrame: The transformed feature set with scaled ordinal features.
    """
    # Initialize MinMaxScaler to scale between -2 and +2
    scaler = MinMaxScaler(feature_range=(-2, 2))

    # Scale the ordinal columns and replace them in the original DataFrame
    X_scaled = X.copy()
    X_scaled[ordinal_columns] = scaler.fit_transform(X[ordinal_columns])

    return X_scaled


def calculate_purity(cluster_labels, true_labels):
    """
    Calculates the purity score for the clusters by measuring how pure the clusters are with respect to the true labels,
    checks if all clusters have the same dominant class, and measures the dominance percentage of Class 1.

    Args:
        cluster_labels (array): Cluster labels generated by the clustering algorithm.
        true_labels (array): The true class labels.

    Returns:
        tuple: 
            - float: Purity score between 0 and 1.
            - bool: True if all clusters have the same dominant class, False otherwise.
            - list: A list of dominance percentages for Class 1 across all clusters where it is dominant.
    """
    df = pd.DataFrame({'Cluster': cluster_labels, 'TrueLabel': true_labels})
    purity = 0
    dominant_classes = []
    class_1_dominance = []  # Stores dominance % of Class 1 where it's the dominant class

    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]['TrueLabel']
        class_counts = cluster_data.value_counts()
        most_frequent_class = class_counts.idxmax()
        most_frequent_class_count = class_counts.max()
        total_cluster_size = len(cluster_data)

        # Calculate purity contribution
        purity += most_frequent_class_count

        # Store dominant class
        dominant_classes.append(most_frequent_class)

        # Calculate dominance percentage
        dominance_percentage = (
            most_frequent_class_count / total_cluster_size) * 100

        # If dominant class is Class 1, record its dominance percentage
        if most_frequent_class == 1:
            class_1_dominance.append(dominance_percentage)

    purity /= len(df)

    # Check if all clusters have the same dominant class
    dominant_class_in_all_clusters = len(set(dominant_classes)) == 1

    return purity, dominant_class_in_all_clusters, class_1_dominance


# Function to create feature_types dictionary
def create_feature_types(numerical_features,
                         binary_category,
                         nominal_category,
                         ordinal_category=ORDINAL_MAPPING.keys()):
    feature_types = {}

    for feature in numerical_features:
        feature_types[feature] = 'numerical'
    for feature in ordinal_category:
        feature_types[feature] = 'ordinal'
    for feature in nominal_category:
        feature_types[feature] = 'nominal'
    for feature in binary_category:
        feature_types[feature] = 'binary'

    return feature_types

# Function to compute correlations


def compute_feature_correlations(df, target_col, feature_types, p_value_threshold=0.05):
    results = []
    target = df[target_col]

    for feature, ftype in feature_types.items():
        if feature == target_col:
            continue

        if ftype == 'numerical':
            corr, p_value = pointbiserialr(df[feature], target)

        elif ftype == 'ordinal':
            corr, p_value = spearmanr(df[feature], target)

        elif ftype == 'nominal':
            # Chi-Square Test
            contingency_table = pd.crosstab(df[feature], target)
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            corr = chi2  # Using chi-square statistic as a measure

        elif ftype == 'binary':
            corr = np.corrcoef(df[feature], target)[0, 1]  # Phi coefficient
            p_value = np.nan  # Not applicable in this case

        else:
            continue

        # Apply significance threshold
        significance = p_value < p_value_threshold if not np.isnan(
            p_value) else True

        results.append({'Feature': feature, 'Type': ftype, 'Correlation': corr if significance else False,
                       'p-value': p_value, 'Significant': significance})

    # Compute Mutual Information for all features
    discrete_features = []

    # Iterate over the feature_types dictionary
    for feature, ftype in feature_types.items():
        if feature != target_col:  # Skip the target column
            # Check if the feature type is in ['ordinal', 'binary']
            if ftype in ['binary']:
                discrete_features.append(True)
            else:
                discrete_features.append(False)

    # Now, use the discrete_features list in mutual_info_classif
    mi_scores = mutual_info_classif(
        df.drop(columns=[target_col]), target, discrete_features=discrete_features)
    for i, feature in enumerate(feature_types.keys()):
        if feature == target_col:
            continue
        for res in results:
            if res['Feature'] == feature:
                res['Mutual Information'] = mi_scores[i]
                break

    return pd.DataFrame(results)

# Example Usage
# df = pd.read_csv("your_data.csv")
# numerical_features = ["feature1", "feature2"]
# ordinal_category = ["feature3"]
# nominal_category = ["feature4"]
# binary_category = ["feature5"]
# feature_types = create_feature_types(numerical_features, ordinal_category, nominal_category, binary_category)
# target_col = "target"
# correlations_df = compute_feature_correlations(df, target_col, feature_types)
# print(correlations_df.sort_values(by='Correlation', ascending=False))
