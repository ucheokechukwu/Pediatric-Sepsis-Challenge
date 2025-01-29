import pandas as pd
import numpy as np

# Load the dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Clean the dataset
def clean_data(data):
    # Drop unnecessary columns
    data.drop(columns=['studyid_adm', 'lengthadm'], inplace=True)

    # Change datatype of 'inhospital_mortality'
    data['inhospital_mortality'] = data['inhospital_mortality'].astype('object')

    # Remove intervention columns
    intervention_columns = data.columns[data.columns.str.contains('admitabx_adm_')]
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
            lambda row: medians[row[age_group_column]] if pd.isnull(row[column_name]) else row[column_name],
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
    data['spo2site2_pc_oxi_adm'].fillna(data['spo2site2_pc_oxi_adm'].mean(), inplace=True)

    # Drop additional columns with high NaN values and low relevance
    columns_to_drop = [
        'spo2other_adm', 'vaccmeaslessource_adm', 'vaccpneumocsource_adm', 
        'vaccdptsource_adm', 'nonexclbreastfed_adm', 'lactate2_mmolpl_adm'
    ]
    data.drop(columns=columns_to_drop, inplace=True)

    # Fill categorical columns with mode
    categorical_columns = ['temp_c_adm', 'spo2site1_pc_oxi_adm', 'spo2onoxy_adm', 'respdistress_adm', 'caprefill_adm', 'bcseye_adm', 'vaccdpt_adm', 'bcsverbal_adm', 'vaccmeasles_adm', 'priorweekantimal_adm', 'priorhosp_adm', 'prioryearwheeze_adm', 'exclbreastfed_adm', 'totalbreastfed_adm', 'deliveryloc_adm', 'travelmethod_adm', 'traveldist_adm', 'badhealthduration_adm', 'caregiver_adm_new', 'caregiverage_adm', 'momalive_adm', 'momagefirstpregknown_adm', 'momhiv_adm', 'householdsize_adm', 'alivechildren_adm', 'deadchildren_adm', 'watersource_adm', 'waterpure_adm', 'cookloc_adm', 'lightfuel_adm', 'bednet_adm', 'hivstatus_adm', 'malariastatuspos_adm', 'momageknown_adm', 'momage_adm', 'hctpretransfusion_adm', 'lactate_mmolpl_adm', 'glucose_mmolpl_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm']
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
def main():
    input_file = "../SyntheticData_training.csv"  # Update with actual file path
    output_file = "cleaned_data.csv"

    data = load_data(input_file)
    cleaned_data = clean_data(data)
    save_data(cleaned_data, output_file)

if __name__ == "__main__":
    main()
