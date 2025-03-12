import pandas as pd
import numpy as np
from functools import lru_cache
target = "inhospital_mortality"


class FeatureEngineer:
    def __init__(self):
        """
        Initializes the Feature Engineer with predefined constants, including the feature keys
        and the file paths of the CDC and WHO Growth Charts for BMI and WFL z-score calculations.
        """
        # Primary feature keys
        self.primary_features_keys = [
            'agecalc_adm', 'height_cm_adm', 'weight_kg_adm', 'sex_adm',
            'caprefill_adm', 'glucose_mmolpl_adm', 'hematocrit_gpdl_adm',
            'lactate_mmolpl_adm', 'lactate2_mmolpl_adm', 'spo2site1_pc_oxi_adm',
            'spo2site2_pc_oxi_adm', 'sqi1_perc_oxi_adm', 'sqi2_perc_oxi_adm',
            'sysbp_mmhg_adm', 'diasbp_mmhg_adm', 'hr_bpm_adm', 'temp_c_adm',
            'bcseye_adm', 'bcsmotor_adm', 'bcsverbal_adm']

        # Target column for the dataset
        self.target_column = 'inhospital_mortality'

        # loading the BMI chart for z-score calculation
        self.bmi_csv_path = "charts/bmiagerev.csv"
        self.df_bmi = pd.read_csv(self.bmi_csv_path, skiprows=[220])

        # Loading the WFL percentiles data for z-score calculation
        self.wfl_csv_path = "charts/WHO-{gender}-Weight-for-length-Percentiles.csv"
        self.wfl_df = pd.concat(
            [pd.read_csv(f"{self.wfl_csv_path.format(gender=gender)}").assign(
                gender=gender) for gender in ['Boys', 'Girls']],
            ignore_index=True)

        # loading charts for Systolic and diastolic percentiles
        self.bp_filepath = "charts/BP_Percentiles.xlsx"
        self.bp_boys_chart = pd.read_excel(self.bp_filepath,
                                           sheet_name="Boys",
                                           index_col=[0, 1],
                                           header=[0, 1])
        self.bp_girls_chart = pd.read_excel(self.bp_filepath,
                                            sheet_name="Girls",
                                            index_col=[0, 1],
                                            header=[0, 1])

    def fit(self, X, y=None):
        """
        Placeholder method for fit.

        :param X: The input data (DataFrame)
        :param y: The target values
        """
        pass  # Not used in this case, but can be extended if needed

    def transform(self, X, y=None):
        """
        Transforms the dataset by handling missing values, engineering secondary features,
        and preparing the final dataset.

        :param X: The input data (DataFrame)
        :param y: The target values
        :return: Transformed DataFrame with primary and secondary features
        """
        X = X.copy()

        # Step 0: Get the symptoms_combo, symptoms_bin, comborbidity
        self._get_combo_features(X)

        # Step 1: Select only the primary features columns + the combination keys
        X = X[self.primary_features_keys +
              ['symptoms', 'symptoms_bin', 'comorbidity']]

        # Step 2: Fill missing values in specific columns
        X["lactate2_mmolpl_adm"].fillna(X["lactate_mmolpl_adm"], inplace=True)
        X["spo2site2_pc_oxi_adm"].fillna(
            X["spo2site1_pc_oxi_adm"], inplace=True)
        X["sqi2_perc_oxi_adm"].fillna(X["sqi1_perc_oxi_adm"], inplace=True)

        # Step 3: Drop rows with missing values
        X[target] = y
        X = X.dropna()
        y = X[target]

        # Step 4: Engineer the secondary features
        # A higher shock index indicates worse perfusion and is associated with shock.
        X['shock_index'] = X['hr_bpm_adm'] / X['sysbp_mmhg_adm']
        # MAP < 65 mmHg is a sign of hypotension and poor perfusion.
        X['map'] = X['diasbp_mmhg_adm'] + \
            X['sysbp_mmhg_adm'] / 3 - X['diasbp_mmhg_adm']
        X['hypotension_flag'] = np.where(X['map'] < 65, False, True)
        # lactate clearance < 10%: Associated with poor outcomes, including higher mortality and organ failure.
        # a negative value indicates rising lactate (worsening tissue hypoxia)
        X['lactate_clearance'] = 100 * \
            (X['lactate_mmolpl_adm'] - X['lactate2_mmolpl_adm']) / \
            X['lactate_mmolpl_adm']
        X['hypoxia_flag'] = np.where(X['lactate_clearance'] < 10, False, True)
        X['extreme_hypoxia_flag'] = np.where(
            X['lactate_clearance'] < 0, False, True)

        # Step 5: Calculate the Blantyre_coma_scale
        # The minimum score is 0 which indicates poor results while the maximum is 5 indicating good results.
        # All scores under 5 are considered abnormal
        X['bcseye_numeric'] = np.where(
            X['bcseye_adm'] == "Watches or follows", 1, 0)
        X['bcsverbal_numeric'] = np.where(
            X['bcsverbal_adm'] == 'Cries appropriately with pain, or, if verbal, speaks', 1, 0)
        conditions = [
            X['bcsmotor_adm'] == 'Localizes painful stimulus',
            X['bcsmotor_adm'] == 'Withdraws limb from painful stimulus',
            X['bcsmotor_adm'] == 'No response or inappropriate response']
        values = [2, 1, 0]
        X['bcsmotor_numeric'] = np.select(conditions, values, default=0)
        X['bcs_adm'] = X['bcseye_numeric'] + \
            X['bcsverbal_numeric'] + X['bcsmotor_numeric']
        X['bcs_adm'] = (X['bcs_adm']).astype('category')

        X['bcs_flag'] = np.where(X['bcs_adm'] == 5, False, True)

        # Step 6: Calculate and add Z-scores to X
        X['z_score'] = self._calculate_z_score(
            X['sex_adm'], X['agecalc_adm'], X['height_cm_adm'], X['weight_kg_adm']
        )

        # Step 7: Calculate the sys and dias percentiles
        # Prepare X DataFrame with age lookup
        X['age_lookup'] = np.where(X['agecalc_adm'] > 6,
                                   # Round the age to years
                                   np.round(X['agecalc_adm']/12), 1.0)

        # Convert sex_adm to boolean for efficiency
        X['is_male'] = X['sex_adm'].str.lower() == 'male'

        # Apply the function with optimized series creation
        results = X.apply(lambda row: self._calculate_bp_percentiles(
            age_lookup=row['age_lookup'],
            is_male=row['is_male'],
            height_cm_adm=row['height_cm_adm'],
            sysbp_mmhg_adm=row['sysbp_mmhg_adm'],
            diasbp_mmhg_adm=row['diasbp_mmhg_adm']
        ), axis=1).tolist()

        X[['systolic_percentile', 'diastolic_percentile']
          ] = pd.DataFrame(results, index=X.index)
        X['sys_rank'] = self._map_percentile_to_rank(
            X['systolic_percentile'], bp_type="systolic")
        X['dias_rank'] = self._map_percentile_to_rank(
            X['diastolic_percentile'], bp_type="diastolic")
        X.drop(columns=['age_lookup', 'is_male',
                        'systolic_percentile', 'diastolic_percentile'], inplace=True)

        # Step 7: Enhance primary features

        # 1 if glucose < 4 mmol/L (hypoglycemia) or > 10 mmol/L (hyperglycemia), 0 otherwise
        X['glycemia_flag'] = np.where((X['glucose_mmolpl_adm'] >= 4) & (
            X['glucose_mmolpl_adm'] <= 10), False, True)

        # 1 if hematocrit < 30% (anemia), 0 otherwise
        X['anemia_flag'] = np.where(
            X['hematocrit_gpdl_adm'] >= 30, False, True)
        # 1 if hematocrit < 20% (anemia), 0 otherwise
        X['severe_anemia_flag'] = np.where(
            X['hematocrit_gpdl_adm'] >= 20, False, True)

        # 1 if lactate ≥ 2.0 mmol/L (hyperlactatemia), 0 otherwise
        X['hyperlactatemia_flag'] = np.where(
            X['lactate_mmolpl_adm'] < 2, False, True)

        # 1 if lactate ≥ 4.0 mmol/L (lactic acidosis), 0 otherwise
        X['lactic_acidosis_flag'] = np.where(
            X['lactate_mmolpl_adm'] < 5, False, True)

        # 1 if temperature > 38°C (fever) or < 36°C (hypothermia), 0 otherwise
        X['hypo_hyper_thermia_flag'] = np.where(
            (X['temp_c_adm'] > 36) & (X['temp_c_adm'] < 38), False, True)

        # Select the more reliable SpO2 value based on SQI
        conditions = [
            X['sqi1_perc_oxi_adm'] > X['sqi2_perc_oxi_adm'],
            X['sqi2_perc_oxi_adm'] > X['sqi1_perc_oxi_adm'],
            # Adjust tolerance if needed
            np.isclose(X['sqi1_perc_oxi_adm'],
                       X['sqi2_perc_oxi_adm'], atol=0.01)
        ]
        choices = [
            X['spo2site1_pc_oxi_adm'],
            X['spo2site2_pc_oxi_adm'],
            np.minimum(X['spo2site1_pc_oxi_adm'], X['spo2site2_pc_oxi_adm'])
        ]
        # Default can be adjusted if needed
        X['spo_selected'] = np.select(conditions, choices, default=np.nan)

        # 1 if SpO2: < 95% on room air (hypoxemia), 0 otherwise
        X['hypoxemia_flag'] = np.where(X['spo_selected'] > 95, False, True)

        # Step Last: Return the transformed DataFrame
        print(
            f"""\n\nFeature engineering completed. 
             Added {len(X.columns)} new features: 
             {X.columns}""")
        return X

    def fit_transform(self, X, y=None):
        """
        Combines fit and transform methods.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _get_combo_features(self, X):
        """get the combination of most significant symptoms and comorbidities"""
        symptom_features = [
            col for col in X.columns if col.startswith('symptoms_adm')]
        for col in symptom_features:
            X[col] = np.where(X[col] == 'Checked', 1, 0)

        comorbidity_features = [
            col for col in X.columns if col.startswith('comorbidity_adm')]
        for col in comorbidity_features:
            X[col] = np.where(X[col] == 'Checked', 1, 0)

        X['symptoms'] = X['symptoms_adm___1'] + X['symptoms_adm___3'] + X['symptoms_adm___6'] + \
            X['symptoms_adm___7'] + X['symptoms_adm___13'] + X['symptoms_adm___14']
        X['symptoms_bin'] = np.where(X['symptoms'] >= 4, 1, 0)

        X['comorbidity'] = X['comorbidity_adm___1'] + X['comorbidity_adm___2'] + \
            X['comorbidity_adm___3'] + X['comorbidity_adm___5'] + X['comorbidity_adm___6'] + \
            X['comorbidity_adm___8'] + X['comorbidity_adm___9'] + \
            X['comorbidity_adm___11']

    def _calculate_bmi_zscore(self, sex_adm, agecalc_adm, height_cm_adm, weight_kg_adm):
        """Vectorized BMI-for-age z-score calculation for children > 24 months."""
        bmi = weight_kg_adm / ((height_cm_adm / 100) **
                               2)  # Vectorized BMI calculation

        # Convert "Male"/"Female" to 1/2 and filter in a vectorized manner
        sex_code = np.where(sex_adm.str.lower() == "male", 1, 2)
        df = self.df_bmi[self.df_bmi["Sex"].isin(sex_code)]
        df = df.sort_values("Agemos")

        # Perform merge_asof in a vectorized way
        age_df = pd.DataFrame({"Agemos": agecalc_adm})
        merged = pd.merge_asof(age_df.sort_values(
            "Agemos"), df, on="Agemos", direction="nearest")

        L, M, S = merged[['L', 'M', 'S']].values.T  # Extract L, M, S as arrays
        # Vectorized z-score calculation
        z_scores = ((bmi / M) ** L - 1) / (L * S)

        return z_scores

    def _calculate_wfl_zscore(self, sex_adm, height_cm_adm, weight_kg_adm):
        """Vectorized weight-for-length z-score calculation for children ≤ 24 months."""
        # Filter the DataFrame based on sex_adm
        sex_code = np.where(sex_adm.str.lower() == "male", 'Boys', 'Girls')
        df = self.wfl_df[self.wfl_df["gender"].isin(
            sex_code)].sort_values("Length")

        # Perform merge_asof in a vectorized way
        length_df = pd.DataFrame({"Length": height_cm_adm})
        merged = pd.merge_asof(length_df.sort_values(
            "Length"), df, on="Length", direction="nearest")

        L, M, S = merged[['L', 'M', 'S']].values.T  # Extract L, M, S as arrays
        z_scores = ((weight_kg_adm / M) ** L - 1) / \
            (L * S)  # Vectorized z-score calculation

        return z_scores

    def _calculate_z_score(self, sex_adm, agecalc_adm, height_cm_adm, weight_kg_adm):
        """Vectorized function to determine and calculate the correct z-score."""
        z_scores = np.where(
            agecalc_adm > 24,
            self._calculate_bmi_zscore(
                sex_adm, agecalc_adm, height_cm_adm, weight_kg_adm),
            self._calculate_wfl_zscore(sex_adm, height_cm_adm, weight_kg_adm)
        )
        return z_scores

    @lru_cache(maxsize=1024)
    def _lookup_bp_tables(self, height_cm_adm, hr_bpm_adm, age_lookup, is_male, is_diastolic):  # More arguments!
        bp_chart = self.bp_boys_chart if is_male else self.bp_girls_chart
        bp_data = bp_chart.loc[age_lookup,
                               'Diastolic BP (mm Hg)' if is_diastolic else 'Systolic BP (mm Hg)']

        height_row = bp_data.loc["Measured height (cm)", :]
        closest_height_index = height_row.sub(height_cm_adm).abs().idxmin()
        heart_rate_values = bp_data[closest_height_index].iloc[1:].astype(
            float)  # Ensure float type for comparison

        min_hr = heart_rate_values.min()
        max_hr = heart_rate_values.max()

        if hr_bpm_adm <= min_hr:
            return (None, heart_rate_values.index[0])
        elif hr_bpm_adm >= max_hr:
            return (heart_rate_values.index[-1], None)
        else:
            lower_percentile = heart_rate_values[heart_rate_values <=
                                                 hr_bpm_adm].index[-1]
            upper_percentile = heart_rate_values[heart_rate_values >
                                                 hr_bpm_adm].index[0]
            return (lower_percentile, upper_percentile)

    # is_male as argument
    def _calculate_bp_percentiles(self, age_lookup, is_male, height_cm_adm, sysbp_mmhg_adm, diasbp_mmhg_adm):
        diastolic_percentile = self._lookup_bp_tables(
            height_cm_adm, diasbp_mmhg_adm, age_lookup, is_male, True)
        systolic_percentile = self._lookup_bp_tables(
            height_cm_adm, sysbp_mmhg_adm, age_lookup, is_male, False)
        return systolic_percentile, diastolic_percentile

    def _map_percentile_to_rank(self, percentile_column, bp_type):
        conditions = [
            percentile_column == (None, '50th'),
            percentile_column == ('50th', '90th'),
            percentile_column == ('90th', '95th'),
            percentile_column == ('95th', '95th + 12 mm Hg'),
            percentile_column == ('95th + 12 mm Hg', None)
        ]

        choices = [
            "Hypotension",
            f"Normal {bp_type.capitalize()} BP",
            f"Elevated {bp_type.capitalize()} BP",
            "Hypertension",
            "Stage 2 Hypertension"]

        return np.select(conditions, choices, default="Undefined")
