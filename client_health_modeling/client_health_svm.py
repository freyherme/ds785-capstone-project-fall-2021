#################################################################################
########            Support Vector Machine Model Selection               ########
#################################################################################

#####################################
###            Packages           ###
#####################################

from datetime import datetime
import csv
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import StratifiedKFold
import seaborn as sn
import matplotlib.pyplot as plt
from itertools import compress

#####################################
###            Inputs             ###
#####################################

models_dict = {
    "baseline_cv": {
        "numeric_features": ['atd_client_health_score'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "2features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "2features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "2features_c_cv": {
        "numeric_features": ['atd_client_health_score', 'site_asmt_ovr_times_accessed'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "3features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "3features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "3features_c_cv": {
        "numeric_features": ['hs_band', 'cumulative_common_assessment_count_per_ay', 'user_created_custom_reports'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "6features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "7features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "7features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": ['csm_name'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "lag_only": {
        "numeric_features": ['hs_1year_lag_band', 'hs_7_lag_band', 'hs_6_lag_band', 'hs_5_lag_band', 'hs_4_lag_band',
                             'hs_1_lag_band'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "8features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed',
                             'user_created_custom_reports'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "8features_b_cv": {
        "numeric_features": ['hs_band', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed',
                             'user_created_custom_reports'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2]
    },
    "full_cv": {
        "numeric_features": ['has_ise', 'integration_educlimber', 'integration_fast', 'integration_google_classroom',
                             'integration_pra', 'asmt_admin_flex', 'asmt_admin_ib', 'asmt_admin_inspect_prebuilt',
                             'cumulative_common_assessment_count_per_ay', 'hs_1_lag_band', 'hs_1year_lag_band',
                             'hs_4_lag_band', 'hs_5_lag_band', 'hs_6_lag_band', 'hs_7_lag_band', 'hs_band',
                             'matrix_distinct_users', 'matrix_times_accessed', 'mltp_asmt_smry_distinct_users',
                             'mltp_asmt_smry_times_accessed', 'rsp_freq_distinct_users', 'rsp_freq_times_accessed',
                             'site_asmt_ovr_distinct_users', 'site_asmt_ovr_times_accessed',
                             'site_peer_comp_distinct_users', 'site_peer_comp_times_accessed',
                             'skills_letter_distinct_users', 'skills_letter_times_accessed', 'subscriber_tenure_days',
                             'summary_asmt_created', 'ticket_count', 'tile_layouts_created_modified',
                             'user_created_custom_reports', 'arr_dna', 'atd_client_health_score',
                             'atd_feature_adoption_score', 'atd_students_assessed_percent', 'atd_users_login_percent',
                             'sa_tenure_in_days', 'teachers_login_percent'],
        "categorical_features": ['csm_name', 'state'],
        "weights": [{0: 35.0, 1: 65.0}],
        "kernel": ['rbf'],
        "c_value": [.25, .5, 1, 1.25]
    },
    "full_kbest10_cv": {
        "numeric_features": ['has_ise', 'integration_educlimber', 'integration_fast', 'integration_google_classroom',
                             'integration_pra', 'asmt_admin_flex', 'asmt_admin_ib', 'asmt_admin_inspect_prebuilt',
                             'cumulative_common_assessment_count_per_ay', 'hs_1_lag_band', 'hs_1year_lag_band',
                             'hs_4_lag_band', 'hs_5_lag_band', 'hs_6_lag_band', 'hs_7_lag_band', 'hs_band',
                             'matrix_distinct_users', 'matrix_times_accessed', 'mltp_asmt_smry_distinct_users',
                             'mltp_asmt_smry_times_accessed', 'rsp_freq_distinct_users', 'rsp_freq_times_accessed',
                             'site_asmt_ovr_distinct_users', 'site_asmt_ovr_times_accessed',
                             'site_peer_comp_distinct_users', 'site_peer_comp_times_accessed',
                             'skills_letter_distinct_users', 'skills_letter_times_accessed', 'subscriber_tenure_days',
                             'summary_asmt_created', 'ticket_count', 'tile_layouts_created_modified',
                             'user_created_custom_reports', 'arr_dna', 'atd_client_health_score',
                             'atd_feature_adoption_score', 'atd_students_assessed_percent', 'atd_users_login_percent',
                             'sa_tenure_in_days', 'teachers_login_percent'],
        "categorical_features": ['csm_name', 'state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2],
        "feature_selection": ("SelectKBest", 10)
    },
    "full_kbest40_cv": {
        "numeric_features": ['has_ise', 'integration_educlimber', 'integration_fast', 'integration_google_classroom',
                             'integration_pra', 'asmt_admin_flex', 'asmt_admin_ib', 'asmt_admin_inspect_prebuilt',
                             'cumulative_common_assessment_count_per_ay', 'hs_1_lag_band', 'hs_1year_lag_band',
                             'hs_4_lag_band', 'hs_5_lag_band', 'hs_6_lag_band', 'hs_7_lag_band', 'hs_band',
                             'matrix_distinct_users', 'matrix_times_accessed', 'mltp_asmt_smry_distinct_users',
                             'mltp_asmt_smry_times_accessed', 'rsp_freq_distinct_users', 'rsp_freq_times_accessed',
                             'site_asmt_ovr_distinct_users', 'site_asmt_ovr_times_accessed',
                             'site_peer_comp_distinct_users', 'site_peer_comp_times_accessed',
                             'skills_letter_distinct_users', 'skills_letter_times_accessed', 'subscriber_tenure_days',
                             'summary_asmt_created', 'ticket_count', 'tile_layouts_created_modified',
                             'user_created_custom_reports', 'arr_dna', 'atd_client_health_score',
                             'atd_feature_adoption_score', 'atd_students_assessed_percent', 'atd_users_login_percent',
                             'sa_tenure_in_days', 'teachers_login_percent'],
        "categorical_features": ['csm_name', 'state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2],
        "feature_selection": ("SelectKBest", 40)
    },
    "collinearity_reduced_a_cv": {
        "numeric_features": ['has_ise', 'integration_educlimber', 'integration_fast', 'integration_google_classroom',
                             'integration_pra', 'asmt_admin_flex', 'asmt_admin_ib', 'asmt_admin_inspect_prebuilt',
                             'cumulative_common_assessment_count_per_ay', 'hs_1_lag_band', 'hs_1year_lag_band',
                             'hs_4_lag_band', 'hs_5_lag_band', 'hs_6_lag_band', 'hs_7_lag_band', 'hs_band',
                             'matrix_distinct_users', 'mltp_asmt_smry_distinct_users',
                             'mltp_asmt_smry_times_accessed', 'rsp_freq_distinct_users',
                             'site_asmt_ovr_distinct_users',
                             'skills_letter_distinct_users', 'subscriber_tenure_days',
                             'summary_asmt_created', 'ticket_count', 'tile_layouts_created_modified',
                             'user_created_custom_reports', 'arr_dna',
                             'atd_students_assessed_percent', 'atd_users_login_percent',
                             'sa_tenure_in_days', 'teachers_login_percent'],
        "categorical_features": ['csm_name', 'state'],
        "weights": [{0: 35.0, 1: 65.0}],
        "kernel": ['rbf'],
        "c_value": [.25, .5, 1, 1.25]
    },
    "collinearity_reduced_kbest20_cv": {
        "numeric_features": ['has_ise', 'integration_educlimber', 'integration_fast', 'integration_google_classroom',
                             'integration_pra', 'asmt_admin_flex', 'asmt_admin_ib', 'asmt_admin_inspect_prebuilt',
                             'cumulative_common_assessment_count_per_ay', 'hs_1_lag_band', 'hs_1year_lag_band',
                             'hs_4_lag_band', 'hs_5_lag_band', 'hs_6_lag_band', 'hs_7_lag_band', 'hs_band',
                             'matrix_distinct_users', 'mltp_asmt_smry_distinct_users',
                             'mltp_asmt_smry_times_accessed', 'rsp_freq_distinct_users',
                             'site_asmt_ovr_distinct_users',
                             'skills_letter_distinct_users', 'subscriber_tenure_days',
                             'summary_asmt_created', 'ticket_count', 'tile_layouts_created_modified',
                             'user_created_custom_reports', 'arr_dna',
                             'atd_students_assessed_percent', 'atd_users_login_percent',
                             'sa_tenure_in_days', 'teachers_login_percent'],
        "categorical_features": ['csm_name', 'state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "kernel": ['rbf'],
        "c_value": [0.2],
        "feature_selection": ("SelectKBest", 20)
    }
}

csv_file = 'svm_model_evals.csv'
models_completed = pd.read_csv(csv_file, usecols = ["model_name", "kernel", "c"])
include_class_weights = True


#####################################
###           Functions           ###
#####################################

def connect_to_db():
    try:
        return psycopg2.connect(
            host = "localhost",
            database = "talend",
            user = "franck",
            password = "_REMOVED_",
            options = '-c statement_timeout=300000'
        )
    except Exception as err:
        print(f"ERROR: connect_to_db: {err}")
        return None


def run_query(sql):
    db_connection = connect_to_db()

    if db_connection is None:
        print(f'ERROR: Unable to establish connection to DB.')
        return

    cursor = db_connection.cursor()

    # sql = "SELECT student_id FROM students LIMIT 1"
    try:
        cursor.execute(sql)
    except Exception as err:
        print(f"ERROR: ", err)
        db_connection.close()
        return

    try:
        sql_results = cursor.fetchall()
    except Exception as err:
        print(f"ERROR:", err)
        db_connection.close()
        return

    print(f"Got results for query ({len(sql_results)})")

    cursor.close()
    db_connection.close()


#####################################
###         Import Data           ###
#####################################

refresh_data = False

if refresh_data:
    db_connection = connect_to_db()
    training_data_sql = "SELECT * FROM client_health_full_training_data_mv WHERE hs_did_change IS NOT NULL"
    print("Getting data...")
    full_data_set = sqlio.read_sql_query(training_data_sql, db_connection)
    db_connection.close()
    print("...query completed.")
    full_data_set.to_pickle('full_data_set.pkl')
else:
    full_data_set = pd.read_pickle('full_data_set.pkl')

# Remove records with missing values
full_data_set = full_data_set.dropna()


#####################################
###     Create Data Pipeline      ###
#####################################


def check_existing_models(current_model_name, current_kernel, current_c_val):
    for i, m in models_completed.iterrows():
        if m['model_name'] == current_model_name \
                and m['kernel'] == current_kernel\
                and m['c'] == current_c_val:
            return True

    return False


for model_name, model_dict in models_dict.items():

    for kernel in model_dict['kernel']:
        for c_val in model_dict['c_value']:
            if check_existing_models(model_name, kernel, c_val):
                print(f". . . . . . . . . . Skipping  {model_name}, kernel: {kernel}, C: {c_val}. . . . . . . . . . . .")
                continue

            print(f"""
            #########################################################
            > {model_name}, kernel: {kernel}, c: {c_val}                   
            #########################################################
            """)

            numeric_pipeline = Pipeline(steps = [
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps = [
                ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers = [
                    ('num', numeric_pipeline, model_dict['numeric_features']),
                    ('cat', categorical_pipeline, model_dict['categorical_features'])
                ]
            )

            if include_class_weights:
                svm_obj = svm.SVC(C = c_val, kernel = kernel, random_state = 1, class_weight = model_dict['weights'][0])
            else:
                svm_obj = svm.SVC(C = c_val, kernel = kernel, random_state = 1)

            model = Pipeline([
                ('preprocessor', preprocessor),
                ('svm_obj', svm_obj)
            ])

            if 'feature_selection' in model_dict:
                if model_dict['feature_selection'][0] == 'SelectKBest':
                    number_of_features = model_dict['feature_selection'][1]
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('select', SelectKBest(k = number_of_features)),
                        ('svm_obj', svm_obj)
                    ])

            X_column_names = model_dict['numeric_features'] + model_dict['categorical_features']

            covid_filter = [True, False]

            for cf in covid_filter:
                print(f">>> COVID Filter: {cf}")

                if cf:
                    before_covid = full_data_set["start_date"] < datetime.strptime("2020-01-01", "%Y-%m-%d").date()
                    after_covid = full_data_set["start_date"] > datetime.strptime("2020-06-01", "%Y-%m-%d").date()
                    wo_covid = before_covid | after_covid
                    training_data = full_data_set.loc[wo_covid]
                else:
                    training_data = full_data_set

                X = training_data[X_column_names]
                y = training_data['hs_did_change']

                # Test for Multicollinearity
                correlation_matrix = X.corr()
                sn.heatmap(correlation_matrix, annot = True)
                plt.savefig(f'{model_name}_corr_matrix.png')
                plt.show()

                k = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 19)

                eval_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                cv_results = cross_validate(estimator = model, X = X, y = y, cv = k, n_jobs = 15,
                                            scoring = eval_metrics, return_estimator = True, verbose = 4)

                cv_accuracy = np.mean(cv_results['test_accuracy'])
                cv_precision = np.mean(cv_results['test_precision'])
                cv_recall = np.mean(cv_results['test_recall'])
                cv_f1 = np.mean(cv_results['test_f1'])
                cv_roc_auc = np.mean(cv_results['test_roc_auc'])

                print(f"F1: {cv_f1} | AUC: {cv_roc_auc} | ACCURACY: {cv_accuracy} | PRECISION: {cv_precision} | RECALL: {cv_recall}")

                if 'feature_selection' in model_dict:
                    if model_dict['feature_selection'][0] == 'SelectKBest':
                        for estimator in cv_results['estimator']:
                            # print(f"X.columns.values: {X.columns.values}")

                            # Access fitted OneHotEncoder object and call get_feature_names_out()
                            encoderObj = estimator.steps[0][1].transformers_[1][1].named_steps['onehot']
                            exploded_feature_names = encoderObj.get_feature_names_out()
                            # print(f"exploded_feature_names: {exploded_feature_names}")

                            feature_names = model_dict['numeric_features'] + exploded_feature_names.tolist()
                            # print(f"feature_names: {feature_names}")

                            kbest_did_select = estimator.steps[1][1].get_support()
                            kbest_did_select_list = kbest_did_select.tolist()
                            # print(f"kbest_did_select: {kbest_did_select_list}")

                            selected_columns = list(compress(feature_names, kbest_did_select_list))
                            print(f"selected_columns: {selected_columns}")

                # Append to CSV

                with open(csv_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
                                     model_name,
                                     model_dict['numeric_features'] + model_dict['categorical_features'],
                                     model_dict['weights'][0] if include_class_weights else '',
                                     cv_accuracy, cv_roc_auc, cv_f1, cv_precision, cv_recall,
                                     kernel, cf, c_val
                                     ])

                print("fin.")
