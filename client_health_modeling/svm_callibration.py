import pickle
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
from datetime import datetime

models_dict = {
    "baseline_cv": {
        "numeric_features": ['atd_client_health_score'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "2features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "2features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "2features_c_cv": {
        "numeric_features": ['atd_client_health_score', 'site_asmt_ovr_times_accessed'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "3features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "3features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "3features_c_cv": {
        "numeric_features": ['hs_band', 'cumulative_common_assessment_count_per_ay', 'user_created_custom_reports'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "6features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "7features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "7features_b_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed'],
        "categorical_features": ['csm_name'],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "lag_only": {
        "numeric_features": ['hs_1year_lag_band', 'hs_7_lag_band', 'hs_6_lag_band', 'hs_5_lag_band', 'hs_4_lag_band',
                             'hs_1_lag_band'],
        "categorical_features": [],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "8features_a_cv": {
        "numeric_features": ['atd_client_health_score', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed',
                             'user_created_custom_reports'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
    },
    "8features_b_cv": {
        "numeric_features": ['hs_band', 'hs_7_lag_band', 'cumulative_common_assessment_count_per_ay',
                             'sa_tenure_in_days', 'subscriber_tenure_days', 'site_asmt_ovr_times_accessed',
                             'user_created_custom_reports'],
        "categorical_features": ['state'],
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
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
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
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
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100],
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
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100],
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
        "weights": [{0: 30.0, 1: 70.0}],
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100]
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
        "solver": ['lbfgs'],
        "c_value": [10, 30, 60, 100],
        "feature_selection": ("SelectKBest", 20)
    }
}
model_name = 'full_cv'
model_title = f"Full SVM Model"
model_subtitle = f"Cost Sensitive w/ COVID outliers removed"

#####################################
###          Load Model           ###
#####################################
svm_model = pickle.load(open('svm_full_cv_model_obj.sav', 'rb'))

#####################################
###           Functions           ###
#####################################

# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


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

# Remove COVID outliers
before_covid = full_data_set["start_date"] < datetime.strptime("2020-01-01", "%Y-%m-%d").date()
after_covid = full_data_set["start_date"] > datetime.strptime("2020-06-01", "%Y-%m-%d").date()
wo_covid = before_covid | after_covid
training_data = full_data_set.loc[wo_covid]

X_column_names = models_dict[model_name]['numeric_features'] + models_dict[model_name]['categorical_features']
X = training_data[X_column_names]
y = training_data['hs_did_change']


########################################
### Pre-Calibrated Reliability Curve ###
########################################
print("Generating pre-calibrated reliability curve...")
# predict probabilities
probs = svm_model.decision_function(X)
# reliability diagram
fop, mpv = calibration_curve(y, probs, n_bins=10, normalize=True)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
pyplot.plot(mpv, fop, marker='.')
pyplot.savefig(f'svm_pre-callibration_reliability_diagram_clean.png')
pyplot.show()

########################################
###   Calibrate Model Probabilities  ###
########################################
print("Calibrating model...")
calibrated_svm_model = CalibratedClassifierCV(svm_model, method='sigmoid', cv=5, n_jobs = 15)
calibrated_svm_model.fit(X, y)

#########################################
### Post-Calibrated Reliability Curve ###
#########################################
print("Generating post-calibration probabilities...")
# predict probabilities
cb_probs = calibrated_svm_model.predict_proba(X)[:, 1]
# reliability diagram
print("Generating post-calibration reliability curve...")
cb_fop, cb_mpv = calibration_curve(y, cb_probs, n_bins=10, normalize=True)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
pyplot.plot(cb_mpv, cb_fop, marker='.')
pyplot.savefig(f'svm_post-callibration_reliability_diagram_clean.png')
pyplot.show()

model_filename = f"svm_{model_name}_callibrated_model_obj_clean.sav"
pickle.dump(calibrated_svm_model, open(model_filename, 'wb'))

training_data_with_probs = training_data
training_data_with_probs['pred_probs'] = cb_probs.tolist()

training_data_with_probs.to_csv('training_data_with_calibrated_predicted_probabilites_clean.csv')

print("fin.")
