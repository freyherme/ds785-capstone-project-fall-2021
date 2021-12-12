##############################################################################
########         Support Vector Machine - Plot ROC Curve              ########
##############################################################################

#####################################
###            Packages           ###
#####################################

from datetime import datetime
import csv
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
from numpy import arange
from numpy import argmax
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import pickle
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



kernel = 'rbf'
c_val = 0.2
include_class_weights = True
covid_filter = [True]

# model_name = 'baseline_cv'
# model_title = f"Baseline SVM Model"
# model_subtitle = f"{'Cost Sensitive' if include_class_weights else ''}"

model_name = 'full_cv'
model_title = f"Full SVM Model"
model_subtitle = f"Cost Sensitive w/ COVID outliers removed"

csv_file = 'svm_model_evals.csv'

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
    full_data_set = pd.read_pickle('full_data_set_paper.pkl')

# Remove records with missing values
full_data_set = full_data_set.dropna()

#####################################
###     Create Data Pipeline      ###
#####################################

print(f"""
##############################################################################
> {model_name}, solver: {kernel}, c: {c_val}                   
##############################################################################
""")

numeric_pipeline = Pipeline(steps = [
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_pipeline, models_dict[model_name]['numeric_features']),
        ('cat', categorical_pipeline, models_dict[model_name]['categorical_features'])
    ]
)

if include_class_weights:
    svm_obj = svm.SVC(C = c_val, kernel = kernel, random_state = 1, class_weight = models_dict[model_name]['weights'][0], probability = True)
else:
    svm_obj = svm.SVC(C = c_val, kernel = kernel, random_state = 1, probability = True)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('svm_obj', svm_obj)
])

if 'feature_selection' in models_dict[model_name]:
    if models_dict[model_name]['feature_selection'][0] == 'SelectKBest':
        number_of_features = models_dict[model_name]['feature_selection'][1]
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('select', SelectKBest(k = number_of_features)),
            ('svm_obj', svm_obj)
        ])

X_column_names = models_dict[model_name]['numeric_features'] + models_dict[model_name]['categorical_features']

for cf in covid_filter:

    if cf:
        before_covid = full_data_set["start_date"] < datetime.strptime("2020-01-01", "%Y-%m-%d").date()
        after_covid = full_data_set["start_date"] > datetime.strptime("2020-06-01", "%Y-%m-%d").date()
        wo_covid = before_covid | after_covid
        training_data = full_data_set.loc[wo_covid]
    else:
        training_data = full_data_set

    X = training_data[X_column_names]
    y = training_data['hs_did_change']

    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify = y, train_size = 0.8)

    model_obj = model.fit(X = x_train, y = y_train)
    model_filename = f"svm_{model_name}_model_obj_split.sav"
    pickle.dump(model_obj, open(model_filename, 'wb'))

    if 'feature_selection' in models_dict[model_name]:
        if models_dict[model_name]['feature_selection'][0] == 'SelectKBest':
            # print(f"X.columns.values: {X.columns.values}")

            # Access fitted OneHotEncoder object and call get_feature_names_out()
            encoderObj = model_obj.steps[0][1].transformers_[1][1].named_steps['onehot']
            exploded_feature_names = encoderObj.get_feature_names_out()
            # print(f"exploded_feature_names: {exploded_feature_names}")

            feature_names = models_dict[model_name]['numeric_features'] + exploded_feature_names.tolist()
            # print(f"feature_names: {feature_names}")

            kbest_did_select = model_obj.steps[1][1].get_support()
            kbest_did_select_list = kbest_did_select.tolist()
            # print(f"kbest_did_select: {kbest_did_select_list}")

            selected_columns = list(compress(feature_names, kbest_did_select_list))
            print(f"selected_columns: {selected_columns}")

    preds = model_obj.predict(x_test)
    predicted_probs = model_obj.predict_proba(x_test)

    # "No Skill" predictions to draw line in ROC curve
    ns_probs = [0 for _ in range(len(y_test))]
    svm_probs = predicted_probs[:, 1]

    ns_auc_score = metrics.roc_auc_score(y_test, ns_probs)
    auc_score = metrics.roc_auc_score(y_test, svm_probs)
    print(f"auc_score: {round(auc_score, 4)}")

    f1_score = metrics.f1_score(y_test, preds)
    print(f"f1_score: {f1_score}")

    # Calculate ROC
    ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, svm_probs)

    # Plot Simple ROC
    plt.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'No Skill')
    plt.plot(lr_fpr, lr_tpr, marker = '.', label = 'SVM')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle(f'{model_title}', fontsize = 14)
    plt.title(f"{model_subtitle}", fontsize = 10)
    # show the legend
    plt.legend()
    # show the plot
    file_name = f"{model_name}_weighted_split" if include_class_weights else model_name
    plt.savefig(f'{file_name}_simple_auc_plot.png')
    plt.show()

    # Plot multi-class ROC
    skplt.metrics.plot_roc(y_test, predicted_probs)
    plt.suptitle(f'{model_title}', fontsize = 14)
    plt.title(f"{model_subtitle}", fontsize = 10)
    multiclass_filename = f'{file_name}_multiclass_auc_plot.png'
    plt.savefig(multiclass_filename)
    plt.show()

    #####################
    # Threshold Tuning ##
    #####################

    # DEFINE THRESHOLDS
    thresholds = arange(0, 1, 0.01)
    # EVALUATE EACH THRESHOLD
    scores = [metrics.f1_score(y_test, to_labels(svm_probs, t)) for t in thresholds]
    # GET BEST THRESHOLD
    ix = argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    best_threshold = thresholds[ix]
    best_threshold_preds = [1 if prob > best_threshold else 0 for prob in svm_probs]

    best_threshold_accuracy = round(metrics.accuracy_score(y_test, best_threshold_preds), 4)
    best_threshold_auc = round(metrics.roc_auc_score(y_test, best_threshold_preds), 4)
    best_threshold_f1 = round(metrics.f1_score(y_test, best_threshold_preds), 4)
    best_threshold_precision = round(metrics.precision_score(y_test, best_threshold_preds), 4)
    best_threshold_recall = round(metrics.recall_score(y_test, best_threshold_preds), 4)

    print(
        f"F1: {best_threshold_f1} | AUC: {best_threshold_auc} | ACCURACY: {best_threshold_accuracy} | PRECISION: {best_threshold_precision} | RECALL: {best_threshold_recall}")

    # Append to CSV

    with open(csv_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
                         f"{model_name}_threshold_tuned_split",
                         models_dict[model_name]['numeric_features'] + models_dict[model_name]['categorical_features'],
                         models_dict[model_name]['weights'][0] if include_class_weights else '',
                         best_threshold_accuracy, round(auc_score, 4), best_threshold_f1, best_threshold_precision, best_threshold_recall,
                         kernel, cf, c_val
                         ])

print("fin.")
