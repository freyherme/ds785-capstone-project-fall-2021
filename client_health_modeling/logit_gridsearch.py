####################################################################################
########      Logistic Regression Hyper-parameter Tuning Grid Search        ########
####################################################################################

#####################################
###            Packages           ###
#####################################

from datetime import datetime
import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#####################################
###            Inputs             ###
#####################################

model_name = 'collinearity_reduced_a_cv'
models_dict = {
    "baseline_cv": {
        "numeric_features": ['atd_client_health_score'],
        "categorical_features": [],
        "weights": [{0: 20.0, 1: 80.0}],
        "solver": ['lbfgs']
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
        "weights": [{0: 20.0, 1: 80.0}],
        "solver": ['lbfgs']
    }
}


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

model_dict = models_dict[model_name]

# Remove records with missing values
full_data_set = full_data_set.dropna()
before_covid = full_data_set["start_date"] < datetime.strptime("2020-01-01", "%Y-%m-%d").date()
after_covid = full_data_set["start_date"] > datetime.strptime("2020-06-01", "%Y-%m-%d").date()
wo_covid = before_covid | after_covid
training_data = full_data_set.loc[wo_covid]

X_column_names = model_dict['numeric_features'] + model_dict['categorical_features']
X = training_data[X_column_names]
y = training_data['hs_did_change']

numeric_pipeline = Pipeline(steps = [
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps = [
    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))
])

lor = LogisticRegression()

preprocessor = ColumnTransformer(
    transformers = [
        ('num', numeric_pipeline, model_dict['numeric_features']),
        ('cat', categorical_pipeline, model_dict['categorical_features'])
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('lor', lor)
])

param_grid = [{
    'lor__C': [80, 85, 90, 100, 150],
    'lor__class_weight': [{0: 35.0, 1: 75.0}, {0: 30.0, 1: 70.0}, {0: 25.0, 1: 75.0}, {0: 20.0, 1: 80.0}],
    'lor__max_iter': [5000]
}]

k = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 19)
logit_grid_search = GridSearchCV(model,
                                 param_grid = param_grid,
                                 scoring = 'f1',
                                 n_jobs = 15,
                                 cv = k,
                                 verbose = 4)

logit_grid_search.fit(X, y)

print(f"{model_name}_svm_grid_search.best_params_: {logit_grid_search.best_params_}")
print(f"{model_name}_svm_grid_search.best_score: {logit_grid_search.best_score_}")
print(f"{model_name}_svm_grid_search.param_grid: {logit_grid_search.param_grid}")
print("fin.")
