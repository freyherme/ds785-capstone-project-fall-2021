SELECT
	u.sf_an,
	u.state,
	u.renewal_this_year,
	u.renewal_next_year,
	u.csm_health,
	u.atd_database,
	u.site_name,
	u.atd_site_name,
	u.csm_name,
	u.start_date,
	u.atd_client_health_score,
	u.atd_client_health_score_band,

	atd_users_login_percent_score,
	atd_users_login_percent,
	atd_users_login_percent_actual,

	atd_users_count,

	atd_students_assessed_percent_score,
	atd_students_assessed_percent,
	atd_students_assessed_percent_actual,

	atd_student_count,

	atd_feature_adoption_score,
	atd_feature_adoption_score_band,

	asmt_created_flex,
	asmt_admin_flex,
	asmt_created_ib,
	asmt_admin_ib,
	user_created_custom_reports,

	asmt_created_flex_score,
	asmt_admin_flex_score,
	asmt_created_ib_score,
	asmt_admin_ib_score,
	user_created_custom_reports_score


FROM mv_monthly_usage_rollup u
WHERE DATE_PART('month', start_date) NOT IN (6, 7, 8, 12)
  AND atd_database IS NOT NULL
  AND atd_users_count > 0