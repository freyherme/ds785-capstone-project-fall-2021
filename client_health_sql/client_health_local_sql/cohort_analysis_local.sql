-- Cohort Analysis

WITH
	record_count AS (
		SELECT COUNT(*) AS row_count
		FROM mv_monthly_usage_rollup_scrubbed
	),

	ca_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ca.cumulative_common_assessment_count_per_ay AS rate,
			RANK()
			OVER (ORDER BY ca.cumulative_common_assessment_count_per_ay) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ca.cumulative_common_assessment_count_per_ay DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_common_assessments ca
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ca.sf_an = ur.sf_an
				 AND ca.month_start::date = ur.start_date
		JOIN record_count ON TRUE

	),

	ca_cohort_analysis AS (
		-- 		SELECT * FROM cr_buckets

		SELECT
			'Cummulative Common Assessments Count (per AY)' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM ca_buckets

		GROUP BY bucket
		ORDER BY bucket DESC

	),

	matrix_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Assessment Matrix Report'
	),

	matrix_cohort_analysis AS (

		SELECT
			matrix_buckets.title || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM matrix_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	asmt_stu_overview_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Assessment Student Overview'
	),

	asmt_stu_overview_cohort_analysis AS (

		SELECT
			asmt_stu_overview_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM asmt_stu_overview_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	tchr_asmt_overview_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Teacher Assessment Overview'
	),

	tchr_asmt_overview_cohort_analysis AS (

		SELECT
			tchr_asmt_overview_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM tchr_asmt_overview_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	site_asmt_overview_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Site Assessment Overview'
	),

	site_asmt_overview_cohort_analysis AS (

		SELECT
			site_asmt_overview_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM site_asmt_overview_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	mltp_asmt_summary_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Multiple Assessment Summary Report'
	),

	mltp_asmt_summary_cohort_analysis AS (

		SELECT
			mltp_asmt_summary_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM mltp_asmt_summary_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	skills_asmt_letter_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Skills Assessment Parent Letter'
	),

	skills_asmt_letter_cohort_analysis AS (

		SELECT
			skills_asmt_letter_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM skills_asmt_letter_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	response_freq_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Assessment Response Frequency'
	),

	response_freq_cohort_analysis AS (

		SELECT
			response_freq_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM response_freq_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	asmt_tchr_pc_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Assessment Teacher Peer Comparison'
	),

	asmt_tchr_pc_cohort_analysis AS (

		SELECT
			asmt_tchr_pc_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM asmt_tchr_pc_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	asmt_site_pc_buckets AS (
		SELECT
			ur.sf_an,
			ur.start_date,
			ur.atd_client_health_score,
			ru.title,
			ru.times_accessed / ur.atd_users_count::numeric AS rate,
			RANK()
			OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric) AS rank,
			FLOOR(RANK()
				  OVER (ORDER BY ru.times_accessed / ur.atd_users_count::numeric DESC) /
				  row_count::numeric * 10) AS bucket
		FROM dna_prebuilt_report_usage ru
		JOIN mv_monthly_usage_rollup_scrubbed ur
			 ON ru.sf_an = ur.sf_an
				 AND ru.month_start::date = ur.start_date
		JOIN record_count ON TRUE
		WHERE ru.title = 'Assessment Site Peer Comparison'
	),

	asmt_site_pc_cohort_analysis AS (

		SELECT
			asmt_site_pc_buckets.title  || ' Runs p/ User' AS cohort,

			-- Calculate rank of cr p/user rate.
			-- Multiply by 10 and divide by client count to create 10 buckets per month if possible
			bucket,
			'[' || MIN(ROUND(rate, 3)) || ', ' || MAX(ROUND(rate, 3)) || ']' AS range,
			AVG(atd_client_health_score) AS avg_hs,
			COUNT(*) AS n

		FROM asmt_site_pc_buckets

		GROUP BY title, bucket
		ORDER BY bucket DESC

	),

	complete_cohort_analyis AS (

		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM asmt_site_pc_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM asmt_tchr_pc_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM response_freq_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM skills_asmt_letter_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM mltp_asmt_summary_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM site_asmt_overview_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM tchr_asmt_overview_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM asmt_stu_overview_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM matrix_cohort_analysis
		UNION ALL
		SELECT *, SUM(n) OVER (PARTITION BY TRUE) AS total_n
		FROM ca_cohort_analysis
		UNION ALL
		SELECT *
		FROM cohort_analysis
	)

	--

	SELECT * FROM complete_cohort_analyis;

-- 	SELECT DISTINCT cohort FROM complete_cohort_analyis;

