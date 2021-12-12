
----------------------------------
--       3 Month Window         --
----------------------------------

WITH
	usage_rollup_scrubbed AS (
		SELECT *
		FROM mv_monthly_usage_rollup
		WHERE DATE_PART('month', start_date) NOT IN (6, 7, 8, 12)
		  AND atd_database IS NOT NULL
	),

	client_health_3month_window AS (
		SELECT
			hs.sf_an,
			hs.atd_database,
-- 		mr.atd_client_health_score,
			hs.atd_client_health_score_band AS hs_band,
			CASE
				WHEN hs.atd_client_health_score_band IN ('Red', 'Yellow') THEN 1
				WHEN hs.atd_client_health_score_band = 'Green'            THEN 2
				END AS band_number,
			hs.start_date,
			MIN(CASE
				WHEN hs2.atd_client_health_score_band IN ('Red', 'Yellow') THEN 1
				WHEN hs2.atd_client_health_score_band = 'Green'            THEN 2
				END
				) AS interval_min_band_number,
			MAX(CASE
				WHEN hs2.atd_client_health_score_band IN ('Red', 'Yellow') THEN 1
				WHEN hs2.atd_client_health_score_band = 'Green'            THEN 2
				END
				) AS interval_max_band_number

		FROM usage_rollup_scrubbed hs
		LEFT JOIN usage_rollup_scrubbed hs2
				  ON hs2.start_date <= (hs.start_date + '3 months'::interval)
					  AND hs2.start_date > hs.start_date
					  AND hs.atd_database = hs2.atd_database
		GROUP BY
			hs.sf_an,
			hs.atd_database,
			hs.atd_client_health_score,
			hs.atd_client_health_score_band,
			hs.start_date
		ORDER BY atd_database, start_date
	),

	hs_did_change AS (

		SELECT *,
			   CASE WHEN band_number = 2
						THEN (
					   CASE
						   WHEN interval_min_band_number IS NULL       THEN NULL
						   WHEN band_number = interval_min_band_number THEN FALSE
						   ELSE TRUE
						   END
					   )
					ELSE (
						CASE
							WHEN interval_max_band_number IS NULL       THEN NULL
							WHEN band_number = interval_max_band_number THEN FALSE
							ELSE TRUE
							END
						)
					END
				   AS hs_did_change,
				CASE WHEN band_number = 2
						THEN (
					   CASE
						   WHEN interval_min_band_number IS NULL       THEN NULL
						   WHEN band_number = interval_min_band_number THEN 'Stayed Green'
						   ELSE 'Green to Red/Yellow'
						   END
					   )
					ELSE (
						CASE
							WHEN interval_max_band_number IS NULL       THEN NULL
							WHEN band_number = interval_max_band_number THEN 'Stayed Red/Yellow'
							ELSE 'Red/Yellow to Green'
							END
						)
					END
				   AS hs_change_type
		FROM client_health_3month_window

	)

-- ##########################
--          Raw Data
-- ##########################

	SELECT *,
		CASE WHEN hs_change_type = 'Stayed Green' THEN 4
			WHEN hs_change_type = 'Red/Yellow to Green' THEN 3
			WHEN hs_change_type = 'Green to Red/Yellow' THEN 2
			WHEN hs_change_type = 'Stayed Red/Yellow' THEN 1
			END AS change_type_order
	FROM hs_did_change
	ORDER BY atd_database, start_date;



-- ##########################
--  Is the dataset balanced?
-- ##########################


-----------------------------
--   hs_did_change counts  --
-----------------------------

-- SELECT hs_did_change, COUNT(*), COUNT(DISTINCT atd_database) AS client_count
-- FROM hs_did_change
-- GROUP BY hs_did_change;


-----------------------------
--   change_type counts  --
-----------------------------

-- SELECT hs_change_type, COUNT(*), COUNT(DISTINCT atd_database) AS client_count
-- FROM hs_did_change
-- -- WHERE hs_did_change.start_date = '2021-03-01'
-- GROUP BY hs_change_type

;