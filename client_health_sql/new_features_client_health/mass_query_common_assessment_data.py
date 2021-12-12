import csv
from multiprocessing import Process, Pool, Queue
import psycopg2


def connect_to_db(server_no, client):
    try:
        return psycopg2.connect(
            host = f"10.21.0.1{server_no}",
            database = client,
            user = "phppgadmin",
            password = "_REMOVED_"
        )
    except:
        return None


def get_db_connection(server_no, client = 'postgres'):
    db_connection = connect_to_db(server_no, client)
    return db_connection


def get_list_of_dbs(server_no):
    print(f"get_list_of_dbs for server {server_no}.......")
    db_connection = get_db_connection(server_no)

    if db_connection is None:
        return []

    cursor = db_connection.cursor()
    sql = f"""
        SELECT d.datname
        FROM pg_catalog.pg_database d
        """

    try:
        cursor.execute(sql)
    except Exception as err:
        print("ERROR: ", err)
        return

    sql_results = cursor.fetchall()
    cursor.close()
    db_connection.close()

    # print(f"get_list_of_dbs - sqlresults: {sql_results}")
    list_of_dbs = []
    for row in sql_results:
        if 'sandbox' in row[0]:
            continue
        list_of_dbs.append(row[0])

    return list_of_dbs


def get_data(client, server_no, q, server_index, db_index):
    print(f"DB {client}...............")

    db_connection = get_db_connection(server_no, client = client)

    if db_connection is None:
        q.put([])
        return 'ERROR: Unable to establish connection to DB.'

    cursor = db_connection.cursor()
    sql = """
        WITH
            sf_an AS (
                SELECT
                    COALESCE(CASE
                                 WHEN EXISTS(SELECT definition_id
                                             FROM config.definitions
                                             WHERE key ILIKE 'salesforce.account_number')
                                     THEN (
                                     SELECT value::text
                                     FROM config.entries
                                     WHERE definition_id = (
                                         SELECT definition_id
                                         FROM config.definitions
                                         WHERE key ILIKE 'salesforce.account_number'
                                     )
                                 )
                                 END, 'none'::text) AS sf_an
            ),
        
            ay_start_end AS (
                SELECT
                    academic_year,
                    MIN(start_date) AS start_date,
                    MAX(end_date) AS end_date
                FROM session_dates
                GROUP BY academic_year
            ),
        
            MONTHS AS (
                SELECT
                        DATE_TRUNC('month', CURRENT_DATE) - (INTERVAL '1 MONTH' * GENERATE_SERIES(0, 48)) AS month_start,
                        DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 MONTH' - INTERVAL '1 DAY' -
                        (INTERVAL '1 MONTH' * GENERATE_SERIES(0, 48)) AS month_end
            ),
        
            stu_count_per_assessment_per_site AS (
                SELECT
                    month_start,
                    se.academic_year,
                    asr.assessment_id,
                    a.title,
                    se.site_id,
                    COUNT(asr.student_id) AS stu_count_per_site
        
                FROM dna_assessments.agg_student_responses asr
                JOIN student_session_aff ssa
                     ON asr.student_id = ssa.student_id
                         AND asr.date_taken BETWEEN ssa.entry_date AND ssa.leave_date
                JOIN sessions se USING (session_id)
                JOIN dna_assessments.assessments a USING (assessment_id)
                JOIN MONTHS ON asr.date_taken BETWEEN MONTHS.month_start AND MONTHS.month_end
                GROUP BY se.academic_year, month_start, asr.assessment_id, a.title, se.site_id
        
                ORDER BY month_start DESC, a.title, stu_count_per_site DESC
            ),
        
            site_count_per_assessment AS (
                SELECT
                    month_start,
                    academic_year,
                    assessment_id,
                    title,
                    COUNT(site_id) AS site_count_per_assessment
                FROM stu_count_per_assessment_per_site scpaps
                WHERE stu_count_per_site > 10
                GROUP BY scpaps.academic_year, month_start, scpaps.assessment_id, scpaps.title
                HAVING COUNT(site_id) > 1
                ORDER BY month_start DESC, site_count_per_assessment DESC, title
            ),
        
            common_assessments_per_month AS (
                SELECT
                    academic_year,
                    month_start,
                    COUNT(assessment_id) AS common_assessment_count,
                    SUM(COUNT(assessment_id)) OVER (
                        PARTITION BY academic_year
                        ORDER BY month_start ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS cumulative_common_assessment_count_per_ay
        
                FROM site_count_per_assessment
                GROUP BY academic_year, month_start
                ORDER BY month_start DESC
            )
        
        SELECT
            sf_an,
            month_start,
            ay_start_end.academic_year,
            COALESCE(ca.common_assessment_count, 0) AS common_assessment_count,
            COALESCE(ca.cumulative_common_assessment_count_per_ay, 0) AS cumulative_common_assessment_count_per_ay
        
        FROM months
        JOIN ay_start_end ON months.month_start BETWEEN ay_start_end.start_date AND ay_start_end.end_date
        LEFT JOIN common_assessments_per_month ca USING (month_start)
        JOIN sf_an ON TRUE
        ;
        """

    try:
        cursor.execute(sql)
    except Exception as err:
        print("ERROR: ", err)
        q.put([])
        return

    if db_index == 0 and server_index == 0:
        with open('dna_common_assessments.csv', 'a') as header_file:
            header_wtr = csv.writer(header_file, delimiter = ',', lineterminator = '\n')
            colnames = [desc[0] for desc in cursor.description]
            colnames.insert(0, 'client')
            header_wtr.writerow(colnames)

    sql_results = cursor.fetchall()
    # print(sql_results)

    cursor.close()
    db_connection.close()

    rowArrays = []

    for row in sql_results:
        rowArray = [value for value in row]
        rowArray.insert(0, client)
        rowArrays.append(rowArray)

    q.put(rowArrays)

    print(f"...............DB {client}")


if __name__ == '__main__':
    list_of_servers = []
    for i in range(1, 32):
        list_of_servers.append(f"{i:02d}")

    for i in range(70, 80):
        list_of_servers.append(f"{i:02d}")

    # list_of_servers = ['03']
    print(list_of_servers)

    for server_index, server in enumerate(list_of_servers):

        # server_data = []

        if int(server) < 0:
            continue

        print("====================================")
        print(f"Server: {server}")
        print("====================================")
        list_of_dbs = get_list_of_dbs(server)
        # print(list_of_dbs)

        my_q = Queue()
        server_array = []

        processes = []

        db_index = 0
        for db in list_of_dbs:

            if db in ['postgres']:
                continue
            if 'template' in db:
                continue
            if 'portal' in db:
                continue
            if 'demo' in db:
                continue
            if 'jasper' in db:
                continue
            if 'job_queue' in db:
                continue
            if 'client' in db:
                continue
            if 'sqlboss' in db:
                continue
            if 'instance' in db:
                continue
            if db[0] == '_':
                continue

            p = Process(target = get_data, args = (db, server, my_q, server_index, db_index))
            db_index += 1
            processes.append(p)
            p.start()

        for i in range(len(processes)):
            server_array += my_q.get()

        for process in processes:
            process.join()

        with open('dna_common_assessments.csv', 'a') as file:
            wtr = csv.writer(file, delimiter = ',', lineterminator = '\n')
            for arr in server_array:
                wtr.writerow(arr)

        print(f"Wrote to csv file for server {server}.")
