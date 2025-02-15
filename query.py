import pandas as pd
import mysql.connector

def fetch_milk_production():
    conn = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        password='*Database630803240081',
        database='livelihoodzones'
    )

    query = """
        SELECT * FROM kia_questionnaire_sessions
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # Create a date field from month and year
    # df['date'] = pd.to_datetime(df['month'].astype(str) + ' ' + df['year'].astype(str), format='%B %Y')

    # Convert DataFrame to JSON and return
    return df.to_dict(orient='records')
