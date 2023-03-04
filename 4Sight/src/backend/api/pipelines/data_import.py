import sys
import pandas as pd
import psycopg2
from psycopg2 import sql

# def file_converter(path):
#   """
#   Function which splits dataframe into train and test sets
#   Parameters
#   ----------
#   path
#     path of the file to convert

#   Returns
#   -------
#   path : str
#     input path if not excel
#   new_path : str
#     new path from input path with csv extension
#   """
#   head, sep, tail = path.partition('.')
#   match tail.lower():
#     case 'xls' | 'xlsx' | 'xlsm' | 'xlsb' | 'odf' | 'ods' | 'odt':
#       read_file = pd.read_excel(path)
#       new_path = head + '.csv'
#       read_file.to_csv(new_path, index=None, header=True)
#       return new_path
#     case 'csv':
#       return path
#     case _:
#       sys.exit("Your file is not a tabular format please use csv or Excel like file.")
          

def read_data(path, date_col):  
  """
  Function which splits dataframe into train and test sets
  Parameters
  ----------
  path : str
    path of the file to read
  date_col : list
    date column name(s) to parse it while reading data
  
  Returns
  -------
  df : pandas DataFrame
    data read from path
  """
  data = pd.read_csv(path,
                    parse_dates={'Date_parsed': date_col},
                    infer_datetime_format=True,
                    on_bad_lines='warn',
                    skip_blank_lines=True)

  try:
    df = data.dropna(axis=0, subset=['Date_parsed'])
    df = df.set_index('Date_parsed')
    df = df.sort_index()
  except:
    print("Unexpected error:", sys.exc_info()[0])
    print(df)
    print('\n', df.dtypes)
  return df


def write_dataframe_to_postgres(df, table_name, db_config):
    # Connect to the database
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Define the column names and types
    columns = df.columns.tolist()
    column_types = ['VARCHAR(255)' for col in columns]

    # Create the table
    create_table_query = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
        sql.Identifier(table_name),
        sql.SQL(',').join(
            sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(col_type))
            for col, col_type in zip(columns, column_types)
        )
    )
    cur.execute(create_table_query)

    data = [tuple(row) for row in df.to_numpy()]                # Convert the DataFrame to a list of tuples

    # Define the insert statement
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(table_name),
        sql.SQL(',').join(map(sql.Identifier, columns)),
        sql.SQL(',').join(sql.Placeholder() * len(columns))
    )

    cur.executemany(insert_query, data)                         # Insert the data
    conn.commit()                                               # Commit the transaction

    # Close the cursor and connection
    cur.close()
    conn.close()

    print(f"Successfully wrote {len(data)} rows to table {table_name} in database {db_config['database']}!")
    

def parse_date_column(df):
    date_columns = []
    df_col = df.columns
    for col in df_col:
        try:
            parsed_date = pd.to_datetime(df[col], infer_datetime_format=True)
            if pd.Series(parsed_date).notna().all():
                date_columns.append(col)
                df[col] = parsed_date
        except ValueError:
            pass

    return df, date_columns


def read_table_from_postgres(table_name, db_config):
    conn = psycopg2.connect(**db_config)                        # Connect to the database
    df = pd.read_sql(f"SELECT * FROM \"{table_name}\"", conn)   # Read the table into a DataFrame
    conn.close()                                                # Close the connection
    df, date_column = parse_date_column(df)
    df.rename(columns={date_column[0]: "Date_parsed"})
    df = df.dropna(axis=0, subset=['Date_parsed'])
    df = df.set_index('Date_parsed')
    df = df.sort_index()

    return df