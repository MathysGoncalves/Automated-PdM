import sys
import pandas as pd


def file_converter(path):
  """
  Function which splits dataframe into train and test sets
  Parameters
  ----------
  path
    path of the file to convert

  Returns
  -------
  path : str
    input path if not excel
  new_path : str
    new path from input path with csv extension
  """
  head, sep, tail = path.partition('.')
  match tail.lower():
    case 'xls' | 'xlsx' | 'xlsm' | 'xlsb' | 'odf' | 'ods' | 'odt':
      read_file = pd.read_excel(path)
      new_path = head + '.csv'
      read_file.to_csv(new_path, index=None, header=True)
      return new_path
    case 'csv':
      return path
    case _:
      sys.exit("Your file is not a tabular format please use csv or Excel like file.")
          

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
