import pandas as pd
import os
import numpy as np
import glob
import math
from sys import argv 
from datetime import datetime
import calendar
#----------------------------------------------------------------------------
# Arguments follow the order as:
# 	1. Location of all output csv file from previous processing step
#----------------------------------------------------------------------------

full_loc = argv[1]
out_loc = argv[2]
#  Create new directory
def MakeDirectory (dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        pass

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
    return df
def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return(week_of_month)

def main():
  
    MakeDirectory(out_loc)
      
    all_files = glob.glob(full_loc + "*")
    
    # Get the set of column list
    for file in all_files:
        file_name = file.split('/')[-1].split('.')[0]
        print ("Current file", file_name)
        df = pd.DataFrame()
        for chunk in pd.read_csv(file, chunksize=5000):
            df = pd.concat([df, chunk])
        
	#df = pd.read_csv(file)
        df['week_month'] = df['ctime'].apply(lambda x: get_week_of_month(datetime.fromtimestamp(x).year, datetime.fromtimestamp(x).month, datetime.fromtimestamp(x).day))
        for name, group in df.groupby('week_month'):
            print ("Current group", name)
            out_file_name = out_loc + file_name +'_' + str(name) +'.csv'    
            group.drop(['week_month'], axis=1,inplace=True)
            print ("Group columns", group.columns, len(group.columns))
            group.to_csv(out_file_name, index=False)
if __name__ == "__main__":
    main()
