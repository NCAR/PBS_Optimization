import pandas as pd
import os
import numpy as np
import glob
import math
from sys import argv 
from datetime import datetime
import calendar
from datetime import datetime
#----------------------------------------------------------------------------
# Arguments follow the order as:
# 	1. Location of all output csv file from previous processing step
#	2. Location of all output csv file (after cleaning bad data)
#	3. Location of all output csv file (after augmenting features)
#----------------------------------------------------------------------------

full_loc = argv[1]
out_loc = argv[2]
cleaned_loc = argv[3]
#  Create new directory
def MakeDirectory (dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        pass

def filter_data(df):
	num_jobs = []
	user = []
	for name, group in df.groupby('user'):
		num_jobs.append(len(group))
		user.append(name)
	
	num_jobs = np.array(num_jobs)
	user = np.array(user)
	
	index_top = np.argsort(num_jobs)[-10:]
	top_user = user[index_top]
	groups = []
	for name, group in df.groupby('user'):
		if (name in top_user):
			groups.append(group)	
	return pd.concat(groups, axis=0)

# Clean up bad data (should never happen)
def filter_bad_data(df):
	df = pd.to_numeric(df, downcast='float',errors='ignore')
	df = df[(df >= 0.0).all(1)
	df = df[df['Exit_status'] <= 256]
	df = df[df['Resource_List.nodect'] < 4032*72]
	#df['date'] = df['ctime'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y/%m/%d %H:%M:%S'))

	return df


def augment_features(full_loc, out_loc):
    all_files = glob.glob(full_loc + "*")

    # Get the set of column list
    column_list = set()
    for file in all_files:
        print ("Current file", file)
        df_training = pd.read_csv(file)
        for col in df_training.columns:
               column_list.add(col)
    print ("Full set of columns", column_list)
    MakeDirectory(out_loc)
    for file in all_files:
        file_name = file.split('/')[-1]
        df_training = pd.read_csv(file)
        for col in column_list:
           if not (col in df_training.columns):
               df_training[col] = np.zeros(shape = (df_training.shape[0], 1))
        print ("Final test num features", len(df_training.columns))
        df_training.to_csv(out_loc + file_name, index=False)



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
        print ("Original data", df.shape)
        #df = filter_data(df)
        df = filter_bad_data(df)
        print ("New data", df.shape)
        df.to_csv(out_loc + file_name + '.csv')
        #for name, group in df.groupby('week_month'):
        #    print ("Current group", name)
        #    out_file_name = out_loc + file_name +'_' + str(name) +'.csv'    
        #    group.drop(['week_month'], axis=1,inplace=True)
        #    print ("Group columns", group.columns, len(group.columns))
        #    group.to_csv(out_file_name, index=False)
    augment_features(out_loc, cleaned_loc)

if __name__ == "__main__":
    main()
