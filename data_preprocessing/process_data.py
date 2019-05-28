import pandas as pd
import os
import numpy as np
import glob
import math
csv_dir = './csv_output/'
training_loc = './training/'
full_loc = './full/'

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

def main():
    MakeDirectory(training_loc)
    MakeDirectory(full_loc)
    t_day = 24* 3600
    t_week = 7 * 24 * 3600
    all_files = glob.glob(full_loc + "/*")
    for file in all_files:

        df_training = pd.read_csv(file)

        # Feature extraction
        df_training = df_training.drop(
            ['Exit_status', 'session', 'exec_vnode', 'exec_host', 'ID', 'rtime', 'ctime', 'etime', 'start', 'run_count'], axis=1)
        print ("Checking column", df_training.columns)
        groups = []
        for name, group in df_training.groupby('user'):
            print("Group", group.shape)
            if (group.shape[0] >= 3):
                group['prev3jobs_runtime'] = group['resources_used.walltime'].shift(3).fillna(0)

            else:

                group['prev3jobs_runtime'] = np.zeros(shape=(group.shape[0],))

            if (group.shape[0] >= 2):
                group['prev2jobs_runtime'] = group['resources_used.walltime'].shift(2).fillna(0)
            else:

                group['prev2jobs_runtime'] = np.zeros(shape=(group.shape[0],))

            if (group.shape[0] >= 1):
                group['prevjob_runtime'] = group['resources_used.walltime'].shift(1).fillna(0)
            else:

                group['prevjob_runtime'] = np.zeros(shape=(group.shape[0],))

            group['avg_last2jobs'] = (group['resources_used.walltime'].shift(2).fillna(0) + group[
                'resources_used.walltime'].shift(1).fillna(0)) / 2.0
            group['avg_last3jobs'] = (group['resources_used.walltime'].shift(3).fillna(0) +
                                      group['resources_used.walltime'].shift(2).fillna(0) +
                                      group['resources_used.walltime'].shift(1).fillna(0)) / 3.0

            for name in group.columns:
                if ((not 'host' in name) and (not 'place' in name) and (not 'nodetype' in name) and (not 'switchblade' in name) and (('Resource_List' in name) or ('select' in name))):
                    avg_hist_all = np.zeros(shape=(group.shape[0],))
                    normalized = np.zeros(shape=(group.shape[0],))
                    print ("Column issue", name)
                    for i in range (1, group.shape[0]):
                        avg_hist_all[i] = np.mean(group[name].iloc[0:i])
                        normalized[i] = group[name].iloc[i] / (avg_hist_all[i]+ 1e-20)

                    group['avg_hist_' + name] = avg_hist_all
                    group['normalized_' + name] = normalized
                    #print ("Averaging features: ", name)
            group['avg_hist_runtime'] = avg_hist_all

            curr_run_jobs = np.zeros(shape=(group.shape[0],))
            curr_run_time = np.zeros(shape=(group.shape[0],))

            for name in group.columns:
                if not ('avg' in name) and not ('normalized' in name):
                    if ((not 'host' in name) and (not 'place' in name) and (not 'nodetype' in name) and (('Resource_List' in name) or ('select' in name))):
                        group['occupied_'+ name] = np.zeros(shape=(group.shape[0],))

            for i in range (1, group.shape[0]):
                count_cur_job = 0
                curr_run = 0.0
                max_run = 0.0
                for j in range (0, i):
                    # Other jobs are still running when this job is queued
                    if (group['qtime'].iloc[i] <= group['end'].iloc[j]):
                        count_cur_job += 1
                        curr_run += group['resources_used.walltime'].iloc[j]
                        max_run = max(max_run, group['resources_used.walltime'].iloc[j])

                        for name in group.columns:
                            if not ('avg' in name) and not ('normalized' in name) and not ('occupied' in name):
                                if ((not 'host' in name) and (not 'place' in name) and (not 'nodetype' in name) and (('Resource_List' in name) or ('select' in name))):

                                    group['occupied_'+ name].iloc[i] += group[name].iloc[j]
                curr_run_jobs[i] = count_cur_job
                curr_run_time[i] = curr_run
                #print('\n')
            group['cur_run_job'] = curr_run_jobs
            group['cur_run_time'] = curr_run_time
            group['cos_day_break_time'] = np.cos((2*math.pi / t_day)* (group['qtime'] % t_day))
            group['sin_day_break_time'] = np.sin((2 * math.pi / t_day) * (group['qtime'] % t_day))
            group['cos_week_break_time'] = np.cos((2 * math.pi / t_day) * (group['qtime'] % t_week))
            group['sin_week_break_time'] = np.sin((2 * math.pi / t_day) * (group['qtime'] % t_week))
            groups.append(group)

        df_final = pd.concat(groups, axis= 0)
        print (df_final.columns)
        print ("Output file ", training_loc + file.split('/')[-1])
        df_final.to_csv(training_loc + file.split('/')[-1], index=False)

if __name__ == "__main__":
    main()
