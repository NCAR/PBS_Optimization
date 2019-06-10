import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
from collections import OrderedDict
import os
import glob
from sys import argv

# Argument inputs are as follows:
#   1-n. Features to be plotted (i.e. queue, month_week, etc.)
csv_dir = './csv_output/'
training_loc = '../training/'
full_loc = '../full/'


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


def generate_contribution_pie_plots(plot_loc, df, field_name):
    counter = 0
    group_by_field = field_name
    acc_user= {}
    acc_usr_job_counts ={}
    day_labels ={'0':'Mo', '1':"Tu", '2': 'We', '3':'Tr', '4':'Fri', '5': 'Sat', '6': 'Sun'}
    time_breakdown= {}
    print ("Group by fields", group_by_field)
    for name,group in df.groupby(group_by_field):
        print ("Name", name)
        if (type(name[1]) == str):
            group_name = name[1].upper()
        else:
            group_name = name[1]

        if (field_name[0] == 'day_week'):
            user_name = day_labels[str(name[0])]
        else:
            user_name = name[0]
        if (field_name[1] == 'day_week'):
            group_name = day_labels[str(name[1])]


        within_15_mins = group[(group['user_mispred'] >= 0) & (group['user_mispred'] <= 0.25)].count()[0]

        within_1h = group[(group['user_mispred'] > 0.25) & (group['user_mispred'] <= 1.0)].count()[0]
        within_3h = group[(group['user_mispred'] > 1.0) & (group['user_mispred'] <= 3.0)].count()[0]
        within_7h = group[(group['user_mispred'] > 3.0) & (group['user_mispred'] <= 7.0)].count()[0]
        more_than_7h = group[(group['user_mispred'] > 7.0)].count()[0]
        under_predh = group[(group['user_mispred'] < 0.0)].count()[0]

        print('BY order', within_15_mins, within_1h, within_3h, within_7h, more_than_7h,
              under_predh)
        time_breakdown[user_name] = list([ within_15_mins, within_1h, within_3h, within_7h, more_than_7h,
              under_predh])
        counter += 1

        if not (group_name in acc_user):
            empty = []
            empty.append(user_name)
            acc_user[group_name] = empty
            acc_usr_job_counts[group_name] = len(group)
        else:

            acc_user[group_name].append(user_name)
            acc_usr_job_counts[group_name] += len(group)
    print ("Acc user", acc_user)

    # Job contribution from each account
    pie_dict = sorted(acc_usr_job_counts.items(), key=lambda k: -k[1])
    pie_dict = OrderedDict(pie_dict)
    total_jobs = float(np.sum(list(acc_usr_job_counts.values())))
    plot_vals = []
    plot_labels = []
    rest_val = 0.0
    for k in pie_dict.keys():
        if (float(pie_dict[k]) / total_jobs >= 0.02):
              plot_vals.append(pie_dict[k])
              plot_labels.append(str(k))
        else:
              rest_val += pie_dict[k]
    plot_vals.append(rest_val)
    plot_labels.append("Rest") 
    patches, _, percent = plt.pie(plot_vals, startangle=90, autopct='%1.1f%%', pctdistance=1.22)
    
    for i in range(len(plot_labels)):
        plot_labels[i] = plot_labels[i] + '(' + percent[i].get_text() + ')'
    plt.legend(patches, plot_labels, bbox_to_anchor=(1.2, 1.025), loc='upper left')
    plt.title('Contribution of each ' + field_name[1] + ' with more than 2% of total jobs (Total ' +
                   str(int(np.sum(list(acc_usr_job_counts.values())))) + ' considered jobs)')

    plt.savefig(plot_loc + field_name[1] + ' Contribution.png', bbox_inches="tight", dpi=300)
    print ("Completed pie plot")

    ### Plot contribution by jobs
    for account in acc_user.keys():
        usr_name =[]
        within_15 = []
        within_1 = []
        within_3 = []
        within_7 = []
        more_than_7 = []
        under_pred = []
        saved_dir = plot_loc + str(account)
        MakeDirectory(saved_dir)
        for usr in acc_user[account]:
            usr_name.append(usr)
            #usr_vals.append(acc_usr_job_counts[usr])
            within_15.append(time_breakdown[usr][0])
            within_1.append(time_breakdown[usr][1])
            within_3.append(time_breakdown[usr][2])
            within_7.append(time_breakdown[usr][3])
            more_than_7.append(time_breakdown[usr][4])
            under_pred.append(time_breakdown[usr][5])

        plot_list = [within_15,within_1, within_3, within_7, more_than_7, under_pred]
        legends = list(['0-15mins', '15 mins-1h', '1h-3h', '3h-7h', '>7h', 'underpred'])
        ind_val = 0
        for i in range (len(plot_list)):
            ind_val = max(ind_val, len(plot_list[i]))
        ind = np.arange(ind_val)

        plt.figure()
        width = 0.35
        print ("Group", account, usr_name)
        print("Generate stacked column plot")
        for index in range(len(plot_list)):
            try:
                plt.bar(ind, plot_list[index], width)
            except:
                print ("Failure")
                print ("Shape test", len(plot_list[index]), len(ind), index)
                print ("Further test", plot_list[index], ind)
        print ("\n")
        plt.ylabel('Number of jobs')
        plt.xlabel("Users ")
        plt.title('Total number of ' + str(len(usr_name)) + ' users (Total considered ' + str(acc_usr_job_counts[account]) + ' jobs)')
        plt.xticks(ind, labels=usr_name)
        plt.tick_params(labelsize=3)
        plt.yscale('log')
        plt.legend(legends)
        plt.savefig(saved_dir + '/stacked_column_summary.pdf', dpi=300, bbox_inches='tight')
        plt.close()
\
def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return(week_of_month)

def main():
    all_files = glob.glob(full_loc + "/*")
    print("Before", all_files)
    all_files = sorted(all_files, key = lambda s: (s.split('/')[-1].split('_')[0], s.split('/')[-1].split('_')[1].split('.')[0]))
    print ("Sorted files", all_files)

    features = []
    comps = argv[1].split('and')
    for comp in comps:
        features.append(comp)
    plot_directory = '../plot_column_' + argv[1] + '/'
    MakeDirectory(plot_directory)

    for file in all_files:
        column_fields = ['Resource_List.walltime', 'resources_used.walltime', 'queue','ctime', 'user', 'account']
        df = pd.read_csv(file, usecols =  column_fields)
        df['user_mispred'] = (df['Resource_List.walltime'] - df['resources_used.walltime']) / 3600.0
        total_count = float(len(df['resources_used.walltime']))
        df['day_week'] = df['ctime'].apply(lambda x : datetime.fromtimestamp(x).weekday())
        df['week_month'] = df['ctime'].apply(lambda x: get_week_of_month(datetime.fromtimestamp(x).year, datetime.fromtimestamp(x).month, datetime.fromtimestamp(x).day))
        df['time_day'] = df['ctime'].apply(lambda x: (datetime.fromtimestamp(x).hour))

        # Generate plot directory
        plot_loc = plot_directory + file.split('/')[-1].split('.')[0] + '/'
        MakeDirectory(plot_loc)
        #Generate conribution plots and individual plots
        generate_contribution_pie_plots(plot_loc, df, features)

        print("\n")


if __name__ == "__main__":
    main()
