import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
import calendar
import re
import os
import glob
import math
from sys import argv

full_loc = argv[1]

#  Create new directory
def MakeDirectory (dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        pass

def generate_pie_plots(plot_loc, df, field_name, custom_labels):
    counter = 0
    if (field_name == 'time_day'):
        bins = pd.cut(df[field_name], [0, 5, 11, 16,20,23])
        group_by_field = bins
    else:
        group_by_field = field_name
    if (custom_labels == []):
        for name,_ in df.groupby(group_by_field):
            custom_labels.append(name)

    for name,group in df.groupby(group_by_field):
        total_users = int(len(group))
        #print ("TOtal users", len(group))
        print ("Current", field_name, custom_labels[counter])
        within_15 = group[(group['user_mispred'] >= 0.0) & (group['user_mispred'] <0.25)].count()[0]
        within_1h = group[(group['user_mispred'] >= 0.25) & (group['user_mispred'] < 1.0)].count()[0]
        within_3h = group[(group['user_mispred'] >= 1.0) & (group['user_mispred'] < 3.0)].count()[0]
        within_7h = group[(group['user_mispred'] >= 3.0) & (group['user_mispred'] < 7.0)].count()[0]
        more_than_7h = group[(group['user_mispred'] >= 7.0)].count()[0]
        under_pred = group[(group['user_mispred'] < 0.0)].count()[0]

        print('BY order', within_15, within_1h, within_3h, within_7h, more_than_7h, under_pred)
        plot_values = list([within_15, within_1h, within_3h, within_7h, more_than_7h, under_pred])
        plt.figure()
        patches, _, percent = plt.pie(plot_values, startangle=90, autopct='%1.1f%%', pctdistance=1.22)

        labels = list(['0-15min', '15min-1h', '1h-3h', '3h-7h', '>=7h', 'underpred'])
        for i in range(len(labels)):
            labels[i] = labels[i] + '(' + percent[i].get_text() + ')'
        plt.legend(patches, labels, bbox_to_anchor=(1.2, 1.025), loc='upper left')
        plt.title(field_name + ' ' + custom_labels[counter] + ' '+ plot_loc.split('/')[-2] + '  (Total ' +
                   str(total_users) + ' considered jobs)')

        plt.savefig(plot_loc + custom_labels[counter] + '.png', bbox_inches="tight", dpi=300)
        counter += 1
        plt.close()

def generate_stacked_column_chart (df, field_name, labels, custom_labels, file):
    within_15 = []
    within_1 = []
    within_3 = []
    within_7 = []
    more_than_7 = []
    under_pred = []
    counter = 0
    if (field_name == 'time_day'):
        bins = pd.cut(df[field_name], [0, 5, 11, 16,20,23])
        group_by_field = bins
    else:
        group_by_field = field_name
    for name, group in df.groupby(group_by_field):

        print("Current", field_name, custom_labels[counter])
        within_15min = group[(group['user_mispred'] >= 0.0) & (group['user_mispred'] < 0.25)].count()[0]
        within_1h = group[(group['user_mispred'] >= 0.25) & (group['user_mispred'] < 1.0)].count()[0]
        within_3h = group[(group['user_mispred'] >= 1.0) & (group['user_mispred'] < 3.0)].count()[0]
        within_7h = group[(group['user_mispred'] >= 3.0) & (group['user_mispred'] < 7.0)].count()[0]
        more_than_7h = group[(group['user_mispred'] >= 7.0)].count()[0]
        under_predh = group[(group['user_mispred'] < 0.0)].count()[0]

        print('BY order (15 mins,1h, 3h, 7h, >=7h, underpred) \n', within_15min, within_1h, within_3h, within_7h, more_than_7h,
              under_predh)
        within_15.append(within_15min)
        within_1.append(within_1h)
        within_3.append(within_3h)
        within_7.append(within_7h)
        more_than_7.append(more_than_7h)
        under_pred.append(under_predh)

        file_name = file.split('/')[-1].split('.')[0]
        label = custom_labels[counter] + '\n' + file_name.split('_')[0][-2:] + '\n' + file_name.split('_')[1]
        labels.append(label)
        counter += 1
    return list([within_15, within_1, within_3, within_7, more_than_7,under_pred]), labels

def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return(week_of_month)

def main():
    all_files = glob.glob(full_loc + "*")
    print("Before", all_files)
    all_files = sorted(all_files, key = lambda s: (s.split('/')[-1].split('_')[0], s.split('/')[-1].split('_')[1].split('.')[0]))
    print ("Sorted files", all_files)
    feature = argv[2]
    plot_directory = '../plot_column_' + feature + '/'
    MakeDirectory(plot_directory)
    within_15 = []
    within_1 = []
    within_3 = []
    within_7 = []
    more_than_7 = []
    under_pred = []
    xlabels = []
    legends = list(['0-15min', '15min-1h', '1h-3h', '3h-7h', '>=7h', 'underpred'])
    plot_list = [within_15, within_1, within_3, within_7, more_than_7, under_pred]
    print("Generate pie plot")

    for file in all_files:
        column_fields = ['Resource_List.walltime', 'resources_used.walltime', 'queue','ctime', 'user', 'account']
        df = pd.read_csv(file, usecols =  column_fields)
        df['user_mispred'] = (df['Resource_List.walltime'] - df['resources_used.walltime']) / 3600.0
        df['day_week'] = df['ctime'].apply(lambda x : datetime.fromtimestamp(x).weekday())
        df['week_month'] = df['ctime'].apply(lambda x: get_week_of_month(datetime.fromtimestamp(x).year, datetime.fromtimestamp(x).month, datetime.fromtimestamp(x).day))
        df['time_day'] = df['ctime'].apply(lambda x: (datetime.fromtimestamp(x).hour))

        # Generate plot directory
        plot_loc = plot_directory + file.split('/')[-1].split('.')[0] + '/'
        MakeDirectory(plot_loc)
        custom_labels = []
        if (argv[1] == 'time_day'):
            custom_labels = ['1-6', '6-12', '12-17', '17-21', '21-24']
        elif(argv[1] == 'day_week'):
            custom_labels = ['Mon', "Tue", 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
        elif (argv[1] == 'week_month'):
            custom_labels = ['1', '2', '3', '4', '5']

        generate_pie_plots(plot_loc, df, feature, custom_labels)

        cur_list, xlabels = generate_stacked_column_chart(df, feature, xlabels,custom_labels, file)
        for i in range (len(cur_list)):
            for j in range (len(cur_list[i])):
                plot_list[i].append(cur_list[i][j])

        print("\n")

        ind = np.arange(len(xlabels))
        plt.figure()
        width = 0.35
        print("Generate stacked column plot")
        for i in range (len(plot_list)):
            if (i == 0):
               plt.bar(ind, plot_list[i], width)
               cumulative = np.array(plot_list[i])
            else:
               plt.bar(ind,plot_list[i], width, bottom=cumulative)
               cumulative = cumulative + np.array(plot_list[i])
        plt.ylabel('Number of jobs')
        plt.xlabel("Period of time")
        plt.xticks(ind, labels=xlabels)
        plt.tick_params(labelsize = 7)
        plt.legend(legends)
        plt.savefig(plot_loc + '/overall_result.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print ("Completed stacked column chart")
if __name__ == "__main__":
    main()
