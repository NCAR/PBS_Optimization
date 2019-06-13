import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import calendar
import re
import os
import glob
import math
from sys import argv

# Argument inputs are as follows:
#   1. Feature to be plotted (i.e. queue, month_week, etc.)

full_loc = '../full/'

#  Create new directory
def MakeDirectory (dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        pass

def gen_histogram(num_jobs):
	num_jobs = np.array(num_jobs)
	print ("Mean", np.mean(num_jobs))
	print ("Max", np.max(num_jobs))
	print ("Min", np.min(num_jobs))
	print ("STD", np.std(num_jobs))
	
	num_jobs = num_jobs[np.argsort(num_jobs)[100:-100]]
	plt.figure()
	plt.hist(num_jobs, bins = 100)
	plt.xlabel('Users')
	plt.ylabel('Number of submitted jobs')
	plt.title("Histogram of submitted jobs for users")
	plt.savefig('test_' + file.split('/')[-1].split('.')[0] + '.pdf', dpi=300)
	plt.close()

# Generic pie plots
def gen_pie_plot(plot_values, labels, plot_loc, name,total):
	plt.figure()
	patches, _, percent = plt.pie(plot_values, startangle=90, autopct='%1.1f%%', pctdistance=1.22)
	
	for i in range(len(labels)):
		labels[i] = labels[i] + '(' + percent[i].get_text() + ')'
	plt.legend(patches, labels, bbox_to_anchor=(1.2, 1.025), loc='upper left')
	plt.title( 'User ' +  name + ' ' + plot_loc.split('/')[-2] + '  (Total ' + \
					  str(total) + ' considered jobs)')
	plt.savefig(plot_loc + name + '.png', bbox_inches="tight", dpi=300)
	plt.close()

## Contribution plots

def gen_contribution_pie_plot(pie_plot_vals, pie_plot_labels, plot_loc, criteria,extra_total_criteria, considered_jobs):
	print ("Start contribution pie plot")
	plt.figure()
	a=np.random.random(11)
	cs=cm.Set1(np.arange(11)/11.)
	patches, _, percent = plt.pie(pie_plot_vals, colors = cs, startangle=90, autopct='%1.1f%%', pctdistance=1.22)
    
	for i in range(len(pie_plot_labels)):
		pie_plot_labels[i] = pie_plot_labels[i] + '(' + percent[i].get_text() + ')'
	plt.legend(patches, pie_plot_labels, bbox_to_anchor=(1.2, 1.025), loc='upper left')
	if (criteria != 'length'):	
		plt.title('Contribution of top 10 users with highest ' + criteria + ' (Total ' +
                   str(round(extra_total_criteria,2)) + ' ' + criteria)
	else:
		plt.title('Contribution of top 10 users with highest job volume (Total ' +
                   str(considered_jobs) + ' jobs)')
	plt.savefig(plot_loc + criteria + ' Contribution.png', bbox_inches="tight", dpi=300)
	print ("Completed pie plot")

# Stacked bar chart
def gen_stacked_column_chart (plot_loc,criteria, plot_list, considered_jobs, total_jobs, top_extra_criteria, extra_total_criteria, usr_name):
	ind = np.arange(len(plot_list[0]))
	legends = list(['0-15mins', '15 mins-1h', '1h-3h', '3h-7h', '>7h', 'underpred'])
	plt.figure()
	width = 0.35
	for index in range (len(plot_list)):
		if (index == 0):	
                     #print ('Value to plot at level ', index, ':', plot_list[index]) 
			plt.bar(ind, plot_list[index], width)
			cumulative = np.array(plot_list[index])
		else:
			new_p = plt.bar(ind,plot_list[index], width, bottom=cumulative)
			cumulative = cumulative + np.array(plot_list[index])
	plt.ylabel('Number of jobs (in log scale)')
	plt.xlabel('User')
	if (criteria != 'length' and not 'ratio' in criteria):	
		plt.title('Total number of ' + str(considered_jobs) + ' jobs (' + str(round((considered_jobs / float(total_jobs)) * 100, 2)) + ' % total jobs) \n Occupy ' \
		+ str(round((top_extra_criteria / float(extra_total_criteria)) * 100, 2)) + \
		' % of total ' + str(round(extra_total_criteria,2)) + ' ' + str(criteria))
	else:
		plt.title('Total number of ' + str(considered_jobs) + ' jobs (' + str(round((considered_jobs / float(total_jobs)) * 100, 2)) + ' % total jobs)')
	plt.xticks(ind, labels=usr_name)
	plt.tick_params(labelsize=7)
	plt.yscale('log')
	print ("Legends", legends)
	print ("Plot List", plot_list)
	plt.legend(legends)
	plt.savefig(plot_loc + 'stacked_column_summary.pdf', dpi=300, bbox_inches='tight')
	plt.close()

def filter_data(plot_loc, df, file, criteria):
	groups =[]
	num_jobs = []
	num_filter = []
	num_users = 0
	total_jobs = 0
	extra_total_criteria = 0.0
	top_extra_criteria = 0.0
	for name, group in df.groupby('user'):
		group['num_jobs'] = len(group)
		num_users += 1
		total_jobs += len(group)
		if (len(group) > 100):
			groups.append(group)
		if (criteria == 'length'):
			num_filter.append(int(len(group)))
		else:
			appended_val = round(group[criteria].sum(axis=0),2)
			num_filter.append(appended_val)
			extra_total_criteria += appended_val		
		num_jobs.append(len(group))
	
	print ("Number of users", num_users)
	n=10
	num_filter = np.array(num_filter)
	top_10_users = num_filter[np.argsort(num_filter)[-n:]]
	w_15 = []
	w_1 = []
	w_3 = []
	w_7 = []
	w_more_7 = []
	w_under = []
	considered_jobs = 0
	usr_name = []
	pie_plot_vals = []
	pie_plot_labels = []
	rest_val = 0.0
	for name, group in df.groupby('user'):
		if (criteria == 'length'):
			compared_val = len(group)
		else:
			compared_val = round(group[criteria].sum(axis=0),2)
		if ( compared_val in top_10_users):
			print("Current user", name)
			usr_name.append(name)
			total = len(group)
			top_extra_criteria += float(compared_val)
			considered_jobs += total
			####
			pie_plot_vals.append(compared_val)
			pie_plot_labels.append(name)
			####
			# Individual pie plots
			within_15 = group[(group['user_mispred'] >= 0.0) & (group['user_mispred'] < 0.25)].count()[0]
			within_1h = group[(group['user_mispred'] > 0.25) & (group['user_mispred'] <= 1.0)].count()[0]
			within_3h = group[(group['user_mispred'] > 1.0) & (group['user_mispred'] <= 3.0)].count()[0]
			within_7h = group[(group['user_mispred'] > 3.0) & (group['user_mispred'] <= 7.0)].count()[0]
			more_than_7h = group[(group['user_mispred'] > 7.0)].count()[0]
			under_pred = group[(group['user_mispred'] < 0.0)].count()[0]
			
			w_15.append(within_15)
			w_1.append(within_1h)
			w_3.append(within_3h)
			w_7.append(within_7h)
			w_more_7.append(more_than_7h)
			w_under.append(under_pred)
			print('BY order', within_15, within_1h, within_3h, within_7h, more_than_7h, under_pred)
			plot_values = list([within_15, within_1h, within_3h, within_7h, more_than_7h, under_pred])
			labels = list(['0-15min', '15min-1h', '1h-3h', '3h-7h', '>7h', 'underpred'])
			gen_pie_plot(plot_values, labels, plot_loc, name,total)
			
		else:
			rest_val += compared_val
		
	### Plot contribution pie plot
	if( 'ratio' in criteria):
		pie_plot_vals.append((top_extra_criteria + rest_val) / float(num_users))
		pie_plot_labels.append("Mean")
		generate_bar_plot(pie_plot_vals, pie_plot_labels, criteria, plot_loc)
	else:
		pie_plot_vals.append(rest_val)
		pie_plot_labels.append("Rest")
		gen_contribution_pie_plot(pie_plot_vals, pie_plot_labels, plot_loc, criteria, extra_total_criteria, considered_jobs)
	###
	plot_list = list([w_15, w_1, w_3, w_7,w_more_7, w_under])
	gen_stacked_column_chart (plot_loc, criteria, plot_list, considered_jobs, total_jobs, top_extra_criteria, extra_total_criteria,usr_name)
	
	
	return pd.concat(groups, axis=0)

def generate_bar_plot(pie_plot_vals,cur_label, criteria, plot_loc):
	ind = np.arange(len(pie_plot_vals))
	plt.bar(ind, pie_plot_vals, width=0.35)
	plt.ylabel('Average misprediction ratio')
	plt.xlabel('User')
	plt.yscale('log')
	plt.xticks(ind, labels=cur_label)
	plt.tick_params(labelsize=7)
	plt.title("Top users by " + criteria)
	plt.savefig(plot_loc + "bar_" + criteria + ".pdf",dpi=300, bbox_inches='tight')

def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return(week_of_month)

def main():
    all_files = glob.glob(full_loc + "*")
    all_files = sorted(all_files, key = lambda s: (s.split('/')[-1].split('_')[0], s.split('/')[-1].split('_')[1].split('.')[0]))
    print ("Sorted files", all_files)
    feature = argv[1]
    plot_directory = '../plot_top_' + feature + '/'
    MakeDirectory(plot_directory)
    within_15 = []
    within_1 = []
    within_3 = []
    within_7 = []
    more_than_7 = []
    under_pred = []
    xlabels = []
    legends = list(['0-15min', '15min-1h', '1h-3h', '3h-7h', '>7h', 'underpred'])
    plot_list = [within_15, within_1, within_3, within_7, more_than_7, under_pred]
    print("Generate pie plot")

    for file in all_files:
        column_fields = ['Resource_List.walltime', 'resources_used.walltime', 'queue','ctime', 'user', 'account','resources_used.cput']
        df = pd.read_csv(file, usecols =  column_fields)
        df['user_mispred'] = (df['Resource_List.walltime'] - df['resources_used.walltime']) / 3600.0
        df['day_week'] = df['ctime'].apply(lambda x : datetime.fromtimestamp(x).weekday())
        df['week_month'] = df['ctime'].apply(lambda x: get_week_of_month(datetime.fromtimestamp(x).year, datetime.fromtimestamp(x).month, datetime.fromtimestamp(x).day))
        df['time_day'] = df['ctime'].apply(lambda x: (datetime.fromtimestamp(x).hour))
        df['mispred_ratio'] = (df['user_mispred'] / (df['Resource_List.walltime']))* 100.0
        df['mispred_ratio'] = df['mispred_ratio'].replace(np.inf, 0.0)
        df['mispred_ratio'] = df['mispred_ratio'].replace(-np.inf, 0.0)

        df['mispred_ratio_runtime'] = np.where(df['resources_used.walltime']>= 1/60, df['user_mispred'] / df['resources_used.walltime'], 0.0) 
        #print ("Mispred ratio", df['mispred_ratio_runtime'])
        # Generate plot directory
        plot_loc = plot_directory + file.split('/')[-1].split('.')[0] + '/'
        MakeDirectory(plot_loc)
        custom_labels = []
        filter_data(plot_loc, df, file, feature)
	
        
if __name__ == "__main__":
    main()
