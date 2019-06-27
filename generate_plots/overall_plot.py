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
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import argparse

## Parsing arguments ###
def parse_argument():
        parser = argparse.ArgumentParser()
        parser.add_argument('--num_top', type=int, default=0)
        parser.add_argument('--groupby_val', type=str, default='user')
        parser.add_argument('--overall_feature', type=str, default='user_mispred')
        parser.add_argument("--overall_distr_plot", type=bool, default=False)
        parser.add_argument("--multi_dim_plot", type=bool, default=False)
        parser.add_argument("--multi_dim_y_feature", type=str, default='num_jobs')
        parser.add_argument("--multi_dim_x_feature", type=str, default = 'mispred_ratio')
        parser.add_argument('--data_path', type = str, default='../apr_2019_full/')
       
        args = parser.parse_args()
        config = args.__dict__
        return config

#  Create new directory
def MakeDirectory (dirname):
    try:
        os.mkdir(dirname)
    except Exception:
        pass

def gen_multi_dim_plot(df, plot_loc,config):
	x_val = []
	y_val = []
	user_names = []
	ratio = []
	print ("Started multi-dim plot")
	for name, group in df.groupby(config['groupby_val']):
		if (config['multi_dim_x_feature'] == 'num_jobs'):
			x = len(group)
		else:
			x = np.array(group[config['multi_dim_x_feature']]).mean()

		if (config['multi_dim_y_feature'] == 'num_jobs'):
			y= len(group)
		else:
			y= np.array(group[config['multi_dim_y_feature']]).mean()
		
		x_val.append(x)
		y_val.append(y)
		user_names.append(name)

	normalized_y = ( y_val - np.amin(y_val)) / (np.amax(y_val) - np.amin(y_val))
	normalized_x = ( x_val - np.amin(x_val)) / (np.amax(x_val) - np.amin(x_val))	
	a=np.random.random(len(y_val))
	ratio = normalized_y * normalized_x
	
	temp = np.array(ratio)
	x_val = np.array(x_val)
	y_val = np.array(y_val)
	user_names = np.array(user_names)

	if (config['num_top'] != 0):
		n=config['num_top']
		index_top = np.argsort(temp)[-n:]
		x_val = x_val[index_top]
		y_val = y_val[index_top]
		user_names = user_names[index_top]
		ratio = ratio[index_top] 
	
	colors = cm.rainbow(np.linspace(0, 1, len(x_val)))	
	fig, ax = plt.subplots()
	for i in range (len(x_val)):
		ax.scatter(x_val[i], y_val[i], c=colors[i], label=user_names[i] + '(' + str(round(ratio[i],2)) + ')')
	if (config['num_top'] != 0):
		ax.legend(ncol=math.ceil(config['num_top']/ 20),prop={'size': 4}, loc='best')
	title = config['multi_dim_y_feature'] + ' vs ' + config['multi_dim_x_feature'] + ' for ' + str(config['groupby_val'])
	if (config['num_top'] != 0):
		title += '(top ' + str(config['num_top']) + ' ' + str(config['groupby_val']) + ')'

	ax.set_title(title)
	ax.set_xlabel(config['multi_dim_x_feature'])
	ax.set_ylabel(config['multi_dim_y_feature'])
	plt.savefig(plot_loc + title +'.pdf', bbox_inches="tight", dpi=1000)
	plt.close()
	print ("Completed multi-dim plot")
			
	
def gen_overall_plot(plot_loc, df, file, config):
	print ("Started overall plot")
	w_15 = 0
	w_1 = 0
	w_3 = 0
	w_7 = 0
	w_more_7 = 0
	w_under = 0	
	for name, group in df.groupby(config['groupby_val']):
		within_15 = group[(group[config['overall_feature']] >= 0.0) & (group[config['overall_feature']] <= 0.25)].count()[0]
		within_1h = group[(group[config['overall_feature']] > 0.25) & (group[config['overall_feature']] <= 1.0)].count()[0]
		within_3h = group[(group[config['overall_feature']] > 1.0) & (group[config['overall_feature']] <= 3.0)].count()[0]
		within_7h = group[(group[config['overall_feature']] > 3.0) & (group[config['overall_feature']] <= 7.0)].count()[0]
		more_than_7h = group[(group[config['overall_feature']] > 7.0)].count()[0]
		under_pred = group[(group[config['overall_feature']] < 0.0)].count()[0]
		w_15+=within_15
		w_1 += within_1h
		w_3 += within_3h
		w_7 += within_7h
		w_more_7 += more_than_7h
		w_under += under_pred
	total = w_15 + w_1 + w_3 + w_7 + w_more_7 + w_under
	plot_values = list([w_15, w_1, w_3, w_7, w_more_7, w_under])
	plt.figure()
	patches, _, percent = plt.pie(plot_values, startangle=90, autopct='%1.1f%%', pctdistance=1.22)

	labels = list(['0-15min', '15min-1h', '1h-3h', '3h-7h', '>7h', 'underpred'])
	for i in range(len(labels)):
		labels[i] = labels[i] + '(' + percent[i].get_text() + ')'
	plt.legend(patches, labels, bbox_to_anchor=(1.2, 1.025), loc='upper left')
	plt.title( 'Overall composition of actual runtime for ' + plot_loc.split('/')[-2] + '  (Total ' + str(total) + ' considered jobs)')

	plt.savefig(plot_loc + 'Overall breakdown composition.png', bbox_inches="tight", dpi=300)
	plt.close()
	print ("Finished overall plot")


def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x==day)[0][0] + 1
    return(week_of_month)

def main():
    config = parse_argument()
    all_files = glob.glob(config['data_path'] + "*")
    all_files = sorted(all_files, key = lambda s: (s.split('/')[-1].split('_')[0], s.split('/')[-1].split('_')[1].split('.')[0]))
    print ("Sorted files", all_files)
    overall_plot_dir = '../plot_overall/'
    MakeDirectory(overall_plot_dir)
    
    for file in all_files:
        column_fields = ['Resource_List.walltime', 'resources_used.walltime','ctime', 'user', 'account']

        df = pd.DataFrame()
        for chunk in pd.read_csv(file, usecols = column_fields, chunksize=50000, engine='python'):
              df = pd.concat([df, chunk])
   
        df['user_mispred'] = (df['Resource_List.walltime'] - df['resources_used.walltime']) / 3600.0
        df['resources_used.walltime'] = df['resources_used.walltime'] / 3600.0
        df['Resource_List.walltime'] = df['Resource_List.walltime'] / 3600.0

        df['day_week'] = df['ctime'].apply(lambda x : datetime.fromtimestamp(x).weekday())
        df['week_month'] = df['ctime'].apply(lambda x: get_week_of_month(datetime.fromtimestamp(x).year, datetime.fromtimestamp(x).month, datetime.fromtimestamp(x).day))
        df['time_day'] = df['ctime'].apply(lambda x: (datetime.fromtimestamp(x).hour))
        
        df['mispred_ratio'] = ((df['Resource_List.walltime'] - df['resources_used.walltime']) / (df['Resource_List.walltime']))
        df['mispred_ratio'] = df['mispred_ratio'].replace(np.inf, 0.0)
        df['mispred_ratio'] = df['mispred_ratio'].replace(-np.inf, 0.0)
        
        df['mispred_ratio_runtime'] = np.where(df['resources_used.walltime']>= 1/60, (df['Resource_List.walltime'] - df['resources_used.walltime']) / df['resources_used.walltime'], 0.0) 
        # Generate plot directory    
        if (config['overall_distr_plot'] == True):
                distr_plot_loc = overall_plot_dir + config['overall_feature'] + '/'
                plot_loc = distr_plot_loc + file.split('/')[-1].split('.')[0] + '/'
                MakeDirectory(distr_plot_loc)
                MakeDirectory(plot_loc)
                gen_overall_plot (plot_loc, df, file, config)        
        
        if (config['multi_dim_plot'] == True):
                multi_dim_plot = overall_plot_dir + config['multi_dim_y_feature'] +'_vs_' + config['multi_dim_x_feature'] + '/'
                plot_loc = multi_dim_plot + file.split('/')[-1].split('.')[0] + '/'
                MakeDirectory(multi_dim_plot)
                MakeDirectory(plot_loc)
                gen_multi_dim_plot(df, plot_loc, config) 

if __name__ == "__main__":
    main()
