import numpy as np
import pandas as pd
import glob

def produce_avgbsld(wait_time, act_runtime, limit):
	max_vals = np.fmax(act_runtime, limit)
	#print ("Max array", max_vals.shape, max_vals)
	ones = np.ones(wait_time.shape)
	slow_down = (wait_time + act_runtime)/ max_vals
	#print ("Slow down" , slow_down.shape, slow_down)
	#print ("Check max", np.fmax(slow_down, ones))
	avg_bsld = np.fmax(slow_down, ones).mean()
	return avg_bsld

def produce_cpu_utilization(processors, hours):
	utilization = 0.0
	for i in range (processors.shape[0]):
		utilization += (hours[i]* processors[i])
		
	return utilization

# Share queue: ncpus * walltime
# Not share queue, if (place = excelhost, then 1 node/ 1 host (1 node = 72 cores); otherwise, 1 core
def produce_user_cpu_utilization(nodect, ncpus_select, hours, place,queue):
	utilization = 0.0
	for i in range (nodect.shape[0]):
		if not ('share' in queue[i]):
			if not ('host' in place[i]):
				utilization += (hours[i] * ncpus_select[i])
			else:
				utilization += (hours[i] * nodect[i] * 72.0)
		else:
			utilization += (hours[i] * ncpus_select[i])
			print ("Share queue found")
	return utilization

def produce_expansion_factor(cpu_requested, total_wall_clock, num_processors, act_runtime):
	return ((cpu_requested[:] * total_wall_clock[:])[:]/(num_processors[:] * act_runtime[:])[:]).mean()

full_loc = '../full_output/'
all_files = glob.glob(full_loc+ '*')

for file in all_files:
	print (file)
	df = pd.read_csv(file, engine='python')
	wait_time = np.array(df['start'] - df['qtime'])
	act_runtime = np.asarray(df['resources_used.walltime'])
	limit = np.full(wait_time.shape, 10)
	avg_bsld = produce_avgbsld(wait_time, act_runtime, limit)
	print ("Average bounded slowdown", avg_bsld)
	print ("Average wait time per job", wait_time.mean(), '(seconds)')	
	num_processors = np.array(df['resources_used.ncpus'])
	hours_runtime = act_runtime / 3600.0
	
	queue = df['queue']
	place = df['Resource_List.place']
	#usr_node_ct = df['Resource_List.nodect']
	utilization = produce_cpu_utilization(num_processors, hours_runtime)
	user_predict_runtime = np.array(df['Resource_List.walltime'])/ 3600.0
	#user_predict_cores = np.array(df['Resource_List.ncpus'])
	user_predict_nodect = np.array(df['Resource_List.nodect'])
	user_ncpu_select = np.array(df['ncpus_select'])
	#prediction_utilization = produce_cpu_utilization(user_predict_cores, user_predict_runtime)
	prediction_utilization = produce_user_cpu_utilization(user_predict_nodect, user_ncpu_select, user_predict_runtime, place,queue) 
	#queue = df['queue']
	#place = df['Resource_List.place']
	print ('----- Using nodect ---------')
	print ("Number of executed nodes", np.array(df['Resource_List.nodect']).sum())
	print ("CPU utilization is", utilization, 'core hours')
	print ("User_predict CPU utilization is", prediction_utilization, 'core hours')
	print ("Percentage of actual usage over capacity:", (utilization/(4032*72*24*7))*100.0)
	print ("Percentage of user_predict usage over capacity:", (prediction_utilization/(4032*72*24*7))*100.0)
	print ("Percentage of actual usage over user_prediction", (utilization / prediction_utilization)* 100)
	total_wall_clock = np.array(df['end'] - df['ctime'])
	cpu_requested = np.array(df['Resource_List.ncpus'])
	expansion_factor = produce_expansion_factor(cpu_requested, total_wall_clock, num_processors, act_runtime)
	print ("Expansion factor", expansion_factor)
	print ("\n")
	
