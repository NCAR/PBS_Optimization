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
	return (processors[:] * hours[:]).mean()

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
	print ("Average wait time per job", wait_time.mean())	
	num_processors = np.array(df['resources_used.ncpus'])
	hours_runtime = act_runtime / 3600.0
	utilization = produce_cpu_utilization(num_processors, hours_runtime)
	print ("CPU utilization is", utilization)
	
	total_wall_clock = np.array(df['end'] - df['ctime'])
	cpu_requested = np.array(df['Resource_List.ncpus'])
	expansion_factor = produce_expansion_factor(cpu_requested, total_wall_clock, num_processors, act_runtime)
	print ("Expansion factor", expansion_factor)
	
	
