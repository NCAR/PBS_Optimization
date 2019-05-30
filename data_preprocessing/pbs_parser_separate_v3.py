from sys import argv,exit
import csv
import os
import time
import calendar
from datetime import timedelta
import pandas as pd
import numpy as np
from itertools import cycle, islice
import glob
import collections

# Argument inputs are as follows:
# 	1. Directory of accounting logs (exact directory without any further subdirectories)
#		Within this directory, there should only be accounting logs


csv_output = '../csv_output'
output_full = '../full_output'

#  Create new directory
def MakeDirectory (dirname):
	try:
		os.mkdir(dirname)
	except Exception:
		pass

# Read all accounting logs into one single text file
def read_accounting(input_dir, out_file):
		content = []
		read_files = glob.glob(input_dir + "/*")

		with open(out_file, "wb") as outfile:
			for f in read_files:
				with open(f, "r") as infile:
					data = infile.readlines()
					for line in data:
						if ((line.split("/")[0].isdigit()) and (line.split(";")[1] == "E") ):
							content.append(line)
							outfile.write(line)
							outfile.write(infile.read())

# Parse messages of accounting logs into dictionary
def parse_acct_record(m):
	squote = 0
	dquote = 0
	paren = 0
	key = ""
	value = ""
	in_key = 1
	rval = {}

	for i in range(0, len(m)):
		#safety checks
		if in_key < 0:
			raise Exception("Unexpected Happened")
		if in_key < 1 and key == "":
			raise Exception("Null Key")

		#parens seem to be super-quotes
		if m[i] == '(':
			paren = paren + 1
		if m[i] == ')':
			paren = paren - 1
		#single quotes are the next strongest escape character
		if m[i] == '\'':
			if squote > 0:
				squote = squote - 1
			else:
				if dquote > 1:
					raise Exception("Don't think this can happen")
				squote = squote + 1
		#then double quotes
		if m[i] == '"':
			if dquote > 0 and squote == 0:
				dquote = dquote - 1
			else:
				dquote = dquote + 1
		#last, equal signs
		if m[i] == '=' and squote == 0 and dquote == 0 and paren == 0:
			if value is "":
				in_key = 0
				continue
			else:
				if not (m[i] == '=' and in_key == 0):
					#pretty sure you can't have an equal in a key
					#print m
					raise Exception("Unhandled Input", m[i])
		if m[i] == ' ' and (squote > 0 or dquote > 0 or paren > 0):
			if in_key == 1:
				key += m[i]
				continue
			else:
				value += m[i]
				continue
		if m[i] == ' ' and in_key==0:
			if not key in rval:
				rval[key] = value

# Parse "SELECT" field of account
			in_key = 1
			key = ""
			value = ""
			continue
		if m[i] == ' ':
			continue
			raise Exception("Unexpected whitespace")
		if in_key == 1:
			key += m[i]
		if in_key == 0:
			value += m[i]
	if in_key == 1 and len(key) > 1:
		print ("Warning: Gibberish: " + key)
	rval[key.rstrip('\n')] = value.rstrip('\n')

	return rval

# Parse "SELECT" field of accounting logs
def parseSelect(v):
    # More than 1 resource list
	select_features = {}
	if ('+' in v):
		parts = v.split('+')
	else:
		parts = [v]
	#print ("Select statement", parts)
	for part in parts:

		element = part.split(":")
		try:
			numCopy = int(element[0])
			start = 1
		except:
			numCopy = 1
			start = 0

		for e in range (start, len(element), 1):
			comp = element[e].split("=")
			if not (comp[0] in select_features):
				try :
					select_features[comp[0]+'_select'] = numCopy * int(comp[1])
				except:
					select_features[comp[0]+'_select']  = comp[1]

			else:
				try:
					select_features[comp[0]+'_select'] += numCopy * int(comp[1])
				except:
					select_features[comp[0] + '_select'] += comp[1]

		if (len(select_features) == 0):
			select_features['chunk_select'] = v

	return select_features

# Convert non-numeric data values into numeric ones
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

def parsing_accounting_file(accounting_file, csv_output, statFileWriter):
	# Conduct extracting features from each job record
	record_id = -1
	file_name = accounting_file.split('/')[-1]
	num_end_rec = 0
	with open(accounting_file, "r") as infile:
		data = infile.readlines()
		doc = []
		keynames = set()
		# Actually building the dictionary
		for entry in data:

                        
                        # Focus only on E records
                        # L: PBS license stats
                        # Q: queued
                        # S: started
                        # D:

                        # Only read E full records
			if ((entry.split("/")[0].isdigit()) and (entry.split(";")[1] == "E") and not('[' in entry.split(';')[2].split('.')[0])):

				fields = entry.split(";")
				message = ' '.join(fields[3:])
				rec = parse_acct_record(message)

				# Ignore jobs with Resource_list.inception field
				if ('Resource_List.inception' in rec.keys()):
					pass

				# Ignore jobs with specified Resource_List.mem or Resource_List.pmem
				# since not all users specify this information
				elif (('Resource_List.mem' in rec.keys()) or ('Resource_List.pmem' in rec.keys())):
					pass

				#Ignore any reserved jobs
				elif (any(l.startswith('resv') for l in rec.keys())):
					pass

				# Submitted jobs without specifying account name, IGNORE
				# since it is old data
				elif not ('account' in rec.keys()):
					pass
				# Ignore other features (i.e. rsv, reserved ID)
				elif ((len(rec.keys()) != 28) and ('Resource_List.mpiprocs' in rec.keys())
							and ('account' in rec.keys())):
					pass

				else:
					updated_dict = {}
					num_end_rec += 1
					record_id = record_id + 1

					# Time that records are written
					rtime = time.strptime(fields[0], "%m/%d/%Y %H:%M:%S")  # localtime() -- local time
					rtime = calendar.timegm(rtime)

					job_num = fields[2].split('.')  # "license" or job number
					entity = job_num[0]

					updated_dict['ID'] = record_id
					keynames.add('ID')
					updated_dict['rtime'] = rtime # record time
					keynames.add('rtime')
					updated_dict['entity'] = entity
					keynames.add('entity')

					if 'Resource_List.walltime' in rec.keys():
					# Assigning mpiprocs = 0 if it is not specified by users
						if not ('Resource_List.mpiprocs' in rec):
							rec['Resource_List.mpiprocs'] = 0

					for key in rec.keys():
						# Resource-related time (in format of HH:MM:SS)
						if ((key == 'resources_used.walltime') or (key == 'Resource_List.walltime')
									or (key == 'resources_used.cput')):
							v = rec[key].split(":")
							v = timedelta(hours=int(v[0]), minutes=int(v[1]),
								seconds=int(v[2])).total_seconds()
							updated_dict[key] = v
							keynames.add(key)

						# Remove kb to make the field numeric for mem and vmem
						elif ('mem' in key or  'vmem' in key):
							if ('kb' in rec[key]):
								v = rec[key].replace('kb', '')
								updated_dict[key] = v
								keynames.add(key)

						# Number of cores and nodes for the select statement
						elif (key == 'Resource_List.select'):

							select_features = parseSelect(rec[key])

							if (len(select_features) != 0):
								select_features = collections.OrderedDict(sorted(select_features.items(), reverse=True))

							for k, v in select_features.items():
								updated_dict [k] = v
								keynames.add(k)
						else:
							updated_dict[key] = rec[key]
							keynames.add(key)

					doc.append(updated_dict)
		outputFileLocation = os.path.join(csv_output, file_name +'.csv')
		outputFile = open(outputFileLocation, 'w')
		output_writer = csv.DictWriter(outputFile, fieldnames = list(keynames))
		output_writer.writeheader()
		output_writer.writerows(doc)
		statFileWriter.writerow([file_name, num_end_rec])
		print("Parsing Completed")
		outputFile.close()

		#Final edit (fill in 0 for empty cells)
		try:
			df = pd.read_csv(outputFileLocation)
			df.fillna(0, inplace=True)
			df.to_csv(outputFileLocation, index=False)
		except:
			print("No E record for " + file_name)
			pass
		print("Final Edit completed")

def main():
	accountingFolderLocation = argv[1]

	MakeDirectory(csv_output)
	print (argv[1].split('/')[-2])
	read_files = glob.glob(accountingFolderLocation + "/*")
	read_files.sort()
	statFileLoc = os.path.join(os.getcwd(),  'stats_distr.csv')
	statFile = open(statFileLoc, 'w')
	statFileWriter = csv.writer(statFile)

	## Parsing accounting file for each log
	for file in read_files:
		print ("Current File is ")
		print (file)
		parsing_accounting_file(file, csv_output,statFileWriter)
		print ("\n")

	# Combining separate CSVs (for each day) into one CSV for the month
	all_csv_files = glob.glob(csv_output + "/*")
	dfs = []
	for csv_file in all_csv_files:
		try:
			cur_df = pd.read_csv(csv_file)
			dfs.append(cur_df)
		except:
			#print ("The current file does not have any E records")
			pass

	df_combined = pd.concat(dfs, axis = 0)
	print (df_combined.columns)
	df_combined.fillna(0, inplace=True)
	df_combined.to_csv(output_full + argv[1].split('/')[-2] +'.csv', index=False)
	print ("Completed Successfully")
if __name__ == "__main__":
	main()
