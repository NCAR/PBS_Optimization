import pandas as pd
from sys import argv
import os
import numpy as np

# Argument order is as follows:
#	1. Folder of previous parsing stage (including all of the csv files)
#   2. Exported Folder (labelled)
#	3. Exported Folder (training)

# Convert all non-numerical-data into random values
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
	if (len(argv) < 3):
		print("Missing arguments! Please double-check")
	else:
		filePath = os.path.join(os.getcwd(), argv[1])
		listOfFiles = os.listdir(filePath)
		fieldList = []

        #Accumulate the most complete feature list from each month accounting log
		for file in listOfFiles:

			eachFilePath = os.path.join(os.getcwd(), argv[1], file)
			df = pd.read_csv(eachFilePath)
			for field in df.columns:
				if not (field in fieldList):
					fieldList.append(field)

		print (fieldList)
        # Modify each file
		for file in listOfFiles:
			eachFilePath = os.path.join(os.getcwd(), argv[1], file)
			df = pd.read_csv(eachFilePath)

			for j in range (len(fieldList)):
				if not (fieldList[j] in df.columns):
					df.insert(j, fieldList[j], np.zeros(len(df.values)))

			df.fillna(0, inplace=True)
			handle_non_numerical_data(df)

			userPredicted = np.array(df['Resource_List.walltime'].values)
			actualRun = np.array(df['resources_used.walltime'].values)
			difference = np.subtract(userPredicted, actualRun)
			difference = difference/ 3600.0

			label = []

			for i in np.nditer(difference):
				if ((i>=0) and (i <=2.0)):
					label.append(1)
				elif ((i>2.0) and (i <= 4.0)):
					label.append(2)
				elif ((i > 4.0) and (i <= 6.0)):
					label.append(3)
				elif ((i > 6.0) and (i <= 8.0)):
					label.append(4)
				elif ((i > 8.0) and (i <= 10.0)):
					label.append(5)
				elif (i > 10.0):
					label.append(5)
				else:
					label.append(0)

			myLabel = np.array(label)
			df.insert(0, 'Labels', myLabel)

			fullOutputPath = os.path.join(os.getcwd(), argv[2], file)
			df.to_csv(fullOutputPath, index = False)

			# Drop any columns representing the information given after jobs are done executing
			df = df[df.columns.drop(list(df.filter(regex='resources_used')))]
			df = df[df.columns.drop('Estimation Error')]

			trimmedOutput = os.path.join(os.getcwd(), argv[3], file)
			df.to_csv(trimmedOutput, index = False)

if __name__ == "__main__":
	main()