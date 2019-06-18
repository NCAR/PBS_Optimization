from sys import argv
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import datetime
import time
acc_log = '../accounting/2019_Apr_1/'
training_loc = '../training_data/'
testing_loc = '../testing_data/'
output_path = '../acc_after_train/'
saved_model = './saved_model/'
rf_report = '../rf_report/'
xgb_report = '../xgb_report/'

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

def create_label(input_val,labels):
	for i in range (input_val.shape[0]):
		if (input_val[i] < 0.0):
			labels.append(0)
		elif (input_val[i] >= 0.0 and input_val[i] < 0.25):
                        labels.append(1)
		elif (input_val[i] >=0.25 and input_val[i] <1.0):
			labels.append(2)
		elif (input_val[i] >=1.0 and input_val[i] <3.0):
			labels.append(3)
		elif (input_val[i] >=3.0 and input_val[i] <7.0):
                        labels.append(4)
		elif (input_val[i] >= 7.0):
                        labels.append(5)
	return labels

def load_data(curFile):
        dfs = []
        y = []
        df = pd.DataFrame()
        for chunk in pd.read_csv(curFile, chunksize=200000, engine='python'):
             df = pd.concat([df, chunk])

        # Create labels
        mispred = np.array(df['Resource_List.walltime'] - df['resources_used.walltime'])/ 3600.0
        y = create_label(mispred.reshape(-1,1),y)
        #print ("Misprediction", mispred)
        df = df[df.columns.drop(list(df.filter(regex='resources_used')))]
        df = df.drop(['start', 'ID', \
		'Exit_status','qtime','end','entity',\
		'rtime', 'exec_vnode', 'exec_host', 'etime', 'ctime', \
		'session', 'start' ], axis=1)
                #print ("Leftover columns", df.columns)
                #dfs.append(df)

        #final_df = pd.concat(dfs, axis=0)
        df = handle_non_numerical_data(df)
        #print ("Leftover columns", df.columns)

        y = np.array(y)
        X = np.array(df)
        scaling = StandardScaler()
        X = scaling.fit_transform(X)
        return X, y

def train_rf(X, y, saved_model):
	x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
	clf = RandomForestClassifier(n_estimators=500, max_features=2, criterion='entropy', min_samples_leaf = 8, n_jobs=4)
	clf.fit(x_train,y_train)
	#joblib.dump(clf,'rf.pkl')
	y_pred = clf.predict(x_test)
	precision, recall,fscore,_ = prfs(y_test, y_pred,average='weighted')
	print ("Evaluation: Precision, recall, fscore", precision, recall, fscore)
	return clf

def test_rf(X, y,saved_model,clf):
	#clf = joblib.load('rf.pkl')
	#clf = saved_model
	y_pred = clf.predict(X)
	precision, recall,fscore,_ = prfs(y, y_pred, average='weighted')
	print ("Testing Precision Recall Fscore", precision, recall, fscore)
	print (classification_report(y,y_pred, digits=4))
	return fscore

def read_into_txt(mapModified):
    content = []
    statusArray = ["Q", "S", "B", "E"]
    #MakeDirectory(output_path)
    #print ("Output path - Current path", output_path, os.getcwd())
    files = glob.glob(acc_log + '*')
    for f in files:
        with open(f, "r") as infile:
            parsedDate = f.split('/')
            output_File = open(output_path + parsedDate[-1], 'w')
            data = infile.readlines()

            for line in data:
                if ((line.split("/")[0].isdigit()) and (line.split(";")[1] == "E")):
                    element = line.split(' ')
                    sessionName = 0
                    qtime = 0
                    wallTimeLoc = 0
                    entity = element[1].split(';')[2].split('.')[0]

                    if not ('[' in entity):

                        for i in range(0,len(element)):
                            if ('session' in element[i]):
                                sessionName = int(element[i].split('=')[1])
                            elif ('qtime' in element[i]):
                                qtime = int(element[i].split('=')[1])
                            elif ('Resource_List.walltime' in element[i]):
                                wallTimeLoc =  i

                        if (((int(entity), qtime, sessionName) in mapModified.keys())):
                            element.append('Resource_List.soft_walltime=' + str(mapModified[(int(entity), qtime, sessionName)]))

                    writeLine = ' '.join(element)
                    output_File.write(writeLine)

                else:
                    output_File.write(line)

# No correction for misprediction in 0-15 mins
def correct_time(class_label, usr_pred_time):
	corrected_time = usr_pred_time
	if (class_label == 2):
		corrected_time = usr_pred_time - 15 * 60
	elif (class_label ==3):
		corrected_time = usr_pred_time - 60 * 60
	elif (class_label == 4):
		corrected_time = usr_pred_time - 3*60*60
	elif (class_label == 5):
		corrected_time = usr_pred_time - 7*60*60
	elif (class_label == 0): # underpred
		corrected_time = usr_pred_time + 15 * 60
	corrected_time = datetime.timedelta(0, corrected_time)
	#print ("Corrected time:", corrected_time, 'Label:',class_label, 'Original prediction:', usr_pred_time)
	return corrected_time

def write_prediction(y, testing_files):
	dfs = []
	for test_file in testing_files:
		df = pd.DataFrame()
		
		for chunk in pd.read_csv(test_file, chunksize=5000, engine='python'):
        		df = pd.concat([df, chunk])
		dfs.append(df)
	df = pd.concat(dfs, axis=0)
	entityArr = df['entity'].values
	qtimeArr = df['qtime'].values
	sessionArr = df['session'].values
	accountArr = df['account'].values
	actual_runtime = df['resources_used.walltime']
	predicted_runtime = df['Resource_List.walltime'].values
	mapModified={}

	for i in range (len(entityArr)):
		correctedTime = correct_time(y[i],predicted_runtime[i])
		mapModified[entityArr[i], qtimeArr[i], sessionArr[i]] = correctedTime
		#print ("Entity-qtime-account-ActRun-Corrected-predicted", entityArr[i], qtimeArr[i], accountArr[i], actual_runtime[i], correctedTime, predicted_runtime[i])
	read_into_txt(mapModified)

def train_xgb(X, y):
	x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
	clf = GradientBoostingClassifier(n_estimators=500, learning_rate=0.8, max_depth=3, max_features="log2")
	clf.fit(x_train,y_train)
	y_pred = clf.predict(x_test)
	precision, recall,fscore,_ = prfs(y_test, y_pred,average='weighted')
	print ("'Evaluation Precision, recall, fscore", precision, recall, fscore)
	return clf

def test_xgb(x,y,clf):
	y_pred = clf.predict(x)
	precision,recall,fscore,_ = prfs(y, y_pred, average='weighted')
	print ("Testing Precision Recall Fscore", precision, recall, fscore)
	print (classification_report(y,y_pred, digits=4))
	return fscore
	
training_files = glob.glob(training_loc + '*')
testing_files = glob.glob(testing_loc + '*')

MakeDirectory(saved_model)
MakeDirectory(rf_report)
MakeDirectory(xgb_report)
rf_df = pd.DataFrame()
xgb_df = pd.DataFrame()
rf_df['Test (down)/ Train (across)'] = np.array([l.split('/')[-1].split('.')[0] for l in testing_files])
xgb_df['Test (down)/ Train (across)'] = np.array([l.split('/')[-1].split('.')[0] for l in testing_files])

for train_file in training_files:
	print ('-----------------------------------------------------')
	print ("Training file ", train_file)

	x_train, y_train = load_data(train_file)
	print ("-------- Training RF Model --------")
	start_time= time.time()
	clf = train_rf(x_train, y_train,saved_model)
	xgb_clf = train_xgb(x_train ,y_train)
	rf_f1 = []
	xgb_f1 = []
	print ("Finish training")
	print ("Training file ", train_file)
	for test_file in testing_files:
		print ("Testing file", test_file)
		x_test, y_test = load_data(test_file)
		rf_fscore = test_rf(x_test,y_test,saved_model, clf)
		rf_f1.append(rf_fscore)
		print ("---------- Finish testing RF --------")

		duration = time.time() - start_time
		print ("RF training-testing process time", duration)
		print ("--------Training XGB model----------")
		xgb_time = time.time()
		xgb_clf = train_xgb(x_train ,y_train)
		xgb_fscore = test_xgb(x_test,y_test,xgb_clf)
		xgb_f1.append(xgb_fscore)
		end_time = time.time() - xgb_time
		print ("------------ Finish testing XGB ----------")
		print ("XGB training-testing process time", end_time)
		print ("\n")
	rf_df[train_file.split('/')[-1].split('.')[0]] = rf_f1
	xgb_df[train_file.split('/')[-1].split('.')[0]] = xgb_f1

	print ("------------------------------------------------------")
rf_df.to_csv(rf_report + 'rf_report_default.csv', index=False)

xgb_df.to_csv(xgb_report + 'xgb_report_default.csv', index=False)


