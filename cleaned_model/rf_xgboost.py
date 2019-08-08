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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import datetime
import time
import sklearn_model as model

#--- Argument parsing ----- #
training_loc = argv[1]
testing_loc = argv[2]
saved_model = './saved_model/'
rf_report = argv[3]
xgb_report = argv[4]
old_data = argv[5]

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

def load_data(curFile, old):
        dfs = []
        y = []
        df = pd.DataFrame()
        for chunk in pd.read_csv(curFile, chunksize=200000, engine='python'):
             df = pd.concat([df, chunk])
        
        if (old == 'False'):
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
        else:
             mispred = np.array(df['requested_time'] - df['run_time']) / 3600.0
             y = create_label(mispred.reshape(-1,1), y)
             df = df.drop(['JobID'], axis=1)
        df = handle_non_numerical_data(df)

        y = np.array(y)
        X = np.array(df)
        scaling = StandardScaler()
        X = scaling.fit_transform(X)
        return X, y

def train_rf(X, y, saved_model,train_name):
	x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=0)
	clf = RandomForestClassifier(n_estimators=500, max_features=2, criterion='entropy', min_samples_leaf = 8, n_jobs=4)
	#rf = model.RandomForest(x_train,y_train,train_name, saved_model)
	#clf = rf.param_tuning()
	clf.fit(x_train,y_train)
	#joblib.dump(clf,'rf.pkl')
	y_pred = clf.predict(x_test)
	precision, recall,fscore,_ = prfs(y_test, y_pred,average='weighted')
	print ("Evaluation: Precision, recall, fscore", precision, recall, fscore)
	return clf

def test_rf(X, y, saved_model, clf):
	y_pred = clf.predict(X)
	precision, recall,fscore,_ = prfs(y_test, y_pred,average='weighted')
	print ("Test: Precision, recall, fscore", precision, recall, fscore)
	print (classification_report(y,y_pred, digits=4))

	return fscore

#def test_rf(X, y,saved_model,clf):
#        totalSampleNum = desiredNum * 6
#
#        returnArr = []
#        returnLabel = np.append(np.zeros(desiredNum), np.zeros(desiredNum))
#        returnLabel = np.append(returnLabel, np.full(desiredNum, 2))
#        returnLabel = np.append(returnLabel, np.full(desiredNum, 3))
#        returnLabel = np.append(returnLabel, np.full(desiredNum, 4))
#        returnLabel = np.append(returnLabel, np.full(desiredNum, 5))
#
#
#        if (class0Num == desiredNum):
#            class0IdxOri = class0Idx[0: desiredNum]
#        else:
#            class0IdxOri = class0Idx[np.random.randint(class0Num, size=desiredNum)]
#
#        if (class1Num == desiredNum):
#            class1IdxOri = class1Idx[0:desiredNum]
#        else:
#            class1IdxOri = class1Idx[np.random.randint(class1Num, size=desiredNum)]
#
#        if (class2Num == desiredNum):
#            class2IdxOri = class2Idx[0:desiredNum]
#        else:
#            class2IdxOri = class2Idx[np.random.randint(class2Num, size=desiredNum)]
#
#        if (class3Num == desiredNum):
#            class3IdxOri = class3Idx[0: desiredNum]
#        else:
#            class3IdxOri = class3Idx[np.random.randint(class3Num, size=desiredNum)]
#        if (class4Num == desiredNum):
#            class4IdxOri = class4Idx[0:desiredNum]
#        else:
#            class4IdxOri = class4Idx[np.random.randint(class4Num, size = desiredNum)]
#        if (class5Num == desiredNum):
#            class5IdxOri = class5Idx[0:desiredNum]
#        else:
#            class5IdxOri = class5Idx[np.random.randint(class5Num, size=desiredNum)]
#
#           
#        counter = 0
#        for i in class0IdxOri:
#            returnArr.append(feature[i,:])
#        
#        for i in class1IdxOri:
#            returnArr.append(feature[i,:])
#            
#        for i in class2IdxOri:
#            returnArr.append(feature[i,:])
#        
#        for i in class3IdxOri:
#            returnArr.append(feature[i,:])
#        for i in class4IdxOri:
#            returnArr.append(feature[i,:])
#        for i in class5IdxOri:
#            returnArr.append(feature[i,:])
#        print ("Feature, y check", len(returnArr), len(returnLabel)) 
#        
#        return returnArr, returnLabel

def train_xgb(x,y,saved_model,train_name):
	x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5, random_state=0)
	#clf = xgb(n_estimators=500, learning_rate=0.8, max_depth=3, max_delta_step=2, max_features="log2")
	#xgb = model.XGBoost(x_train, y_train, train_name, saved_model)
	#clf = xgb.param_tuning()
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
	train_name = train_file.split('/')[-1].split('.')[0]
	x_train, y_train = load_data(train_file, old_data)
	print ("-------- Training RF Model --------")
	start_time= time.time()
	clf = train_rf(x_train, y_train,saved_model, train_name)
	xgb_clf = train_xgb(x_train ,y_train, saved_model,train_name)
	rf_f1 = []
	xgb_f1 = []
	print ("Finish training")
	print ("Training file ", train_file)
	for test_file in testing_files:
		print ("Testing file", test_file)
		test_name = test_file.split('/')[-1].split('.')[0]
		x_test, y_test = load_data(test_file, old_data)
		rf_fscore = test_rf(x_test,y_test,saved_model, clf)
		rf_f1.append(rf_fscore)
		print ("---------- Finish testing RF --------")

		duration = time.time() - start_time
		print ("RF training-testing process time", duration)
		print ("--------Training XGB model----------")
		xgb_time = time.time()
		#xgb_clf = train_xgb(x_train ,y_train)
		xgb_fscore = test_xgb(x_test,y_test,xgb_clf)
		xgb_f1.append(xgb_fscore)
		end_time = time.time() - xgb_time
		print ("------------ Finish testing XGB ----------")
		print ("XGB training-testing process time", end_time)
		print ("\n")
	rf_df[train_file.split('/')[-1].split('.')[0]] = rf_f1
	xgb_df[train_file.split('/')[-1].split('.')[0]] = xgb_f1

	print ("------------------------------------------------------")
rf_df.to_csv(rf_report + 'rf_report_tuned.csv', index=False)
xgb_df.to_csv(xgb_report + 'xgb_report_tuned.csv', index=False)


