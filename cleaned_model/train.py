import torch
import torch.nn as nn
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from random import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as prfs
import os
import shutil
from sys import argv
import model
import argparse
# Argument:
##  Model name as either 'cnn'/'rnn'/'ff'

#training_loc = '../training_data/'
#testing_loc = '../testing_data/'
#report = '../' + argv[1] + '_report/'
#ckpt_dir = './best_' + argv[1] + '_model/'
a = Random()
a.seed(1)

#-------Hyperparameter of network------------#
#num_epochs = 200
output_size = 6
#batch_size = 10
#learning_rate = 0.001
#hidden_size = 256

criterion = nn.CrossEntropyLoss()

def parse_argument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--hidden_size', type=int, default=128)
	parser.add_argument("--ckpt", type=bool, default=False)
	parser.add_argument("--ckpt_path", type = str, default ='ckpt')
	parser.add_argument('--train_path', type = str, default='train')
	parser.add_argument('--test_path', type = str, default = 'test')
	parser.add_argument('--learning_rate', type= float, default = 1e-3)
	parser.add_argument('--report_path', type=str, default='report')
	parser.add_argument('--model_type', type=str, default='ff')
	parser.add_argument('--dropout', type=float, default=1.0)
	parser.add_argument('--device', type=str, default='cpu')
	parser.add_argument('--num_layers',type=int, default=2)
	parser.add_argument('-num_classes', type=int, default=6)	
	args = parser.parse_args()
	config = args.__dict__
	return config
	
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

def calculate_resources(x, df):
        resources = np.zeros(shape=(x.shape[0], 1), dtype=np.float64)
        for column in df.columns:
                if ('Resource_List' in column):
                        resources +=  np.asarray(df[column], dtype=np.float64).reshape(-1,1)
        return resources


def load_data(curFile):
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

        final_df = handle_non_numerical_data(df)

        y = np.array(y)
        X = np.array(final_df)

        scaling = StandardScaler()
        X = scaling.fit_transform(X)
        resources = calculate_resources(X, final_df)
        return X, y, resources

def generate_batch(n, batch_size):
        batch_index = a.sample(range(n), batch_size)
        return batch_index

def train_model(x_t, y_t, res_t, x_eval, y_eval, res_eval, model, optimizer, config):
    # Start training process
    counter = 0
    best_f1 = 0.0
    f1_total = 0.0
    epoch_no_improve = 0
    n_epoch_stop = 3
    batch_num = int(x_t.shape[0] / config['batch_size'])

    for epoch in range(config['num_epochs']):
        # Start training process
        for batch in range(batch_num):
            model.train()
            batch_index = generate_batch(x_t.shape[0], config['batch_size'])
            batch_x = x_t[batch_index]
            batch_y = y_t[batch_index]
            batch_res = res_t[batch_index]
            if (config['model_type'] == 'cnn' or config['model_type'] == 'resnet'):
                batch_x = batch_x.reshape(-1, 1, 1, batch_x.shape[1])
            elif (config['model_type'] == 'rnn'):
                batch_x = batch_x.reshape(-1, 1, batch_x.shape[1])
           
            batch_x = torch.from_numpy(batch_x).float().to(config['device'])
            batch_y = torch.from_numpy(batch_y).long().to(config['device'])
            batch_res = torch.from_numpy(np.array(batch_res)).float()
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_pred = np.argmax(outputs.cpu().detach(), 1)
            counter += 1
            model.eval()
            _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
            f1_total += f1
            if (counter) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Best F1: {:.4f}'.format(epoch + 1, config['num_epochs'],
                                                                                          batch + 1, batch_num,loss, f1))

            counter += 1
        print("------epoch : ", epoch, " Loss: ", loss.item(), " Training F1:", round((f1_total / batch_num), 4))
        model.eval()
        eval_f1 = evaluate_test(x_eval, y_eval, res_eval, model,config, False)
        f1_total += eval_f1
        if (eval_f1 >= best_f1):
            best_f1 = eval_f1
            torch.save(model.state_dict(), config['ckpt_path'] + 'best_model.pth')

        print("Current F1 score (evaluation) ", eval_f1)
        print("Best F1 score (evaluation) ", best_f1)
    print("Training is completed")


def evaluate_test(x_eval, y_eval, res_eval, model, config,test=False):
    total_pred = np.array([], dtype=np.float64)
    total_y_test = np.array([], dtype=np.int64)

    test_batch_num = int(math.ceil(x_eval.shape[0] / float(config['batch_size'])))
    with torch.no_grad():
        for i in range(test_batch_num):
            begin_index = i * config['batch_size']
            end_index = min((i + 1) * config['batch_size'], x_eval.shape[0])
            batch_test_x = x_eval[begin_index: end_index]
            batch_test_y = y_eval[begin_index: end_index]
            batch_test_res = res_eval[begin_index: end_index]

            if (config['model_type'] == 'cnn' or config['model_type'] == 'resnet'):
                batch_test_x = batch_test_x.reshape(-1, 1, 1, batch_test_x.shape[1])
            elif (config['model_type'] == 'rnn'):
                batch_test_x = batch_test_x.reshape(-1, 1, batch_test_x.shape[1])
            
            batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
            batch_test_y = torch.from_numpy(batch_test_y).long().to(config['device'])
            batch_test_res = torch.from_numpy(batch_test_res).float()
            output_test = model(batch_test_x)
            _, predicted = torch.max(output_test.data, 1)
            total_pred = np.concatenate((total_pred, predicted.cpu()))
            total_y_test = np.concatenate((total_y_test, batch_test_y.cpu()))

    if (test == True):
        print(classification_report(total_y_test, total_pred, digits=4))
        acc = accuracy_score(total_y_test, total_pred)
        print("Testing accuracy", acc)
    _, _, f1, _ = prfs(total_y_test, total_pred, average='weighted')

    return f1

def main():
    config = parse_argument()
    print ("Config", config)
    training_files = glob.glob(config['train_path'] + '*')
    testing_files = glob.glob(config['test_path'] + '*')
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #print("Available device", device)

    MakeDirectory(config['report_path'])
    df = pd.DataFrame()
    df['Test (down)/ Train (across)'] = np.array([l.split('/')[-1].split('.')[0] for l in testing_files])

    for train_file in training_files:
        print("Training file", train_file)
        MakeDirectory(config['ckpt_path'])

        x_train, y_train, train_resources = load_data(train_file)
        indices = np.arange(x_train.shape[0])
        x_t, x_eval, y_t, y_eval, idx_t, idx_te = train_test_split(x_train, y_train, indices, test_size=0.2,
                                                                   random_state=0)
        res_t = train_resources[idx_t]
        res_eval = train_resources[idx_te]
        input_size = x_t.shape[1]
        print ("---------------------Finish loading data-----------------------")
        print ("--- Initializing variables --- ")
        if (config['model_type'] == 'cnn'):
            training_model = model.ConvNet(config).to(config['device'])
            optimizer = torch.optim.Adam(training_model.parameters(), lr = config['learning_rate'])
            train_model(x_t, y_t, res_t, x_eval, y_eval, res_eval, training_model, optimizer, config)

        elif (config['model_type'] == 'rnn'):
            training_model = model.RNN(input_size, config).to(config['device'])
            optimizer = torch.optim.Adam(training_model.parameters(), lr = config['learning_rate'])
            train_model(x_t, y_t, res_t, x_eval, y_eval, res_eval, training_model, optimizer, config)
            
        elif (config['model_type'] == 'ff'):
            training_model = model.NeuralNet(input_size,config).to(config['device'])
            optimizer = torch.optim.Adam(training_model.parameters(), lr = config['learning_rate'])
            train_model(x_t, y_t, res_t, x_eval, y_eval, res_eval, training_model, optimizer, config)
 
        elif( config['model_type'] == 'resnet'):
            training_model = model.ResNet(model.ResidualBlock, [2,2,2], config).to(config['device'])
            optimizer = torch.optim.Adam(training_model.parameters(), lr = config['learning_rate'])
            train_model(x_t, y_t, res_t, x_eval, y_eval, res_eval, training_model, optimizer, config)
 
         
    #i print ("Overall testing/evaluation F1 is ", f1)
        test_results = []
        print ("-------------- Starting testing -------------------------")
        for test_file in testing_files:
            x_test, y_test, test_resources = load_data(test_file)
            #print("--------- Starting testing ---------")
            print("Testing file", test_file)

            training_model.load_state_dict(torch.load(config['ckpt_path'] + 'best_model.pth'))
            test_f1 = evaluate_test(x_test, y_test, test_resources, training_model, config, True)
            test_results.append(test_f1)
            print("Testing F1 score", test_f1)
            print("-------------------------------")

        print("Testing is completed")
        shutil.rmtree(config['ckpt_path'])
        print("Current best model directory is removed")
             
        df[train_file.split('/')[-1].split('.')[0]] = test_results    

    output_report = config['report_path'] + config['model_type'] + '_report'
    for name in config.keys():
          if (name != 'model_type' and not 'path'in name):
                output_report += '_' + str(name) + str(config[name])
    
    df.to_csv(output_report + '.csv', index=False)

if __name__ == "__main__":
      main()
