import torch
import torch.nn as nn
import glob
import torch.nn.functional as F
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
import transfer_model
import argparse

# Argument:
##  Model name as either 'cnn'/'rnn'/'ff'

#training_loc = '../training_small/'
#testing_loc = '../testing_small/'
#report = '../' + argv[1] + '_report/'
#ckpt_dir = './best_' + argv[1] + '_model/'
a = Random()
a.seed(1)

#-------Hyperparameter of network------------#
#num_epochs = 100
#output_size = 6
#batch_size = 32
#learning_rate = 0.001
#hidden_size = 128
#dropout = 0


## Parsing arguments ###
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
        parser.add_argument('--report_path', type=str, default='../report/')
        parser.add_argument('--model_type', type=str, default='ff')
        parser.add_argument('--dropout', type=float, default=1.0)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--num_layers',type=int, default=2)
        parser.add_argument('--num_classes', type=int, default=6)
        parser.add_argument('--l2_decay', type=float, default=5e-4)
        parser.add_argument('--momentum', type=float, default = 0.9)
        parser.add_argument('--log_interval', type=int, default=10)
        
        args = parser.parse_args()
        config = args.__dict__
        return config

criterion = nn.CrossEntropyLoss()


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

def train_dnn(x_t, y_t, res_t, x_eval, y_eval, res_eval, x_test,y_test, config):
    # Start training process
    counter = 0
    best_f1 = 0.0
    f1_total = 0.0
    epoch_no_improve = 0
    n_epoch_stop = 3
    #momentum = 0.9
    #log_interval = 10
    #l2_decay = 5e-4

    batch_num = int(x_t.shape[0] / config['batch_size'])
    if (config['model_type'] == 'dan'):
        training_model = transfer_model.DANNet(config).to(config['device'])

    else:
        training_model= transfer_model.CNNModel(config).to(config['device'])

    loss_class = torch.nn.NLLLoss().to(config['device'])
    loss_domain = torch.nn.NLLLoss().to(config['device'])

    for epoch in range(config['num_epochs']):
        # Start training process
        if (config['model_type'] == 'dan'):
            LEARNING_RATE = config['learning_rate'] / math.pow((1 + 10 * (epoch - 1) / config['num_epochs']), 0.75)
            print('learning rate{: .4f}'.format(LEARNING_RATE) )
            optimizer = torch.optim.SGD([
        {'params': training_model.sharedNet.parameters()},
        {'params': training_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=config['momentum'], weight_decay=config['l2_decay'])
        elif (config['model_type'] == 'dann'):
            optimizer = torch.optim.Adam(training_model.parameters(), lr= config['learning_rate'])
 
        for batch in range(batch_num):
            training_model.train()
            batch_index = generate_batch(x_t.shape[0], config['batch_size'])
            test_batch_idx = generate_batch(x_test.shape[0], config['batch_size'])
            batch_x = x_t[batch_index]
            batch_y = y_t[batch_index]
            batch_res = res_t[batch_index]
            batch_x = batch_x.reshape(-1, 1, 1, batch_x.shape[1])
            batch_x = torch.from_numpy(batch_x).float().to(config['device'])
            batch_y = torch.from_numpy(batch_y).long().to(config['device'])
            batch_res = torch.from_numpy(np.array(batch_res)).float()
            batch_test_x = x_test[test_batch_idx].reshape(-1,1,1,x_test.shape[1])
            batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
            # Forward pass
            #print ("Shape source and target", batch_x.shape, batch_test_x.shape)
            alpha = 0.0
            if (config['model_type'] == 'dan'):
                  label_pred, loss_mmd = training_model(batch_x, batch_test_x)
                  batch_test_x = x_test[test_batch_idx].reshape(-1,1,1,x_test.shape[1])
                  batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
                  label_pred, loss_mmd = training_model(batch_x, batch_test_x)
            #loss = criterion(outputs, batch_y)
            # loss = _loss(batch_res, batch_y, outputs)
                  loss_cls = F.nll_loss(F.log_softmax(label_pred, dim=1), batch_y)
                  gamma = 2 / (1 + math.exp(-10 * (epoch) /config['num_epochs'])) - 1
                  err = loss_cls + gamma * loss_mmd
                  # Backward and optimize
                  optimizer.zero_grad()

                  err.backward()
                  optimizer.step()
                  #print ("Label prediction", label_pred)
                  tr_pred = np.argmax(label_pred.cpu().detach(), 1)
                  #tr_pred = np.argmax(outputs.cpu().detach(), 1)
                  counter += 1
                  training_model.eval()
                  _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
                  f1_total += f1

            elif (config['model_type'] == 'dann'):
                 
                  p = float(counter + epoch * x_t.shape[0]) / config['num_epochs'] / x_t.shape[0]
                  alpha = 2. / (1. + np.exp(-10 * p)) - 1
                  training_model.zero_grad() 
                  #batch_x = batch_x.reshape(-1, 1, 1, x_test.shape[1])
                  
                  #class_label = torch.LongTensor(batch_size)
                  domain_label = torch.zeros(config['batch_size'])
                  domain_label = domain_label.long().to(config['device'])
                  
                  # Using source data
                  class_output, domain_output = training_model(batch_x, alpha) 
                  err_s_label = loss_class(class_output, batch_y)
                  err_s_domain = loss_domain(domain_output, domain_label)
                  
                  ### Using target data
                  #x_test_batch = torch.FloatTensor(batch_size, 1, x_test.shape[1], x_test.shape[1])
                  
                  domain_label = torch.ones(config['batch_size'])
                  domain_label = domain_label.long().to(config['device'])
                  
                  _, domain_output = training_model(batch_test_x, alpha)
                  err_t_domain = loss_domain(domain_output, domain_label)
                  err = err_t_domain + err_s_domain + err_s_label
                  #print ("Error is ", err)
                  err.backward()
                  optimizer.step()
                  
                  training_model.eval()
                  class_output,_ = training_model(batch_x, alpha)
                  tr_pred = class_output.data.cpu().max(1, keepdim=True)[1]
                  _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
                  f1_total += f1
                  if (counter + 1) % 100 == 0:
                      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Best F1: {:.4f}'.format(epoch + 1, config['num_epochs'], batch + 1, batch_num, err.item(), f1))
                  counter += 1
        print("------epoch : ", epoch, " Loss: ", err.item(), " Training F1:", round((f1_total / batch_num), 4))
        training_model.eval()
        eval_f1 = evaluate_test(x_eval, y_eval, res_eval, training_model,config, False)
        f1_total += eval_f1
        if (eval_f1 >= best_f1):
            best_f1 = eval_f1
            torch.save(training_model.state_dict(), config['ckpt_path'] + 'best_model.pth')

        print("Current F1 score (evaluation) ", eval_f1)
        print("Best F1 score (evaluation) ", best_f1)
    print ("All completed")
    


def evaluate_test(x_eval, y_eval, res_eval, model, config, test=False):
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
            if ( config['model_type'] == 'dan' or config['model_type'] == 'dann'):
                batch_test_x = batch_test_x.reshape(-1, 1, 1, batch_test_x.shape[1])

            batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
            batch_test_y = torch.from_numpy(batch_test_y).long().to(config['device'])
            batch_test_res = torch.from_numpy(batch_test_res).float()
            output_test,_ = model(batch_test_x,batch_test_x)
            _, predicted = torch.max(output_test.data, 1)
            # loss = _loss(batch_test_res, batch_test_y, output_test)
            total_pred = np.concatenate((total_pred, predicted.cpu()))
            total_y_test = np.concatenate((total_y_test, batch_test_y.cpu()))

    # overall_loss = _loss(res_eval, y_eval, total_pred)
    # overall_loss = criterion(total_pred, total_y_test)
    # acc = accuracy_score(total_y_test, total_pred)
    if (test == True):
        print(classification_report(total_y_test, total_pred, digits=4))
        acc = accuracy_score(total_y_test, total_pred)
        print("Testing accuracy", acc)
    _, _, f1, _ = prfs(total_y_test, total_pred, average='weighted')

    # print ("Overall testing/evaluation F1 is ", f1)
    return f1

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    config = parse_argument()
    training_files = glob.glob(config['train_path'] + '*')
    testing_files = glob.glob(config['test_path'] + '*')
    #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #print("Available device", device)

    MakeDirectory(config['report_path'])
    df = pd.DataFrame()
    df['Test (down)/ Train (across)'] = np.array([l.split('/')[-1].split('.')[0] for l in testing_files])

    for train_file in training_files:
        print("Training file", train_file)
        x_train, y_train, train_resources = load_data(train_file)
        indices = np.arange(x_train.shape[0])
        x_t, x_eval, y_t, y_eval, idx_t, idx_te = train_test_split(x_train, y_train, indices, test_size=0.2,
                                                                   random_state=0)
        res_t = train_resources[idx_t]
        res_eval = train_resources[idx_te]
        input_size = x_t.shape[1]
        MakeDirectory(config['ckpt_path'])

        print("--------------- Finish loading data ------------------")
        print('Initializing Variables')
         
        for test_file in testing_files:
            if (config['model_type'] == 'dann' or config['model_type'] == 'dan'):
                  x_test, y_test, test_resources = load_data(test_file)
                  print("--------- Starting testing ---------")
                  print("Testing file", test_file)

                  train_dnn(x_t, y_t, res_t, x_eval, y_eval,res_eval,x_test, y_test, config)
                  # Testing
                  training_model = transfer_model.DANNet(config).to(device)
                  test_f1 = evaluate_test(x_test, y_test, test_resources, training_model, config, True)
                  print("Testing F1 score", test_f1)
                  print("-------------------------------")

            df[train_file.split('/')[-1].split('.')[0]] = test_results 
        
        print("Testing is completed")
        shutil.rmtree(ckpt_dir)
        print("Current best model directory is removed")
        output_report = config['report_path'] + config['model_type'] + '_report'
        for name in config.keys():
          if (name != 'model_type' and not 'path'in name):
                output_report += '_' + str(name) + str(config[name])
    
        df.to_csv(output_report + '.csv', index=False)


if __name__ == "__main__":
      main()
