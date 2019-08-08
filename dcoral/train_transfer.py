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
#import transfer_model
import argparse
import dcoral as transfer_model
from matplotlib.ticker import NullFormatter
from openTSNE import TSNE
import matplotlib.pyplot as plt
#from torch.scatter import *
# Argument:
##  Model name as either 'cnn'/'rnn'/'ff'

a = Random()
a.seed(1)

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
        parser.add_argument('--learning_rate', type= float, default = 1e-5)
        parser.add_argument('--report_path', type=str, default='../report/')
        parser.add_argument('--model_type', type=str, default='ff')
        parser.add_argument('--dropout', type=float, default=1.0)
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--num_layers',type=int, default=2)
        parser.add_argument('--num_classes', type=int, default=6)
        parser.add_argument('--l2_decay', type=float, default=5e-4)
        parser.add_argument('--momentum', type=float, default = 0.9)
        parser.add_argument('--log_interval', type=int, default=10)
        parser.add_argument('--old', type=bool, default=False)
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

def plot_tsne(source_data, source_name, target_data, target_name, plot_directory):
	fig, ax = plt.subplots()
	perplexities =[100]
	for i, perplexity in enumerate(perplexities):
		tsne = TSNE(n_components=2, initialization='pca', random_state=0, perplexity=perplexity, n_iter=1000, neighbors='approx')
		x_source_transformed = tsne.fit(source_data)
		x_target_transformed = tsne.fit(target_data)
		ax.set_title('Perplexity=%d' % perplexity)
		ax.scatter(x_source_transformed[:,0], x_source_transformed[:,1], c='r', label = 'source')
		ax.scatter(x_target_transformed[:,0], x_target_transformed[:,1], c= 'b', label = 'target')
		ax.xaxis.set_major_formatter(NullFormatter())
		ax.yaxis.set_major_formatter(NullFormatter())
		ax.axis('tight')
		ax.legend()
		plt.savefig(plot_directory + 'tsne_source' + source_name + '_target' + target_name + '.png' , dpi=500)


def load_data(curFile,config):
        y = []
        old_workloads = ['SDSC', 'CIEMAT', 'Curie']
        df = pd.DataFrame()
        for chunk in pd.read_csv(curFile, chunksize=200000, engine='python'):
                        df = pd.concat([df, chunk])
        
        if (config['old'] == True):
             mispred = np.array(df['requested_time'] - df['run_time']) / 3600.0
             y = create_label(mispred.reshape(-1,1),y)
             df = df.drop(['JobID'], axis=1)
             
        else:
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

def jan_loss(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)

def train_dnn(x_t, y_t, res_t, x_eval, y_eval, res_eval, x_test,y_test, config):
    # Start training process
    counter = 0
    best_f1 = 0.0
    f1_total = 0.0
    epoch_no_improve = 0
    n_epoch_stop = 3
    if (config['old'] == False):
          size = 2048
    else:
          size = 1024

    transformed_source = np.zeros(shape=(1, size), dtype=np.float64)
    batch_num = int(x_t.shape[0] / config['batch_size'])
    if (config['model_type'] == 'dan'):
        training_model = transfer_model.DANNet(config).to(config['device'])

    elif (config['model_type'] == 'dann'):
        training_model= transfer_model.CNNModel(config).to(config['device'])
    elif (config['model_type'] == 'jdda'):
        training_model = transfer_model.JDDA(config).to(config['device'])
    elif (config['model_type'] == 'jdda_rnn'):
        training_model = transfer_model.JDDA_RNN(x_t.shape[1], config).to(config['device'])
    elif (config['model_type'] == 'jdda_ff'):
        training_model = transfer_model.JDDA_FF(x_t.shape[1], config).to(config['device'])
    loss_class = torch.nn.NLLLoss().to(config['device'])
    loss_domain = torch.nn.NLLLoss().to(config['device'])
    domain_loss = torch.nn.CrossEntropyLoss().to(config['device'])
    for epoch in range(config['num_epochs']):
        # Start training process
        if (config['model_type'] == 'dan'):
            LEARNING_RATE = config['learning_rate'] / math.pow((1 + 10 * (epoch - 1) / config['num_epochs']), 0.75)
            print('learning rate{: .4f}'.format(LEARNING_RATE) )
            optimizer = torch.optim.SGD([
        {'params': training_model.sharedNet.parameters()},
        {'params': training_model.cls_fc.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=config['momentum'], weight_decay=config['l2_decay'])
        elif (config['model_type'] == 'dann' or  config['model_type'] == 'jdda' or config['model_type'] =='jdda_rnn'):

            LEARNING_RATE = config['learning_rate'] / math.pow((1 + 10 * (epoch - 1) / config['num_epochs']), 0.75)
            #if (epoch > 49):
            #      for param in training_model.layer1.parameters():
            #              param.requires_grad=False
            #      for param in training_model.layer2.parameters():
            #              param.requires_grad=False
            #      for param in training_model.layer3.parameters():
            #              param.requires_grad=False
            #      for param in training_model.layer4.parameters():
            #              param.requires_grad=False
            #      for param in training_model.layer5.parameters():
            #              param.requires_grad=False

            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,training_model.parameters()), lr=LEARNING_RATE, weight_decay=config['l2_decay'])
            for name, param in training_model.named_parameters():
                     if (param.requires_grad):
                           print (name)

        f1_total = 0.0
        for batch in range(batch_num):
            training_model.train()
            batch_index = generate_batch(x_t.shape[0], config['batch_size'])
            test_batch_idx = generate_batch(x_test.shape[0], config['batch_size'])
            batch_x = x_t[batch_index]
            batch_y = y_t[batch_index]
            batch_res = res_t[batch_index]
            batch_x = batch_x.reshape(-1, 1, batch_x.shape[1])
            batch_x = torch.from_numpy(batch_x).float().to(config['device'])
            batch_y = torch.from_numpy(batch_y).long().to(config['device'])
            batch_res = torch.from_numpy(np.array(batch_res)).float()
            batch_test_x = x_test[test_batch_idx].reshape(-1,1, x_test.shape[1])
            batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
            # Forward pass
            alpha = 0.0
            if (config['model_type'] == 'dan'):
                  label_pred, loss_mmd = training_model(batch_x, batch_test_x)
                  batch_test_x = x_test[test_batch_idx].reshape(-1,1, x_test.shape[1])
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
                  tr_pred = np.argmax(label_pred.cpu().detach(), 1)
                  counter += 1
                  training_model.eval()
                  _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
                  f1_total += f1

            elif (config['model_type'] == 'dann'):
                 
                  p = float(counter + epoch * x_t.shape[0]) / config['num_epochs'] / x_t.shape[0]
                  alpha = 2. / (1. + np.exp(-10 * p)) - 1
                  optimizer.zero_grad() 
                  
                  domain_label = torch.zeros(config['batch_size'])
                  domain_label = domain_label.long().to(config['device'])
                  
                  # Using source data
                  class_output, domain_output = training_model(batch_x, alpha) 
                  err_s_label = loss_class(class_output, batch_y)
                  err_s_domain = loss_domain(domain_output, domain_label)
                  
                  ### Using target data
                  
                  domain_label = torch.ones(config['batch_size'])
                  domain_label = domain_label.long().to(config['device'])
                  
                  _, domain_output = training_model(batch_test_x, alpha)
                  err_t_domain = loss_domain(domain_output, domain_label)
                  err = err_t_domain + err_s_domain + err_s_label
                  err.backward()
                  optimizer.step()
                  
                  training_model.eval()
                  class_output,_ = training_model(batch_x, alpha)
                  tr_pred = class_output.data.cpu().max(1, keepdim=True)[1]
                  _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
                  f1_total += f1
            elif (config['model_type'] == 'jan'):
                  pass
            elif ((config['model_type'] == 'jdda') or (config['model_type'] == 'jdda_rnn') or (config['model_type'] == 'jdda_ff')):
                 output_source, hidden_source = training_model(batch_x) 
                 output_target, hidden_target = training_model(batch_test_x)
                 #source_loss = domain_loss(output_source, batch_y)
                 #inter_loss, intra_loss, centers_update_op = get_center_loss(hidden_source, batch_y, 0.5,10)

                 if (epoch >=0):
                      domain_discrep = coral_loss(hidden_source, hidden_target, config)
                      #discrim_loss = discriminative_loss(hidden_source, batch_y, config)
                 #inter_loss, intra_loss, centers_update_op = get_center_loss(hidden_source, batch_y, 0.5,10)
                 #print ("Inter-intra-centers",inter_loss,intra_loss,centers_update_op)
                      source_loss = domain_loss(output_source, batch_y)
                      err = source_loss + domain_discrep
                      #err = source_loss + 8* domain_discrep + 0.01* discrim_loss
                 else:
                      source_loss = domain_loss(output_source, batch_y)
                      err = source_loss
                 optimizer.zero_grad()
                 err.backward()
                 optimizer.step()

                 training_model.eval()
                 class_output,_ = training_model(batch_x)
                 tr_pred = class_output.data.cpu().max(1, keepdim=True)[1]
                 _, _, f1, _ = prfs(batch_y.cpu(), tr_pred, average='weighted')
                 f1_total += f1
                 if (epoch == config['num_epochs'] - 1):
                          transformed_source = np.concatenate((transformed_source, hidden_source.cpu().detach())) 

            if (counter + 1) % 100 == 0:
                  print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Best F1: {:.4f}'.format(epoch + 1, config['num_epochs'], batch + 1, batch_num, err.item(), f1))
                  counter += 1

        print("------epoch : ", epoch, " Loss: ", err.item(), " Training F1:", round((f1_total / batch_num), 4))
        training_model.eval()
        eval_f1,_ = evaluate_test(x_eval, y_eval, res_eval, training_model,config, False)
        f1_total += eval_f1

        if (eval_f1 >= best_f1):
            best_f1 = eval_f1
            torch.save(training_model.state_dict(), config['ckpt_path'] + 'best_model.pth')
        
        print("Current F1 score (evaluation) ", eval_f1)
        print("Best F1 score (evaluation) ", best_f1)
    transformed_source = np.delete(transformed_source, (0), axis=0)
    print ("All completed")
    print ("Transformed source", transformed_source.shape)
    return transformed_source
    

def coral_loss(h_src, h_trg, config, gamma=1e-3):
	h_src = h_src - torch.mean(h_src, dim=0)
	h_trg = h_trg - torch.mean(h_trg, dim=0)
	cov_src = (1. / (h_src.shape[0]) * torch.mm(h_src, torch.transpose(h_src, 0, 1)))
	cov_trg = (1. / (h_trg.shape[0]) * torch.mm(h_trg, torch.transpose(h_trg, 0, 1)))
	coral= ((cov_src - cov_trg) ** 2).mean()
	return coral

def discriminative_loss(x_source, y_source, config):
	margin0 = 0
	margin1 = 100
	batch_size = x_source.shape[0]
	W = torch.zeros(batch_size, batch_size).to(config['device'])
	x_source =x_source.reshape(batch_size, -1)
	for i in range (W.shape[0]):
		for j in range (W.shape[1]):
			if (y_source[i] == y_source[j]):
				W[i,j] = 1
	
	norm = lambda x: torch.sum(x**2 , dim=1)
	pdist =nn.PairwiseDistance(p=2)
	F0 = pdist(x_source, x_source)
	F0 = (torch.max(torch.zeros_like(F0), F0 - margin0))**2
	F1 = (torch.max(torch.zeros_like(F0), margin1 - F0))**2
	intra_loss = (F0.mul(W)).mean()
	inter_loss = (F1.mul( 1-W)).mean()
	discrim_loss = (intra_loss + inter_loss) / (batch_size * batch_size)
	return discrim_loss

def evaluate_test(x_eval, y_eval, res_eval, model, config, test=False):
    total_pred = np.array([], dtype=np.float64)
    total_y_test = np.array([], dtype=np.int64)
    if (config['old'] == False):
            size = 2048
    else:
            size = 1024

    transformed_target = np.zeros(shape=(1, size),dtype=np.float64)
    test_batch_num = int(math.ceil(x_eval.shape[0] / float(config['batch_size'])))
    cum_acc = []
    cum_f1 = []
    with torch.no_grad():
        for i in range(test_batch_num):
            begin_index = i * config['batch_size']
            end_index = min((i + 1) * config['batch_size'], x_eval.shape[0])
            batch_test_x = x_eval[begin_index: end_index]
            batch_test_y = y_eval[begin_index: end_index]
            batch_test_res = res_eval[begin_index: end_index]
            if ( config['model_type'] == 'dan' or config['model_type'] == 'dann'):
                batch_test_x = batch_test_x.reshape(-1, 1, 1, batch_test_x.shape[1])
            elif (config['model_type'] == 'jdda' or config['model_type'] == 'jdda_rnn' or config['model_type'] == 'jdda_ff'):
                batch_test_x = batch_test_x.reshape(-1,1, batch_test_x.shape[1])
            batch_test_x = torch.from_numpy(batch_test_x).float().to(config['device'])
            batch_test_y = torch.from_numpy(batch_test_y).long().to(config['device'])
            batch_test_res = torch.from_numpy(batch_test_res).float()
            output_test,hidden_test = model(batch_test_x)
            _, predicted = torch.max(output_test.data, 1)
            # loss = _loss(batch_test_res, batch_test_y, output_test)
            total_pred = np.concatenate((total_pred, predicted.cpu()))
            total_y_test = np.concatenate((total_y_test, batch_test_y.cpu()))
            batch_acc = accuracy_score(batch_test_y.cpu().numpy(), predicted.cpu().numpy())
            _, _, batch_f1, _ = prfs(batch_test_y.cpu().numpy(), predicted.cpu().numpy(),  average='weighted')
            cum_acc.append(batch_acc)
            cum_f1.append(batch_f1)
            transformed_target = np.concatenate((transformed_target, hidden_test.cpu().numpy()))
    if (test == True):
        print(classification_report(total_y_test, total_pred, digits=4))
        acc = accuracy_score(total_y_test, total_pred)
        cum_acc = np.array(cum_acc)
        cum_f1 = np.array(cum_f1)
        print("Testing accuracy", acc)
        print ("Average acc and std ", np.mean(cum_acc), np.std(cum_acc))
        print ("Average F1 and std ", np.mean(cum_f1), np.std(cum_f1))
    _, _, f1, _ = prfs(total_y_test, total_pred, average='weighted')
    transformed_target = np.delete(transformed_target, (0), axis=0)
    print ("Target - Eval", transformed_target.shape, x_eval.shape)
    return f1, transformed_target

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    #assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    #assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])

l2_loss = nn.MSELoss()

def get_center_loss(features, labels, alpha=0.5, num_classes=6):
	len_features = features.shape[1]
	labels = labels.reshape(-1, 1)
	centers = torch.zeros(num_classes, len_features)
	print ("Shape test feature - label", features.shape, labels.shape)
	idx = np.arange(6)
	
	centers0 = torch.zeros(num_classes, 64).scatter_add_(1, labels.cpu(), features.cpu())
	#centers0 = unsorted_segment_sum(features.cpu(), labels.cpu(), num_classes)
	print ("Center check", centers0.shape)
	EdgeWeights = torch.ones(num_classes, num_classes) - torch.eye(num_classes)
	margin = 100
	norm = lambda x: torch.sum(x ** 2 , 1)
	center_pairwise_dist = (centers0.repeat(2) - centers0.transpose(0,1)).transpose(0,1)
	loss_0 = torch.sum(torch.bmm(torch.max(0.0, margin - center_pairwise_distance), EdgeWeights))


	centers_batch = torch.gather(centers, labels)
	diff = centers_batch - features
	unique_label, unique_idx, unique_count = torch.unique(labels, return_inverse=True, return_counts=True)
	appear_times = torch.gather(unique_count, unique_idx)
	appear_times = appear_times.reshape(-1,1)
	diff = diff / (torch.ones(shape=appear_times.shape) + appear_times)
	diff = alpha * diff

	loss_1 = nn.MSELoss(features - center_batch)
	centers_update_op = scatter_sub(centers, labels, out=diff)

	return loss_0, loss_1, centers_update_op
	#unique_label, unique_idx, unique_count = 

def main():
    config = parse_argument()
    plot_directory = './tsne_post/'
    MakeDirectory(plot_directory)
    print ("Config", config)
    training_files = glob.glob(config['train_path'] + '*')
    testing_files = glob.glob(config['test_path'] + '*')

    MakeDirectory(config['report_path'])
    df = pd.DataFrame()
    df['Test (down)/ Train (across)'] = np.array([l.split('/')[-1].split('.')[0] for l in testing_files])

    for train_file in training_files:
        print("Training file", train_file)
        x_train, y_train, train_resources = load_data(train_file, config)
        indices = np.arange(x_train.shape[0])
        x_t, x_eval, y_t, y_eval, idx_t, idx_te = train_test_split(x_train, y_train, indices, test_size=0.2,
                                                                   random_state=0)
        res_t = train_resources[idx_t]
        res_eval = train_resources[idx_te]
        input_size = x_t.shape[1]
        MakeDirectory(config['ckpt_path'])
        
        source_name = train_file.split('/')[-1].split('.')[0]
        print("--------------- Finish loading data ------------------")
        print('Initializing Variables')
        test_results = []
        for test_file in testing_files:
            target_name = test_file.split('/')[-1].split('.')[0]
            if (config['model_type'] == 'dann' or config['model_type'] == 'dan' or (config['model_type'] == 'jdda') or (config['model_type'] == 'jdda_rnn')):
                  x_test, y_test, test_resources = load_data(test_file, config)
                  print("--------- Starting testing ---------")
                  print("Testing file", test_file)

                  transformed_source = train_dnn(x_t, y_t, res_t, x_eval, y_eval,res_eval,x_test, y_test, config)
                  # Testing
                  if (config['model_type'] == 'dan'):
                      training_model = transfer_model.DANNet(config).to(config['device'])
                  elif (config['model_type'] == 'dann'):
                      training_model= transfer_model.CNNModel(config).to(config['device'])
                  elif (config['model_type'] == 'jdda'):
                      training_model = transfer_model.JDDA(config).to(config['device'])
                  elif (config['model_type'] == 'jdda_rnn'):
                      training_model = transfer_model.JDDA_RNN(x_train.shape[1], config).to(config['device'])
                  training_model.load_state_dict(torch.load(config['ckpt_path'] + 'best_model.pth'))
                  test_f1, transformed_target = evaluate_test(x_test, y_test, test_resources, training_model, config, True)
                  test_results.append(test_f1)
                  print("Testing F1 score", test_f1)
                  print("-------------------------------")
                  #plot_tsne(transformed_source, source_name, transformed_target, target_name, plot_directory)
        df[train_file.split('/')[-1].split('.')[0]] = test_results 
        
        print("Testing is completed")
        shutil.rmtree(config['ckpt_path'])
        print("Current best model directory is removed")
        output_report = config['report_path'] + config['model_type'] + '_report'
        for name in config.keys():
          if (name != 'model_type'):
                  if ('train_path' in name or 'test_path' in name):
                       output_report += '_'+  str(name) + str(config[name].split('/')[-2])
                  elif ('ckpt_path' in name or 'report_path' in name):
                       pass
                  else:
                       output_report += '_' + str(name) + str(config[name])

    
        df.to_csv(output_report + '.csv', index=False)

if __name__ == "__main__":
      main()
