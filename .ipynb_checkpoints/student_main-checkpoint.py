#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from pathlib import Path
import torch
import gc
import random
import numpy as np
import os
import csv
from utils.student_utils import run_for_seed

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--dataset', default='Haptics')
parser.add_argument('--n_channels', default=1, type=int)
parser.add_argument('--output_classes', default=7, type=int)#automated within code
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--epochs', default=800, type=int)
parser.add_argument('--patience', default=50, type=int)
parser.add_argument('--seed', default=1001, type=int)
parser.add_argument('--seed_array', type=int, nargs='+', default=[24,10200], help='The array of seed integers')
parser.add_argument('--val_size', default=0.2)
parser.add_argument('--data_folder', default=Path('../ucr_data'))
# parser.add_argument('--data_folder', default=Path('../UAE_Multi_datasets/'))
parser.add_argument('--save_dir', default='student_models/')
parser.add_argument('--teacher_load', default='teacher_models_bulk/')
parser.add_argument('--common_dir', default='student_results/')
parser.add_argument('--result_csv', default='test_results_student.csv')
parser.add_argument('--avg_result_csv', default='avg_test_results_student.csv')
parser.add_argument('--max_result_csv', default='max_test_results_student.csv')
parser.add_argument('--warm_up', default=300, type=int) 
parser.add_argument('--warm_up_plus', default=0, type=int) 

#for LSTM hidden size and number of layers
parser.add_argument('--teacher_hidden_size', default=100, type=int)
parser.add_argument('--student_hidden_size', default=32, type=int)
parser.add_argument('--teacher_num_layers', default=3, type=int)
parser.add_argument('--student_num_layers', default=2, type=int)


#distillation ratios
parser.add_argument('--task_ratio', default=1, type=float)
parser.add_argument('--dist_ratio', default=0, type=float)
parser.add_argument('--dtw_rkd_ratio', default=0, type=float)
parser.add_argument('--hinton_loss_ratio', default=0, type=float)
parser.add_argument('--bench_rkd_dist_ratio', default=0, type=float)
parser.add_argument('--bench_rkd_angle_ratio', default=0, type=float)
parser.add_argument('--bench_fitnet_ratio', default=0, type=float)
parser.add_argument('--bench_attention_ratio', default=0, type=float)
parser.add_argument('--bench_temp_dist_ratio', default=0, type=float)
parser.add_argument('--bench_DKD_ratio', default=0, type=float)
parser.add_argument('--bench_VID_ratio', default=0, type=float)
parser.add_argument('--bench_gdpd_ratio', default=0, type=float)
parser.add_argument('--ratios', type=float, nargs='+', default=[0.01 ,0.1], help='The array of distillation ratios')

#for diffusion model
parser.add_argument('--student_diff_chan', default=32, type=float)
parser.add_argument('--teacher_diff_chan', default=32, type=float)
parser.add_argument('--kernel_size', default=3, type=int)
parser.add_argument('--inference_steps', default=5, type=int)
parser.add_argument('--num_train_timesteps', default=1000, type=int)
parser.add_argument('--use_ae', default=True)
parser.add_argument('--ae_channels', default=32, type=float)


#for truncating each time series to first given number of time steps
parser.add_argument('--is_truncate', default=1, type=int)
parser.add_argument('--truncate_ratio', default=0.8, type=float)
parser.add_argument('--channel_chunk', default=3, type=int)

parser.add_argument('--is_downsample', default=1, type=int)
parser.add_argument('--downsample_size', default=100, type=int)
parser.add_argument('--align_corners', default=True)

#for slaiencyKD
parser.add_argument('--num_samples', default=50, type=int)
parser.add_argument('--sub_seq_len', default=5, type=int)

#for fitnets
parser.add_argument('--load', default=None)
parser.add_argument('--last_weight', default=None)

#for HKD and DKD(default weights from orig papaer)
parser.add_argument('--temperature', default=2, type=float)
parser.add_argument('--alpha_DKD', default=1.0, type=float)
parser.add_argument('--beta_DKD', default=8.0, type=float)
parser.add_argument('--dtw_gamma', default=1.2, type=float)

#for VID
parser.add_argument('--init_pred_var_VID', default=5.0, type=float)
parser.add_argument('--eps_VID', default=1e-5, type=float)

parser.add_argument('--base', default='FCNBaselineSmall')
parser.add_argument('--teacher_base',default='FCNBaseline')
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')
parser.add_argument('--teacher_l2normalize', choices=['true', 'false'], default='true')

inp_args = parser.parse_args()
inp_args

#clear cache
torch.cuda.empty_cache()
gc.collect() 

inp_args.teacher_load = inp_args.teacher_load+ inp_args.dataset + '/best.pth'


# In[2]:


def train_for_seed(inp_args, save_path_attributes):
    collected_test_res = []    
    for random_seed in inp_args.seed_array:
        inp_args.seed = random_seed
        #eliminate randomness
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
            set_random_seed(inp_args.seed)
    
        test_results , student = run_for_seed(inp_args)
        metrics = list(test_results.values())        
        collected_test_res.append(metrics)

    
        file_path = inp_args.common_dir + inp_args.result_csv
        full_list = save_path_attributes +[inp_args.seed]+ [round(f, 4) if type(f) == np.float64  else f for f in  list(test_results.values())]+[inp_args.truncate_ratio]
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(full_list)  

        if inp_args.save_dir is not None:
            save_path = f"{inp_args.save_dir}{'_'.join(map(str, save_path_attributes))}_{random_seed}"
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(student.state_dict(), "%s/%s"%(save_path, "best.pth"))
            
        
    arr = np.asarray(collected_test_res, dtype=np.float64) #drop last three columns
    metrics_array = arr[:, :-3] 
    mean_metrics = metrics_array.mean(axis=0)
    mean_metrics = [round(f, 4) if type(f) == np.float64  else f for f in  mean_metrics]
    
    std_metrics  = metrics_array.std(axis=0, ddof=1) 
    std_metrics = [round(f, 4) if type(f) == np.float64  else f for f in  std_metrics]
    
    
    file_path = inp_args.common_dir + inp_args.avg_result_csv
    full_list = [save_path_attributes[0]]+ [inp_args.seed]+ [inp_args.truncate_ratio]+  mean_metrics+ std_metrics

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(full_list) 
        
    return [round(100 * f,2) for f in  mean_metrics] + list(std_metrics)


# In[3]:


max_test_res_over_distillation_ratios = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
max_ratio = 0.0
for ratio in inp_args.ratios:
    inp_args.bench_gdpd_ratio = ratio
    inp_args.hinton_loss_ratio=ratio
    save_path_attributes = [
        inp_args.dataset,
        inp_args.task_ratio,
        inp_args.hinton_loss_ratio,
        inp_args.bench_rkd_dist_ratio,
        inp_args.bench_rkd_angle_ratio,
        inp_args.bench_fitnet_ratio,
        inp_args.bench_attention_ratio,
        inp_args.bench_temp_dist_ratio,
        inp_args.bench_DKD_ratio,
        inp_args.bench_VID_ratio,
        inp_args.bench_gdpd_ratio
        
    ]
    
    current_test_res =  train_for_seed(inp_args, save_path_attributes)
    if current_test_res[1] > max_test_res_over_distillation_ratios[1]: #max entry selected based on avg_auc_prc
        max_test_res_over_distillation_ratios = current_test_res
        max_ratio = ratio

file_path = inp_args.common_dir + inp_args.max_result_csv
full_list = [save_path_attributes[0]]+ [max_ratio]+ [inp_args.truncate_ratio]+ max_test_res_over_distillation_ratios
with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(full_list) 


# In[ ]:




