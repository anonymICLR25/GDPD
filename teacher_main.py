import argparse
from pathlib import Path
import torch
import gc
import random
import numpy as np
import os
import csv
import copy
from typing import cast, Any, Dict, List, Tuple, Optional
import json
from utils.teacher_utils_Inception import run_for_seed

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))

parser.add_argument('--mode', choices=["train", "eval"], default="train")
parser.add_argument('--load', default=None)
parser.add_argument('--base', default='FCNBaseline')
parser.add_argument('--dataset',default='WordsSynonyms')
parser.add_argument('--n_channels', default=1, type=int)
parser.add_argument('--output_classes', default=3, type=int) # automate within the code
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')
parser.add_argument('--lr_decay_gamma', default=0.5, type=float)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--patience', default=50, type=int)
parser.add_argument('--downsample_size', default=100, type=int)
parser.add_argument('--val_size', default=0.2)
parser.add_argument('--seed', default=1001, type=int)
parser.add_argument('--seed_array', type=int, nargs='+', default=[1, 24, 300, 49000, 1001], help='The array of seed integers')
parser.add_argument('--common_dir', default='teacher_results/')
parser.add_argument('--save_dir', default='teacher_models/')
parser.add_argument('--data_folder', default=Path('../UAE_Multi_datasets/'))
# parser.add_argument('--data_folder', default=Path('../ucr_data/'))
parser.add_argument('--result_csv', default='test_results_teacher.csv')
parser.add_argument('--avg_result_csv', default='avg_test_results_teacher.csv')
parser.add_argument('--max_result_csv', default='max_test_results_teacher.csv')

#for LSTM architecture
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--hidden_size', default=100, type=int)

#for Resnet architecture
parser.add_argument('--mid_channels', default=64, type=int)

inp_args = parser.parse_args()
inp_args

#clear cache
torch.cuda.empty_cache()
gc.collect() 


#construct the save path
inp_args.save_dir = inp_args.save_dir+ inp_args.dataset + '/'


def train_for_seed(inp_args):
    avg_test_res_over_seeds = [0,0,0,0] #ROC_AUC, AUC-PRC, AVG_PRECISION, ACC
    max_test_res_over_seeds = [-1,-1,-1,-1] #ROC_AUC, AUC-PRC, AVG_PRECISION, ACC
    max_state_dict = None
    for random_seed in inp_args.seed_array:
        inp_args.seed = random_seed
        #eliminate randomness
        try:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        except:
            get_ipython().run_line_magic('env', 'CUBLAS_WORKSPACE_CONFIG=:4096:8')
        torch.use_deterministic_algorithms(True)
        for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
            set_random_seed(inp_args.seed)
    
        test_results , model = run_for_seed(inp_args)
        test_results = list(test_results.values())
        avg_test_res_over_seeds= [a+b for a,b in zip(avg_test_res_over_seeds , test_results)]
        if test_results[1] > max_test_res_over_seeds[1]: #max entry selected based on AUC-PRC
            max_test_res_over_seeds = test_results
            max_state_dict = copy.deepcopy(model.state_dict())
    
        #save each entry in a csv file
        file_path = inp_args.common_dir + inp_args.result_csv
        full_list = [inp_args.dataset] +[inp_args.seed]+ [round(f, 4) if type(f) == np.float64  else f for f in test_results]
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(full_list)  
            
    #save average values in a csv file
    avg_test_res_over_seeds = [a/len(inp_args.seed_array) for a in avg_test_res_over_seeds]
    file_path = inp_args.common_dir + inp_args.avg_result_csv
    full_list = [inp_args.dataset]+ [inp_args.seed]+ [round(f, 4) if type(f) == np.float64  else f for f in  avg_test_res_over_seeds]
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(full_list) 

    #save max entry in a csv file
    file_path = inp_args.common_dir + inp_args.max_result_csv
    full_list = [inp_args.dataset] + [inp_args.seed]+ [round(100 * f,2) for f in  max_test_res_over_seeds]
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(full_list) 

    #save weights corresponding to max weight for distillation
    model.load_state_dict(cast(Dict[str, torch.Tensor], max_state_dict)) #load model from final iteration with weights corresponging to max-result-entry
    if inp_args.save_dir is not None:
        if not os.path.isdir(inp_args.save_dir):
            os.mkdir(inp_args.save_dir)
        torch.save(model.state_dict(), "%s/%s"%(inp_args.save_dir, "best.pth"))
    with open("%s/result.txt"%inp_args.save_dir, 'w') as f:
        f.write(json.dumps(max_test_res_over_seeds))

#train for random seed array
train_for_seed(inp_args)


  