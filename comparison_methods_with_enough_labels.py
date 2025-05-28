from data_process.data_load import load_pre_data, load_ft_data, load_test_data
from train.CNN_train import CNNTrain
from train.WSL_train import FtTrain,Test
from train.Benchmark_train import BenchmarkTrain
import argparse
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tool.metrix import eval_metrix
import os


def get_args():
    """
    Parse command line arguments
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Hyper Parameters')
    # 标签数据仅来自单一工况
    # Label data only from a single operating condition
    
    train_cells = {
          'Dataset 1': ['25-1','25-2','25-3','25-4'],
          'Dataset 2': ['1C-4','1C-5','1C-6','1C-7'],
          'Dataset 3': ['0-CC-1','0-CC-2'],
          'Dataset 4': ['CY25-05_1-#12','CY25-05_1-#14','CY25-05_1-#16'],
          'Dataset 5': ['2C-5','2C-6','2C-7'],
          'Dataset 6': ['25_1a_100','25_1b_100','25_1c_100']
          }
    test_cells = {
          'Dataset 1': ['25-'+str(i+1) for i in range(4,8)]+['45-'+str(i+1) for i in range(6)],
          'Dataset 2': ['1C-8','1C-9','1C-10']+['2C-4','2C-5']+['3C-'+str(i+1) for i in range(3,10)],
          'Dataset 3': ['0-CC-3','10-CC-1','10-CC-2','10-CC-3','25-CC-1','25-CC-2','25-CC-3','40-CC-1','40-CC-2','40-CC-3'],
          'Dataset 4': ['CY25-05_1-#18','CY25-05_1-#19']+['CY35-05_1-#1','CY35-05_1-#2']+['CY45-05_1-#'+str(i+1) for i in range(20,28)],
          'Dataset 5': ['2C-8']+['3C-'+str(i+1) for i in range(4,15)]+['4C-5','4C-6'],
          'Dataset 6': ['25_1d_100','25_0.5a_100','25_0.5b_100','25_2a_100','25_2b_100','25_3a_100','25_3b_100','25_3c_100','25_3d_100']+
                        ['35_1a_100','35_1b_100','35_1c_100','35_1d_100','35_2a_100','35_2b_100']
    }
    parser.add_argument('--condition', type=str, default='single_condition_label', help='Conditions included in labeled data')

    # 标签数据覆盖所有工况
    # Label data covers all operating conditions

    # train_cells = {
    #       'Dataset 1': ['25-1','25-2','45-1','45-2'],
    #       'Dataset 2': ['1C-4','1C-5','2C-4','3C-4','3C-5'],
    #       'Dataset 3': ['0-CC-2','10-CC-1','25-CC-1','40-CC-1'],
    #       'Dataset 4': ['CY25-05_1-#12','CY25-05_1-#14','CY25-05_1-#16','CY35-05_1-#1','CY45-05_1-#21','CY45-05_1-#22','CY45-05_1-#23'],
    #       'Dataset 5': ['2C-5','2C-7','3C-5','3C-6','4C-5'],
    #       'Dataset 6': ['25_0.5a_100','25_1a_100','25_2a_100','25_3a_100','35_1a_100','35_2a_100',]
    #       }
    # test_cells = {
    #         'Dataset 1': ['25-'+str(i+1) for i in range(2,8)]+['45-'+str(i+1) for i in range(2,6)],
    #         'Dataset 2': ['1C-'+str(i+1) for i in range(5,10)]+['2C-5']+['3C-'+str(i+1) for i in range(5,10)],
    #         'Dataset 3': ['0-CC-1','0-CC-3','10-CC-2','10-CC-3','25-CC-2','25-CC-3','40-CC-2','40-CC-3'],
    #         'Dataset 4': ['CY25-05_1-#18','CY25-05_1-#19']+['CY35-05_1-#2']+['CY45-05_1-#'+str(i+1) for i in range(23,28)],
    #         'Dataset 5': ['2C-6','2C-8']+['3C-'+str(i+1) for i in range(6,15)]+['4C-6'],
    #         'Dataset 6': ['25_0.5b_100','25_1b_100','25_1c_100','25_1d_100','25_2b_100','25_3b_100','25_3c_100','25_3d_100']+
    #                     ['35_1b_100','35_1c_100','35_1d_100','35_2b_100']
    # }
    # parser.add_argument('--condition', type=str, default='full_condition_label', help='Conditions included in labeled data')

    parser.add_argument('--target_datasets', type=list, default=['Dataset 1','Dataset 2','Dataset 3', 'Dataset 4', 'Dataset 5', 'Dataset 6'], help='The name of target dataset')
    parser.add_argument('--ft_data_num',type=str,default='all',help='Numbers of training samples')
    parser.add_argument('--train_cells',type=dict,default=train_cells,help='Training cells with all label')
    parser.add_argument('--test_cells',type=dict,default=test_cells,help='Test cells')
    args = parser.parse_args()
    return args


def get_train_data(args):
    train_x,train_y = [],[]
    for train_ci in args.train_cell:
        setattr(args, 'ft_cell', train_ci)
        data_ci = load_ft_data(args)
        train_x.append(data_ci['X'])
        train_y.append(data_ci['Y'])
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    return train_x, train_y

def cnn_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition,args.target_dataset,'CNN')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        trainer = CNNTrain(args)
        trainer.train(train_x,train_y)

        eval_metrics = {}
        test_results = {}
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = trainer.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

def Benchmark_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition, args.target_dataset,'Benchmark')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        trainer = BenchmarkTrain(args)
        trainer.train(train_x,train_y)

        eval_metrics = {}
        test_results = {}
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = trainer.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

def WSL_el_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition,args.target_dataset,'WSL_el')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)

        setattr(args,'source_dataset',args.target_dataset)
        setattr(args,'ft_epochs',50)
        setattr(args,'ft_batch_size',128)
        setattr(args,'ft_lr',0.0005)
        setattr(args, 'pre_model_file', f'results/soh_individual_dataset_results/{args.target_dataset}')
        ft_train = FtTrain(args)
        ft_train.train(train_x,train_y)

        eval_metrics = {}
        test_results = {}
        tester = Test(args)
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            true_capacity = test_data['Y']
            est_capacity, metric = tester.test(test_data['X'], true_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

def rf_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition,args.target_dataset,'RF')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        rf = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=9)
        rf.fit(train_x, train_y)

        eval_metrics = {}
        test_results = {}
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            test_x = scaler.fit_transform(test_data['X'])
            true_capacity = test_data['Y']
            est_capacity = rf.predict(test_x)
            metric = eval_metrix(true_capacity, est_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

def gpr_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition,args.target_dataset,'GPR')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        kernel = Matern(nu=2.5)
        gpr = GaussianProcessRegressor(kernel=kernel,alpha=1e-5,n_restarts_optimizer=3,random_state=9)
        gpr.fit(train_x, train_y)

        eval_metrics = {}
        test_results = {}
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            test_x = scaler.fit_transform(test_data['X'])
            true_capacity = test_data['Y']
            est_capacity = gpr.predict(test_x)
            metric = eval_metrix(true_capacity, est_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

def svr_results():
    args = get_args()
    train_cells = args.train_cells
    test_cells = args.test_cells
    for target_dataset in args.target_datasets:
        setattr(args, 'target_dataset', target_dataset)
        setattr(args, 'train_cell', train_cells[target_dataset])
        setattr(args, 'test_cell', test_cells[target_dataset])
        save_file = os.path.join('results', 'comparison_methods_with_enough_labels_results',args.condition, args.target_dataset,'SVR')
        setattr(args, 'ft_files', save_file)
        if not os.path.exists(args.ft_files):
                    os.makedirs(args.ft_files)
        train_x,train_y = get_train_data(args)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        svr = SVR(kernel='rbf',epsilon=0.001)
        svr.fit(train_x, train_y)

        eval_metrics = {}
        test_results = {}
        for test_ci in args.test_cell:
            test_data = load_test_data(args.target_dataset,test_ci)
            test_x = scaler.fit_transform(test_data['X'])
            true_capacity = test_data['Y']
            est_capacity = svr.predict(test_x)
            metric = eval_metrix(true_capacity, est_capacity)
            
            test_results[f'{test_ci}_est'] = est_capacity
            test_results[f'{test_ci}_true'] = true_capacity
            eval_metrics[test_ci] = metric
        test_results_df = pd.DataFrame({k: pd.Series(v) for k, v in test_results.items()})
        test_results_df.to_csv(f'{args.ft_files }/test_results.csv', index=False)
        
        eval_metrics_df = pd.DataFrame(eval_metrics, index=['RMSE', 'MAPE'])
        eval_metrics_df.T.to_csv(f'{args.ft_files }/eval_metrics.csv')

if __name__ == '__main__':
    # cnn_results()
    # Benchmark_results()
    # rf_results()
    # gpr_results()
    # svr_results()
    # WSL_el_results()
    pass
    