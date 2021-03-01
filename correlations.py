"""
calculate correlation given time series
"""

import os
import sys
import numpy as np
from itertools import combinations
from nilearn.connectome import ConnectivityMeasure

def load_ts(atlas_name, sess_type='pain'):
    """load time series files into list (pain/relief/rest type)"""
    input_dir = '../output/'+atlas_name
    sess_ls = []
    for func_file in os.listdir(input_dir):
        if func_file.endswith('.npy'):
            fname = func_file.split('.')[0]
            time_series = np.load(os.path.join(input_dir,fname+'.npy'))
            if sess_type in func_file:
                sess_ls.append(time_series)
    return sess_ls

def static_corr(atlas_name, corr_type='correlation', sess_type='pain'):
    """calculate static correlation (correlation/partial)"""
    # pool sessions
    input_dir = '../output/'+atlas_name 
    sess_ls = load_ts(input_dir, sess_type)
    sess_mat = np.stack(sess_ls)
    # calculate correlation
    if corr_type in ['correlation', 'partial correlation', 'covariance', 'precision']:
        connectivity = ConnectivityMeasure(kind=corr_type, vectorize=True)
        connectomes = connectivity.fit_transform(sess_mat)
        print(connectomes.shape)
    else:
        raise ValueError('Correlation type specified does not exits')
    # save correlation
    save_dir = os.path.join(input_dir, 'static_corr')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = corr_type.split(' ')[0]+'_'+sess_type+'.npy'
    output_file = os.path.join(save_dir, file_name)
    np.save(output_file, connectomes)
    print(f'output saved to {output_file}')
    return connectomes

def dynamic_corr(func_file):
    """calculate garch-dcc"""
    sj_mat = np.load(os.path.join(func_file))
    comb = list(combinations(np.arange(sj_mat.shape[1]),2))
    # calculate dcc
    dcc_ls = []
    for c in comb:
        t1 = sj_mat[:,c[0]]
        t2 = sj_mat[:,c[1]]
        dcc_out = calc_dcc(t1, t2)
        dcc_ls.append(dcc_out)
    dcc_sj = np.stack(dcc_ls)
    print(dcc_sj.shape)
    # save correlation
    tmp = func_file.split('/')
    path_name = tmp[:-1]
    file_name = tmp[-1].split('.')[0]
    output_dir = os.path.join(*path_name)
    save_dir = os.path.join(output_dir, 'dynamic_corr')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_name = file_name+'_dcc.npy'
    output_file = os.path.join(save_dir, file_name)
    np.save(output_file, dcc_sj)
    print(f'dcc output saved to {output_file}')
    return dcc_sj


def calc_dcc(t1, t2):
    """calcualte dcc between 2 time series"""
    from DCC_GARCH.GARCH import GARCH, garch_loss_gen
    t1_model = GARCH(1,1)
    t1_model.set_loss(garch_loss_gen(1,1))
    t1_model.set_max_itr(1)
    t1_model.fit(t1)
    t1_sigma = t1_model.sigma(t1)
    t1_epsilon = t1/t1_sigma

    t2_model = GARCH(1,1)
    t2_model.set_loss(garch_loss_gen(1,1))
    t2_model.set_max_itr(1)
    t2_model.fit(t2)
    t2_sigma = t2_model.sigma(t2)
    t2_epsilon = t2/t2_sigma    

    epsilon = np.array([t1_epsilon, t2_epsilon])

    from DCC_GARCH.DCC import DCC, dcc_loss_gen, R_gen
    dcc_model = DCC()
    dcc_model.set_loss(dcc_loss_gen())
    dcc_model.fit(epsilon)

    # get DCC R (conditional correlation matrix)
    ab = dcc_model.get_ab()
    tr = epsilon
    R_ls = R_gen(tr,ab)
    R = np.array(R_ls)
    # flatten Rt
    K = R.shape[1]
    Rt_triu = R[:,np.triu(np.ones((K,K)),1)>0].T
    dcc_out = Rt_triu[0]
    # print(dcc_out.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(dcc_out)
    # plt.savefig('./test.png')
    return dcc_out





if __name__=="__main__":
    # for atlas in ['yeo','fan','msdl']:
    #     for sess in ['pain', 'rest', 'relief']:
            # # static corr
            # for corr in ['correlation', 'partial correlation', 'covariance', 'precision']:
            #     static_corr(atlas, corr_type=corr, sess_type=sess)
    
    # dynamic correlation
    func_file = sys.argv[1]
    dynamic_corr(func_file)