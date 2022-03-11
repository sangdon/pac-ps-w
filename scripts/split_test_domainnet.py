import os, sys
import numpy as np

ROOT='data/domainnet'
DOMAIN_LIST=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

if __name__ == '__main__':
    seed = 0
    split_ratio = 0.5
    for dn in DOMAIN_LIST:
        fn_test = os.path.join(ROOT, dn + '_test.txt')
        data = open(fn_test, 'r').readlines()
        
        n = len(data)
        n_val = int(n*split_ratio)
        n_test = n - n_val
        print(f'n = {n}, n_val = {n_val}, n_test = {n_test}')
        
        np.random.seed(seed)
        i_rnd = np.random.permutation(n)
        i_rnd_val = i_rnd[:n_val]
        i_rnd_test = i_rnd[n_val:]
        i_rnd_val.sort()
        i_rnd_test.sort()
        
        data_val = [data[i] for i in i_rnd_val]
        data_test = [data[i] for i in i_rnd_test]

        open(os.path.join(ROOT, dn+'_val_split.txt'), 'w').writelines(data_val)
        open(os.path.join(ROOT, dn+'_test_split.txt'), 'w').writelines(data_test)
        

    
