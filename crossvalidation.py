'''
Created on 16 Jul 2015

@author: navjotkukreja
'''

class cv:
    def __init__(self, data, n_fold_cv):
        n_folds = self.chunks(data, n_fold_cv)

        labels = []
        train_paths = []
        test_paths = []
        selected_fold = 1
        for i in range(0, len(n_folds)):
            if i==selected_fold:
                test_paths=n_folds[i]
            else:
                train_paths+=n_folds[i]
        return (train_paths, test_paths)
    
    def __chunks_(self, lst,n):
        return [ lst[i::n] for i in xrange(n) ]