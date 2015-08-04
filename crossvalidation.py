'''
Created on 16 Jul 2015

@author: navjotkukreja
'''
from random import shuffle
class cv:
    def __init__(self, data, n_fold_cv):
        shuffle(data)
        n_folds = self.__chunks_(data, n_fold_cv)

        labels = []
        train_paths = []
        test_paths = []
        selected_fold = 1
        for i in range(0, len(n_folds)):
            if i==selected_fold:
                test_paths=n_folds[i]
            else:
                train_paths+=n_folds[i]
        self.train = train_paths
        self.test = test_paths
    
    def get(self):
        return (self.train, self.test)
    
    def __chunks_(self, lst,n):
        return [ lst[i::n] for i in xrange(n) ]