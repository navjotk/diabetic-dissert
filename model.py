'''
Created on 16 Jul 2015

@author: navjotkukreja
'''

import numpy as np
from sklearn.metrics import confusion_matrix
import optunity
import optunity.metrics
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
import log
import sklearn

class model:
    def __init__(self, images, labels, n_folds_cv):
        self.__space_ = {'kernel': {'linear': {'C': [0, 2]},
                    'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                    'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                    }
         }
        
        self.__sgd_space_ = {'alpha1': [0.0001, 5], 'power_t':[0.1,0.9]}
        self.__log = log.Logger()
        self.__cv_decorator_ = optunity.cross_validated(x=images, y=labels, num_folds=n_folds_cv)
        
    
    def fixed_params(self, C, logGamma):
        self.__log.write("Training Model")
        self.__svm_rbf_tuned_auroc_ = self.__cv_decorator_(self.__svm_rbf_tuned_auroc_)
        return self.__svm_rbf_tuned_auroc_(C=C, logGamma=logGamma)
    
    def optimise_svc(self):
        self.__log.write("Optimising SVC model")
        self.__svm_tuned_auroc_ = self.__cv_decorator_(self.__svm_tuned_auroc_)
        optimal_svm_pars, info, _ = optunity.maximize_structured(self.__svm_tuned_auroc_, self.__space_, num_evals=150)
        print("Optimal parameters" + str(optimal_svm_pars))
        print("AUROC of tuned SVM: %1.3f" % info.optimum)

    def optimise_sgd(self):
        self.__log.write("Optimising SGD model")
        self.__sgd_tuned_auroc_ = self.__cv_decorator_(self.__sgd_tuned_auroc_)
        optimal_sgd_pars, info, _ = optunity.maximize_structured(self.__sgd_tuned_auroc_, self.__sgd_space_, num_evals=150)
        print("Optimal parameters" + str(optimal_sgd_pars))
        print("AUROC of tuned SVM: %1.3f" % info.optimum)

    def __train_model_(self, x_train, y_train, kernel, C, logGamma, degree, coef0):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C, cache_size=7000)
        elif kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, cache_size=7000)
        elif kernel == 'rbf':
            model = SVC(kernel=kernel, C=C, gamma=10 ** logGamma, cache_size=7000)
        else:
            raise ArgumentError("Unknown kernel function: %s" % kernel)
        model.fit(x_train, y_train)
        return model
    
    def __svm_rbf_tuned_auroc_(self, x_train, y_train, x_test, y_test, C, logGamma):
        model = SVC(C=C, gamma=10 ** logGamma, cache_size=7000).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
    def __svm_tuned_auroc_(self, x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
        model = self.__train_model_(x_train, y_train, kernel, C, logGamma, degree, coef0)
        decision_values = model.predict(x_test)
        return roc_auc_score(y_test, decision_values)
    
    def __sgd_tuned_auroc_(self, x_train, y_train, x_test, y_test, alpha=0.0001, power_t=0.5):
        model = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="l2", alpha=alpha, fit_intercept=True, n_jobs=-1, power_t=power_t).fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc
    
