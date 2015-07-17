import optunity
import optunity.metrics

# comment this line if you are running the notebook
import sklearn.svm
import numpy as np

decision_values = np.array([-0.69354811, -0.69354743, -0.69354744, -0.69354754, -0.69354715, -0.69354866, -0.69354775, -0.69355032, -0.69355325])
y_test = [0, 0, 0, 0, 0, 0, 0, 0, 0]

auc = optunity.metrics.roc_auc(y_test, decision_values)
