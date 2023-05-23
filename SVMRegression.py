import numpy as np
from sklearn.svm import LinearSVR, SVR

np.random.seed(42)

X = 2*np.random.random((100, 1))
y = 3*X+4+np.random.randn(100, 1)

svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X, y.ravel())

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y.ravel())
