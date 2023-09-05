import numpy as np

# treat Z as a vector for everything vector related
def tanh(Z):
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))

def softmax(Z):
    preds = []
    sum = 0
    for i in Z:
        sum += np.exp(i)
    for i in Z:
        preds.append(np.exp(i)/sum)
    return np.array(preds)

def del_S_del_z(Z, preds):
    jacobian = np.zeros((Z.shape[0], Z.shape[0], 1))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[0]):
            if(i == j):
                jacobian[i][j][0] = preds[i] * (1 - preds[j])
            else:
                jacobian[i][j][0] = -preds[j] * preds[i]
    return jacobian

