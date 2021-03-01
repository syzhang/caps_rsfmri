import numpy as np
from scipy.optimize import minimize
# from DCC_GARCH.DCC.DCC_loss import Q_gen, Q_average

class DCC():

    def __init__(self, max_itr=2, early_stopping=True):
        self.max_itr = max_itr
        self.early_stopping = early_stopping
        self.ab = np.array([0.5, 0.5])
        self.method =  'SLSQP'
        def ub(x):
            return 1. - x[0] - x[1]
        def lb1(x):
            return x[0]
        def lb2(x):
            return x[1]
        self.constraints = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},{'type':'ineq', 'fun':lb2}]

    def set_ab(self,ab): # ndarray
        self.ab = ab

    def get_ab(self):
        return self.ab

    def set_method(self,method):
        self.method = method

    def set_loss(self, loss_func):
        #"loss function L is a meta-function, s.t. L(r) = f(theta)."
        self.loss_func = loss_func

    def get_loss_func(self):
        if self.loss_func is None:
            raise Exception("No Loss Function Found!")
        else:
            return self.loss_func

    def set_max_itr(self, max_itr):
        self.max_itr = max_itr

    def get_max_itr(self):
        return self.max_itr

    def fit(self, train_data):
        #train_data: numpy.array([[e1_T,...e1_0],\
        #                         [e2_T,...e2_0],\
        #                         ...,
        #                         [en_T,...en_0]])

        tr = train_data

        # Optimize using scipy and save theta
        tr_losses = []
        j = 0
        count = 0
        while j < self.get_max_itr():
            j += 1
            ab0 = np.array(self.get_ab())
            res = minimize(self.get_loss_func()(tr), ab0, method=self.method,
                           options={'disp': True},constraints=self.constraints)
            ab = res.x
            self.set_ab(ab)

            tr_loss = self.get_loss_func()(tr)(ab)
            tr_losses.append(tr_loss)
            print("Iteration: %d. Training loss: %.3E." % (j, tr_loss))

            # Early stopping
            if self.early_stopping is True:
                if j > 10:
                    if abs(tr_losses[-1] - tr_losses[-2]) / tr_losses[-2] < 0.0001:
                        count += 1
                        if count >= 2:
                            print("Early Stopping...")
                            return tr_losses

        return tr_losses


    def Q(self,y):
        Q = Q_gen(y,self.ab)
        return Q

    def Q_bar(self,y):
        return Q_average(y)



def Q_average(tr):
    # return average of outer product of [eT,...e0]
    # et = [r(1t)/s(1t),...r(nt)/s(nt)]
    T = tr.shape[1]
    n = tr.shape[0]
    sum = np.zeros([n,n])
    for i in range(T):
        sum += np.outer(tr[:,i],tr[:,i])
    return sum/T

def Q_gen(tr,ab):
    # generate [QT,...Q0]
    Q_int = Q_average(tr)
    Q_list = [Q_int]
    T = tr.shape[1] - 1
    a = ab[0]
    b = ab[1]
    for i in range(T):
        et_1 = tr[:,T-i]
        Qt_1 = Q_list[0]
        Qt = (1.0-a-b)*Q_int + a*np.outer(et_1,et_1) + b*Qt_1
        Q_list = [Qt] + Q_list
    return Q_list

def R_gen(tr,ab):
    Q = Q_gen(tr,ab)
    # output [RT,...R0]
    R_list = []
    n = Q[0].shape[0]
    for i in Q:
        temp = 1.0/np.sqrt(np.abs(i))
        temp = temp * np.eye(n)
        R = np.dot(np.dot(temp,i),temp)
        R_list = R_list + [R]
    return R_list

def dcc_loss(tr, ab):
    R = R_gen(tr,ab)

    def dcc_loss_helper(tr=tr,R=R):
        loss = 0.0
        for i in range(len(R)):
            Ri = R[i]
            Ri_ = np.linalg.inv(Ri)
            ei = tr[:,i]
            loss += np.log(np.linalg.det(Ri)) + np.dot(np.dot(ei,Ri_),ei)
        # print('training loss %f' % loss)
        return loss

    return dcc_loss_helper()


def dcc_loss_gen():
    def loss1(tr):
        def loss(ab):
            return dcc_loss(tr, ab)
        return loss
    return loss1