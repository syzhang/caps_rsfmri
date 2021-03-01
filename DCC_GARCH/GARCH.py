import numpy as np
from scipy.optimize import minimize
# from lib import *
# from DCC_GARCH.GARCH.GARCH_loss import garch_process


class GARCH():

    def __init__(self, p=1, q=1, max_itr=3, early_stopping=True):
        # p the lag of r_t, q the lag of s_t
        self.p = p
        self.q = q
        theta0 = [0.005] + [0.1 for i in range(p)] + [0.1 for i in range(p)] + [0.85 for i in range(q)]
        self.theta = np.array(theta0)
        self.max_itr = max_itr
        self.early_stopping = early_stopping
        def ub(x):
            return 1. - x[1] - 0.5*x[2] - x[3]
        def lb1(x):
            return x[1] + x[2]
        def lb2(x):
            return x[0]
        def lb3(x):
            return x[1]
        def lb4(x):
            return x[3]
        self.constraints = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},
                            {'type':'ineq', 'fun':lb2},{'type':'ineq', 'fun':lb3},
                            {'type':'ineq', 'fun':lb4}]
        self.method = 'COBYLA'

    def set_theta(self, theta):
        self.theta = np.array(theta)

    def get_theta(self):
        return self.theta

    def get_p(self):
        return self.p

    def get_q(self):
        return self.q

    def set_loss(self, loss_func):
        "loss function L is a meta-function, s.t. L(r) = f(theta)."
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

    def set_method(self, method):
        self.method = method

    def get_method(self):
        return self.method

    def fit(self, train_data):  # train_data: [rT,...r0]
        tr = train_data

        # Optimize using scipy and save theta
        tr_losses = []
        j = 0
        count = 0
        while j < self.get_max_itr():
            j += 1
            theta0 = self.get_theta()
            res = minimize(self.get_loss_func()(tr), theta0, method=self.method,
                           options={'disp': True}, constraints=self.constraints)
            theta = res.x
            self.set_theta(theta)

            tr_loss = self.get_loss_func()(tr)(theta)
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

    def sigma(self, y):  # test data: [rT,...r0]
        s = garch_process(y, self.theta, self.p, self.q)
        return np.array(s)

def garch_process(r, theta, p=1, q=1):
    w, alpha, gamma, beta = theta[0], theta[1:1 + p], theta[1 + p:1 + p + p], theta[1 + p + p:]
    if len(gamma) is not q:
        raise Exception('Parameter Length Incorrect!')
    r = np.array(r)
    T = len(r) - 1

    def garch_update(s, r, t, alpha, beta, gamma, p=p, q=q, T=T):
        "s = [st-1,...s0], r = [rT,...,r0], t is time" \
        "alpha, beta and gamma are from above" \
        "returns new_s = [st,...,s0]"
        r_temp = r[T - t + 1:T - t + 1 + q]  # [rt-1,...,rt-q]
        s_temp = s[0:p]  # [st-1,...st-p]

        var = np.array(s_temp) ** 2
        r_squared = np.array(r_temp) ** 2
        gjr = r_squared*(np.array(r_temp)<0)
        st = np.sqrt(np.abs(np.dot(np.array(beta), var) + np.dot(np.array(alpha), r_squared)
                      + np.dot(np.array(gamma), gjr) + w))

        new_s = [st] + s

        return new_s #[sT,...,s0]

    #"Initialize values of s and m as data variance and mean"
    s_int = np.std(r)
    L = max(p, q)
    s = [s_int for i in range(0, L)]

    for t in range(L, T + 1):
        s = garch_update(s, r, t, alpha, beta, gamma)

    return s

def garch_loss(r, theta, p, q):
    s = garch_process(r, theta, p, q)

    def garch_loss_helper(r=r, s=s):
        s = np.array(s)
        loss = 0.0
        for i in range(len(r)):
            loss += np.log(s[i] ** 2) + (r[i]/s[i])**2

    #print('training loss %f' % loss)
        return loss

    return garch_loss_helper()


def garch_loss_gen(p=1, q=1):
    def loss1(r):
        def loss(theta):
            return garch_loss(r, theta, p, q)
        return loss
    return loss1