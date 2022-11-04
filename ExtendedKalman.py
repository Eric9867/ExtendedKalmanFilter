import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    '''
    Discrete time
    '''
    def __init__(self):
        self.h = 1 # time step
        self.H = 20 # height of lighthouse
        self.d = 200 # position of lighthouse
        self.kp = 500 # proportional gain
        self.set_point = np.arctan(1/10) # phi_f
        self.r = 10 #1 #2.5 # 10

        self.t_sim = 25 
        self.N = int(self.t_sim / self.h + 1) # number of time steps

        self.x_true = np.zeros((self.N), dtype=float) # state
        self.z_est = np.zeros((self.N), dtype=float) # state
        self.x = np.zeros((self.N), dtype=float) # state estimate
        self.u = np.zeros((self.N), dtype=float) # input
        self.z = np.zeros((self.N), dtype=float) # measmnt

        self.x_true[0] = 100 # x0
        self.x[0] = 100 # x0
        self.i = 0 # index of last data entry
        self.update_measmnt()
        self.z_est[0] = self.z[0]
        self.update_input()

        # EXTENDED KALMAN
        self.A = 1
        self.B = self.h # time step
        self.Q = np.ones((self.N), dtype=float) * 1/(3*self.r) 
        self.R = np.ones((self.N), dtype=float) * self.d**4/(3* self.H**2 * self.r**2) / 1e14 # consider constant 

        self.D = np.zeros((self.N), dtype=float) # from measmnt model
        self.P = np.ones((self.N), dtype=float) # state covariance
        self.S = np.ones((self.N), dtype=float) # measmnt covariance
        self.W = np.ones((self.N), dtype=float) # Kalman gain
        
        self.P[0] = self.Q[0]
        self.S[0] = self.R[0]
        self.D[0] = self.H / (self.H**2 + (self.d - self.x[self.i])**2)
        self.W[0] = self.P[0] * self.D[0] / self.S[0]


    def get_state(self):
        return self.x[self.i]

    def get_input(self):
        return self.u[self.i]

    def get_measmnt(self):
        return self.z_est[self.i]

    def get_state_noise(self):
        return np.random.uniform(-self.r, self.r)

    def get_measmnt_noise(self):
        w = (self.H/self.d**2) * self.r
        return np.random.uniform(-w, w)

    def predict_state(self): 
        self.x[self.i + 1] = self.get_state() + self.h * self.get_input() #+ self.get_state_noise()
        return

    def predict_measmnt(self):
        self.z_est[self.i+1] = np.arctan(self.H/(self.d - self.x[self.i + 1])) #+ self.get_measmnt_noise()

    def update_state_true(self):
        self.x_true[self.i + 1] = self.x_true[self.i] + self.h * self.get_input() + self.get_state_noise()
        return

    def update_state(self): # estimate
        prev = self.get_state()
        self.x[self.i] = self.get_state() + self.W[self.i] * (self.z[self.i] - self.z_est[self.i])
        if self.i < 40:
            # print(f"KALMAN GAIN: {self.W[self.i]} \t\t MEASMNT RESIDUAL: {self.z[self.i] - self.z_est[self.i]}")
            print(f"state: {self.x[self.i]} \t\t ground truth: {self.x_true[self.i]}")
        return

    def update_measmnt(self):
        self.z[self.i] = np.arctan(self.H/(self.d - self.x_true[self.i])) + self.get_measmnt_noise()
        return

    def update_input(self):
        self.u[self.i] = -self.kp * (self.get_measmnt() - self.set_point)
        return

    def update_D(self):
        self.D[self.i+1] = self.H / (self.H**2 + (self.d - self.x[self.i + 1])**2) # based on prediction

    def predict_cov(self):
        k = self.i
        self.P[k+1] = self.A * self.P[k] * self.A + self.Q[k+1]
        self.update_D()
        self.S[k+1] = self.D[k+1] * self.P[k+1] * self.D[k+1] + self.R[k+1]
        self.W[k+1] = self.P[k+1] * self.D[k+1] / self.S[k+1] * 1.5
        return

    def update_cov(self):
        k = self.i
        self.P[k] = self.P[k] - self.W[k] *self.S[k] * self.W[k]


    def step(self):
        self.predict_state()
        self.predict_measmnt()
        self.predict_cov()

        self.update_state_true()
        self.i += 1
        self.update_measmnt()
        self.update_state()
        self.update_input()

    def solve(self):
        while(self.i < self.N - 1):
            self.step()

    def plot(self):
        t = np.linspace(0, self.t_sim, self.N)

        title1 = 'State $x$'
        title2 = 'Input $u$'
        title3 = 'Measurement $z = \phi$'

        figure, axis = plt.subplots(2, 2)
        
        # axis[0, 0].plot(t, Y1, t, self.x_true)
        # plt.show()

        axis[0, 0].plot(t, self.x, 'r')
        axis[0, 0].plot(t, self.x_true, 'k:')
        axis[0, 0].legend(['$\hat x$', 'x'])
        axis[0, 0].set_title(title1)
        
        axis[0, 1].plot(t, self.u, 'b')
        axis[1, 0].legend(['$\hat z$', 'z'])
        axis[0, 1].set_title(title2)
        
        axis[1, 0].plot(t, self.z, 'r')
        axis[1, 0].plot(t, self.z_est, 'k')
        axis[1, 0].legend(['$\hat z$', 'z'])
        axis[1, 0].plot( \
            (0,self.t_sim), (self.set_point,self.set_point), ':')
        axis[1, 0].set_title(title3)
        plt.show()



if __name__ == '__main__':
    model = KalmanFilter()
    model.solve()
    model.plot()
    print()