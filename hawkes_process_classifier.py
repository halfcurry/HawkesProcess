import numpy as np
from random import randint
from scipy.optimize import minimize, check_grad
import ast, math

class hawkes_process_classifier:

    def neg_log_likelihood(self, param, *args):
        # print time_stamps
        # time_stamps=time_stamps[1]
        if args[0] == 'mu':
            mu = param
            print("Mu: ", mu)

            alpha = args[1]
            beta = args[2]
            time_stamps = args[3]
        elif args[0] == 'alpha':

            alpha = param
            print("Alpha: ", alpha)

            mu = args[1]
            beta = args[2]
            time_stamps = args[3]

        elif args[0] == 'beta':

            beta = param
            print("Beta: ", beta)

            mu = args[1]
            alpha = args[2]
            time_stamps = args[3]

        elif args[0] == 'all':
            mu = param[0]
            alpha = param[1]
            beta = param[2]

            time_stamps = args[1]

        # time_diff = time_stamps[-1] - time_stamps
        # time_exp = np.exp(-beta * time_diff) - 1
        # first_sum = (alpha * 1.0 / max(1e-10, beta) )* np.sum(time_exp)

        # print time_stamps

        R = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            R[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + R[i - 1])

        temp = np.sum(np.log(mu + alpha * R))

        temp1 = (alpha/beta) * np.sum(np.exp(-beta*(time_stamps[-1]-time_stamps)) -1)

        ll = (-mu * time_stamps[-1]) + temp + temp1

        # print("Negative likelihood: ", -ll)

        return -ll


    def gradient_mu(self, param, *args):

        # mu = np.exp(param)
        mu = param

        alpha = args[1]
        beta = args[2]
        time_stamps = args[3]

        r = np.zeros(time_stamps.shape)

        # print time_stamps.shape

        # for i in range(1, len(time_stamps)):
        #     r[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + r[i - 1])
        #
        # # first_sum = np.sum(1.0/(mu + alpha * r))
        #
        # first_sum = 0
        # for elt in r:
        #     first_sum=first_sum+(1.0/max(1e-10,(mu+alpha*elt)))
        #
        # del_mu = (-time_stamps[-1] + first_sum)

        A = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            A[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + A[i - 1])

        temp = (mu + alpha * A)
        del_mu = (-time_stamps[-1] + np.sum(1.0 / temp))

        return -del_mu

    def grad_mu(self, param, *args):

        mu = np.exp(param)

        gr_mu = self.gradient_mu(mu, *args)
        gr_mu = gr_mu * mu

        print("Negative of gradient_mu: ", gr_mu)

        return gr_mu

    def gradient_alpha(self, param, *args):

        # alpha = np.exp(param)
        alpha = param

        mu = args[1]
        beta = args[2]
        time_stamps = args[3]

        # time_diff = time_stamps[-1] - time_stamps
        # time_exp = np.exp(-beta * time_diff) - 1
        # first_sum = (alpha * 1.0 / max(1e-10, beta)) * np.sum(time_exp)
        #
        # r = np.zeros(time_stamps.shape)
        #
        # # print time_stamps.shape
        #
        # for i in range(1, len(time_stamps)):
        #     r[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + r[i - 1])
        #
        # second_sum = 0
        # for elt in r:
        #     second_sum = second_sum + ( elt/ max(1e-10,(mu + alpha * elt)) )
        #
        # del_alpha = (second_sum + ((1.0/beta) * first_sum))

        A = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            A[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + A[i - 1])

        temp = (mu + alpha * A)

        del_alpha = np.sum(A / temp) + math.pow(beta, -1) * np.sum(np.exp(-beta * (time_stamps[-1] - time_stamps)) - 1)

        return -del_alpha

    def grad_alpha(self, param, *args):

        alpha = np.exp(param)

        gr_alpha = self.gradient_alpha(alpha, *args)
        gr_alpha = gr_alpha * alpha

        print("Negative of gradient_alpha: ", gr_alpha)

        return gr_alpha

    def gradient_beta(self, param, *args):

        beta = param
        mu = args[1]
        alpha = args[2]
        time_stamps = args[3]

        # time_diff = time_stamps[-1] - time_stamps
        # time_exp = np.exp(-beta * time_diff) - 1

        # a = np.zeros(time_stamps.shape)
        # b = np.zeros(time_stamps.shape)
        #
        # for i in range(1, len(time_stamps)):
        #     a[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + a[i - 1])
        #
        # for i in range(1, len(time_stamps)):
        #
        #     for j in range(1,i-1):
        #
        #         b[i]=b[i]+(time_stamps[i]-time_stamps[j])*(np.exp(-beta*(time_stamps[i]-time_stamps[j])))
        #
        #
        # first_sum=0
        # second_sum=0
        #
        # for i in range(len(time_stamps)):
        #
        #     first_sum=first_sum+(alpha*b[i]/max((mu+alpha*a[i]),1e-10))
        #
        #     second_sum=second_sum+ (1.0/beta)*(time_stamps[-1]-time_stamps[i])*np.exp(-beta*(time_stamps[-1]-time_stamps[i]))+(1.0/(beta*beta))*np.exp(-beta*(time_stamps[-1]-time_stamps[i]))

        # del_beta = (-first_sum-alpha*second_sum)

        A = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            A[i] = np.exp(-beta * (time_stamps[i] - time_stamps[i - 1])) * (1 + A[i - 1])

        temp = (mu + alpha * A)

        B = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            tmp = time_stamps[i] - time_stamps[:i]
            tmp1 = np.exp(-beta * tmp)
            B[i] = np.dot(tmp, tmp1)

        B_temp = alpha * B
        temp2 = B_temp / temp

        first_term = math.pow(beta, -1) * (time_stamps[-1] - time_stamps) * np.exp(
            -beta * (time_stamps[-1] - time_stamps))
        second_term = math.pow(beta, -2) * np.exp(-beta * (time_stamps[-1] - time_stamps))

        del_beta = -np.sum(alpha * (first_term + second_term)) - np.sum(temp2)

        return -del_beta

    def grad_beta(self, param, *args):

        beta = np.exp(param)

        gr_beta = self.gradient_beta(beta, *args)
        gr_beta = gr_beta * beta

        print("Negative of gradient_beta: ", gr_beta)

        return gr_beta

    def gradient_combined(self, param, *args):

        mu = param[0]
        alpha = param[1]
        beta = param[2]

        time_stamps = args[1]

        # for del_mu

        A = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            A[i] = np.exp(-beta*(time_stamps[i]-time_stamps[i-1]))*(1+A[i-1])

        temp = (mu + alpha*A)
        del_mu = (-time_stamps[-1] + np.sum(1.0/temp))

        # for del_alpha

        del_alpha = np.sum(A/temp) + math.pow(beta,-1) * np.sum(np.exp(-beta*(time_stamps[-1]-time_stamps))-1)

        # for del_beta

        B = np.zeros(time_stamps.shape)

        for i in range(1, len(time_stamps)):
            tmp = time_stamps[i] - time_stamps[:i]
            tmp1 = np.exp(-beta * tmp)
            B[i] = np.dot(tmp, tmp1)

        B_temp = alpha*B
        temp2 = B_temp/temp

        first_term = math.pow(beta,-1) * (time_stamps[-1]-time_stamps) * np.exp(-beta*(time_stamps[-1]-time_stamps))
        second_term = math.pow(beta,-2) * np.exp(-beta*(time_stamps[-1]-time_stamps))

        del_beta = -np.sum(alpha*(first_term+second_term)) - np.sum(temp2)

        return np.array([-del_mu, -del_alpha, -del_beta])


    def check_grad_mu(self, time_stamps):
        beta = 0.01
        alpha = 0.1

        args = ('mu', alpha, beta, time_stamps)
        # res = check_grad(self.neg_log_likelihood, self.gradient_mu, [], args = args)
        #
        for i in range(100):
            mu = i/10

            print("Mu: ", mu, "Err: ", check_grad(self.neg_log_likelihood, self.grad_mu, np.array([np.log(mu)]), *args))

    def check_grad_alpha(self, time_stamps):
        beta = 0.01
        mu = 0.1

        args = ('alpha', mu, beta, time_stamps)
        # res = check_grad(self.neg_log_likelihood, self.gradient_mu, [], args = args)
        #
        for i in range(100):
            alpha = i / 10

            print("Alpha: ", alpha, "Err: ", check_grad(self.neg_log_likelihood, self.grad_alpha, np.array([alpha]), *args))


    def estimate_params(self, time_stamps):

        res1 = res2 = res3 = res4 = []

        ###################################################

        # mu0 = 0.1
        # beta0 = 0.01
        # alpha0 = 0.1
        #
        # x = [mu0]
        #
        # args = ('mu', alpha0, beta0, time_stamps)
        #
        # n_ll = lambda x: self.neg_log_likelihood(x, *args)
        # n_grad = lambda x: self.gradient_mu(x, *args)
        #
        # cons = ({'type': 'ineq', 'fun': lambda x: np.array([x[0] - 1e-5])})
        #
        # print("Estimating mu...")
        # res1 = minimize(n_ll, x, args=args,
        #                 method='SLSQP',
        #                 bounds=((1e-5, None)),
        #                 constraints=cons,
        #                 options={'disp': False, 'maxiter': 200},
        #                 # jac=n_grad
        #                 )
        #
        #
        # ##########################################################
        #
        # alpha0 = 0.1
        # beta0 = 0.01
        # mu0 = 0.1
        #
        # x = [alpha0]
        #
        # args = ('alpha', mu0, beta0, time_stamps)
        #
        # n_ll = lambda x: self.neg_log_likelihood(x, *args)
        # n_grad = lambda x: self.gradient_alpha(x, *args)
        #
        # cons = ({'type': 'ineq', 'fun': lambda x: np.array([beta0 - alpha0 - 1e-5])})
        #
        # print("Estimating alpha...")
        # res2 = minimize(n_ll, x, args=args, method='SLSQP', bounds=((1e-5, None)), constraints=cons, options={'disp': False, 'maxiter': 500},)
        #                 # jac=n_grad)
        #
        # ##########################################################
        #
        # mu0 = 0.1
        # alpha0 = 0.1
        # beta0 = 0.01
        #
        # x =  [beta0]
        #
        # args = ('beta', mu0, alpha0, time_stamps)
        #
        # n_ll = lambda x: self.neg_log_likelihood(x, *args)
        # n_grad = lambda x: self.gradient_beta(x, *args)
        #
        # cons = ({'type': 'ineq', 'fun': lambda x: np.array([beta0 - alpha0 - 1e-5])})
        #
        # print("Estimating beta...")
        #
        # res3 = minimize(n_ll, x, args=args, method='SLSQP', bounds=((1e-5, None)), constraints=cons, options={'disp': False, 'maxiter': 500},)
        #                 # jac=n_grad)

        #########################################################

        mu0 = 0.1
        alpha0 = 0.5
        beta0 = 0.01

        x = [mu0, alpha0, beta0]

        args = ('all', time_stamps)

        n_ll = lambda x: self.neg_log_likelihood(x, *args)
        n_grad = lambda x: self.gradient_combined(x, *args)

        cons = ({'type': 'ineq', 'fun': lambda x: np.array([beta0 - alpha0 - 1e-5])})

        print("Estimating params...")

        res4 = minimize(n_ll, x, method='SLSQP', bounds=((1e-5, None), (1e-5, None), (1e-5, None)), constraints=cons, options={'disp': False, 'maxiter': 200},
                        jac = n_grad
                        )

        return res1, res2, res3, res4

    def main(self):


        with open('sample_timestamps.txt', 'r') as f:
            test_timestamps = ast.literal_eval(f.read())

        test_timestamps = np.array(test_timestamps)

        # self.check_grad_alpha(test_timestamps)
        # self.check_grad_mu(test_timestamps)

        res1, res2, res3, res4 = self.estimate_params(test_timestamps)

        print(res1)

        print("\n\n", res2)

        print("\n\n", res3)

        print("\n\n", res4)

        # print("\n\nEstimated mu: ", res1.x)
        # print("Estimated alpha: ", res2.x)
        # print("Estimated beta: ", res3.x)
        print("Estimated all: ", res4.x)


    def fit(self):
        pass

    def predict(self):
        pass



if __name__ == '__main__':
    hawkes_process_classifier().main()
