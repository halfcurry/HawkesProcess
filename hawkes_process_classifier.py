import numpy as np
from random import randint
from scipy.optimize import minimize, check_grad

class hawkes_process_classifier:

    def neg_log_likelihood(self, param, *args):
        # print time_stamps
        # time_stamps=time_stamps[1]
        if args[0] == 'mu':
            mu = np.exp(param)
            print("Mu: ", mu)

            alpha = args[1]
            beta = args[2]
        elif args[0] == 'alpha':

            alpha = np.exp(param)
            print("Alpha: ", alpha)

            mu = args[1]
            beta = args[2]

        elif args[0] == 'beta':

            beta = np.exp(param)
            print("Beta: ", beta)

            mu = args[1]
            alpha = args[2]

        time_stamps = args[3]

        time_diff = time_stamps[-1] - time_stamps
        time_exp = np.exp(-beta * time_diff) - 1
        first_sum = (alpha * 1.0 / max(1e-10, beta) )* np.sum(time_exp)

        # print time_stamps

        r = np.zeros(time_stamps.shape)

        # print time_stamps.shape

        for time_ctr in range(1, len(time_stamps)):
            r[time_ctr] = np.exp(-beta * (time_stamps[time_ctr] - time_stamps[time_ctr - 1])) * (1 + r[time_ctr - 1])

        second_sum = np.sum(np.log(mu + alpha * r))

        ll = -(-mu * time_stamps[-1] + first_sum + second_sum)

        print("Negative likelihood: ", ll)

        return ll


    def gradient_mu(self, param, *args):

        mu = np.exp(param)

        alpha = args[1]
        beta = args[2]
        time_stamps = args[3]

        r = np.zeros(time_stamps.shape)

        # print time_stamps.shape

        for time_ctr in range(1, len(time_stamps)):
            r[time_ctr] = np.exp(-beta * (time_stamps[time_ctr] - time_stamps[time_ctr - 1])) * (1 + r[time_ctr - 1])

        # first_sum = np.sum(1.0/(mu + alpha * r))

        first_sum = 0
        for elt in r:
            first_sum=first_sum+(1.0/max(1e-10,(mu+alpha*elt)))

        del_mu = (-time_stamps[-1] + first_sum)*np.exp(param)

        print("Negative of gradient_mu: ", -del_mu)

        return -del_mu

    def gradient_alpha(self, param, *args):

        alpha = np.exp(param)
        mu = args[1]
        beta = args[2]
        time_stamps = args[3]

        time_diff = time_stamps[-1] - time_stamps
        time_exp = np.exp(-beta * time_diff) - 1
        first_sum = (alpha * 1.0 / max(1e-10, beta)) * np.sum(time_exp)

        r = np.zeros(time_stamps.shape)

        # print time_stamps.shape

        for time_ctr in range(1, len(time_stamps)):
            r[time_ctr] = np.exp(-beta * (time_stamps[time_ctr] - time_stamps[time_ctr - 1])) * (1 + r[time_ctr - 1])

        second_sum = 0
        for elt in r:
            second_sum = second_sum + ( elt/ max(1e-10,(mu + alpha * elt)) )

        del_alpha = (second_sum + ((1.0/beta) * first_sum))*np.exp(param)

        print("Negative of gradient_alpha: ", -del_alpha)
        return -del_alpha

    def gradient_beta(self, param, *args):

        beta = np.exp(param)
        mu = args[1]
        alpha = args[2]
        time_stamps = args[3]

        time_diff = time_stamps[-1] - time_stamps
        time_exp = np.exp(-beta * time_diff) - 1

        a = np.zeros(time_stamps.shape)

        b = np.zeros(time_stamps.shape)

        for time_ctr in range(1, len(time_stamps)):
            a[time_ctr] = np.exp(-beta * (time_stamps[time_ctr] - time_stamps[time_ctr - 1])) * (1 + a[time_ctr - 1])

        for i in range(1, len(time_stamps)):

            for j in range(1,i-1):

                b[i]=b[i]+(time_stamps[i]-time_stamps[j])*(np.exp(-beta*(time_stamps[i]-time_stamps[j])))


        first_sum=0
        second_sum=0

        for i in range(len(time_stamps)):

            first_sum=first_sum+(alpha*b[i]/max((mu+alpha*a[i]),1e-10))

            second_sum=second_sum+ (1.0/beta)*(time_stamps[-1]-time_stamps[i])*np.exp(-beta*(time_stamps[-1]-time_stamps[i]))+(1.0/(beta*beta))*np.exp(-beta*(time_stamps[-1]-time_stamps[i]))


        del_beta = (-first_sum-alpha*second_sum)*np.exp(param)
        print("Negative of gradient_beta: ", -del_beta)

        return -del_beta

    def check_grad_mu(self, time_stamps):
        beta = 0.1
        alpha = 0.1

        args = (alpha, beta, time_stamps)
        # res = check_grad(self.neg_log_likelihood, self.gradient_mu, [], args = args)
        #
        for i in range(100):
            mu = i/10

            print("Mu: ", mu, "Err: ", check_grad(self.neg_log_likelihood, self.gradient_mu, np.array([mu]), *args))

    def estimate_params(self, time_stamps):
        mu0 = np.log(0.1)
        beta = 0.01
        alpha = 0.1

        initial_guess = mu0

        args = ('mu', alpha, beta, time_stamps)

        print("Estimating mu...")
        res1 = minimize(self.neg_log_likelihood, initial_guess, args = args, method = 'BFGS', options = {'maxiter':100},
                       jac = self.gradient_mu)

        alpha0 = np.log(0.1)
        beta = 0.01
        mu = 0.1

        initial_guess = alpha0

        args = ('alpha', mu, beta, time_stamps)

        print("Estimating alpha...")
        res2 = minimize(self.neg_log_likelihood, initial_guess, args = args, method = 'BFGS', options = {'maxiter':100},
                       jac = self.gradient_alpha)

        alpha = 0.1
        beta0 = np.log(10)
        mu = 0.1

        initial_guess = beta0

        args = ('beta', mu, alpha, time_stamps)

        print("Estimating beta...")
        res3 = minimize(self.neg_log_likelihood, initial_guess, args=args, method='L-BFGS-B', options={'maxiter': 100},
                        jac=self.gradient_beta)

        return res1, res2, res3

    def main(self):

        test_timestamps = np.array([2.8914986, 8.09118015, 10.35220284, 15.72422933,
                                  20.41385768, 24.76074457, 58.6856862, 59.66601112,
                                  86.23050124, 95.78644578, 97.79439315, 109.11365526,
                                  115.38968376, 115.99595348, 138.26479343, 156.01784925,
                                  178.28689279, 181.86195694, 193.8846334, 199.726623,
                                  206.28067149, 213.24882906, 225.44679125, 238.64851829,
                                  246.26655032, 250.73418687, 255.75212692, 273.53467842])

        res1, res2, res3 = self.estimate_params(test_timestamps)

        print(res1)

        # self.check_grad_mu(np.array([2.8914986, 8.09118015, 10.35220284, 15.72422933,
        #                           20.41385768, 24.76074457, 58.6856862, 59.66601112,
        #                           86.23050124, 95.78644578, 97.79439315, 109.11365526,
        #                           115.38968376, 115.99595348, 138.26479343, 156.01784925,
        #                           178.28689279, 181.86195694, 193.8846334, 199.726623,
        #                           206.28067149, 213.24882906, 225.44679125, 238.64851829,
        #                           246.26655032, 250.73418687, 255.75212692, 273.53467842])
        #                       )

        print("\n\n", res2)

        print("\n\n", res3)

        print("\n\nEstimated mu: ", np.exp(res1.x))
        print("Estimated alpha: ", np.exp(res2.x))
        print("Estimated beta: ", np.exp(res3.x))


    def fit(self):
        pass

    def predict(self):
        pass



if __name__ == '__main__':
    hawkes_process_classifier().main()
