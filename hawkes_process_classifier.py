import numpy as np
from random import randint
from scipy.optimize import minimize

class hawkes_process_classifier:

    def neg_log_likelihood(self, param, *triplet):
        # print time_stamps
        # time_stamps=time_stamps[1]

        mu = param
        alpha = triplet[0]
        w = triplet[1]
        time_stamps = triplet[2]

        time_diff = time_stamps[-1] - time_stamps
        time_exp = np.exp(-w * time_diff) - 1
        first_sum = alpha * 1.0 / w * np.sum(time_exp)

        # print time_stamps

        r = np.zeros(time_stamps.shape)

        # print time_stamps.shape

        for time_ctr in range(1, len(time_stamps)):
            r[time_ctr] = np.exp(-w * (time_stamps[time_ctr] - time_stamps[time_ctr - 1])) * (1 + r[time_ctr - 1])

        second_sum = np.sum(np.log(mu + alpha * r))

        ll = -(-mu * time_stamps[-1] + first_sum + second_sum)

        return -ll

    def estimate_params(self, time_stamps):
        mu0 = 0.1
        w = 0.1
        alpha = 0.1

        initial_guess = mu0

        triplet = (alpha, w, time_stamps)

        res = minimize(self.neg_log_likelihood, initial_guess, args = triplet, method = 'L-BFGS-B')

        return res

    def main(self):
        print(self.estimate_params(np.array([2.8914986, 8.09118015, 10.35220284, 15.72422933,
                                  20.41385768, 24.76074457, 58.6856862, 59.66601112,
                                  86.23050124, 95.78644578, 97.79439315, 109.11365526,
                                  115.38968376, 115.99595348, 138.26479343, 156.01784925,
                                  178.28689279, 181.86195694, 193.8846334, 199.726623,
                                  206.28067149, 213.24882906, 225.44679125, 238.64851829,
                                  246.26655032, 250.73418687, 255.75212692, 273.53467842])
                              )
              )

    def fit(self):
        pass

    def predict(self):
        pass



if __name__ == '__main__':
    hawkes_process_classifier().main()
