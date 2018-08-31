import numpy as np
from random import randint

class hawkes_process_simulation:

    def exponential_kernel(self, beta, t):
        return beta*np.exp(-beta*t)

    def hawkes_intensity(self, mu, beta, alpha, k, t, points):
        # mu is the constant base intensity of a particular generating label (y)
        # beta is the decay of the kernel function
        # alpha is adjacency matrix (can be scalar too)
        # neta? is a matrix of size |Y|×|Y| which encodes the degrees of influence between pairs of labels assigned to the tweets
        # TODO: Use alpha or not?
        # k is a kernel function, using exponential kernel for now
        # t is the considered time
        # points is the list of vectors (baseline: only timestamps)
        p = []
        for t_i in points:
            if t_i < t:
                # p.append(self.exponential_kernel(beta,t-t_i))
                p.append(k(beta, t - t_i))
        return mu + alpha * sum(p)

    def generate_point(self, mu, beta, alpha, k, t, points):
        # generates a point according to hawkes intensity, i.e does simulation of a single point by thinning

        # intensity value at current t : set upper bound of poisson intensity
        lambd = self.hawkes_intensity(mu, beta, alpha, k, t, points)

        # generate time lag from homogeneous exp distribution with this intensity
        s = np.random.exponential(scale = 1/lambd)

        #intensity with new t = t+s
        lambd_new = self.hawkes_intensity(mu, beta, alpha, k, t + s, points)

        ratio = lambd_new/lambd

        # randomly accept/ reject based on if the ratio is lesser than a uniform random number
        if ratio >= np.random.uniform():
            #update the current time
            t = t + s

        return (t,lambd)


    def simulate_hawkes_window(self, mu, beta, alpha, k, time_window, n_users):
        # thinning algorithm
        # suppose we need to simulate any number of events in (0-time_window) for n_users
        t = 0
        user_points = {}
        for user in range(0, n_users):
            print(user)
            user_points[user] = []
        all_samples = []
        prev_t = t
        while t < time_window:
            prev_t = t
            user = randint(0, n_users-1)
            t,lambd = self.generate_point(mu, beta, alpha, k, t, user_points[user])
            if prev_t != t:
                user_points[user].append(t)
                all_samples.append(t)

        return user_points, all_samples

    def simulate_hawkes_events(self, mu, beta, alpha, k, N, n_users):
        #suppose we need to simulate N events in total distributed among n_users
        t = 0
        user_points = {}
        for user in range(0, n_users):
            print(user)
            user_points[user] = []
        all_samples = []
        prev_t = t
        i = len(all_samples)
        while i <= N:
            prev_t = t
            user = randint(0, n_users-1)
            t, lambd = self.generate_point(mu, beta, alpha, k, t, user_points[user])
            if prev_t != t:
                user_points[user].append(t)
                all_samples.append(t)
            i = len(all_samples)

        return user_points, all_samples

    def main(self):
        print('Simulating Hawkes for 300s...')
        user_points, all_samples = self.simulate_hawkes_window(0.5, 0.01, 0.9, self.exponential_kernel, 300, 1)

        with open('sample_timestamps.txt', 'w') as f:
            f.write(str(all_samples))

        # print('Simulating Hawkes for 1000 events...')
        # self.simulate_hawkes_events(0.1, 0.01, 0.1, self.exponential_kernel, 1000, 1)

if __name__ == '__main__':
    hawkes_process_simulation().main()



