import numpy as np

class hawkes_process:

    def exponential_kernel(self, w, t):
        return w*np.exp(-w*t)

    def hawkes_intensity(self, mu, w, alpha, k, t, points):
        # mu is the constant base intensity of a particular generating label (y)
        # w is omega, the decay of the kernel function
        # alpha is a matrix of size |Y|×|Y| which encodes the degrees of influence between pairs of labels assigned to the tweets
        # TODO: Use alpha or not?
        # k is a kernel function, using exponential kernel for now
        # t is the considered time
        # points is the list of vectors (baseline: only timestamps)
        p = []
        for t_i in points:
            if t_i < t:
                p.append(self.exponential_kernel(w,t-t_i))
        return mu + sum(p)

    def generate_point(self):
        # generates a point according to hawkes intensity
        pass


    def simulate_hawkes(self):
        # thinning algorithm
        pass
