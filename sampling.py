"""
This code snippet implements a basic sampling operation from the FSDM algorithm.
"""

# %% imports
import numpy as np
import torch

# %% sampler
class HeunSampler:
    """
    Implements a sampler using Heun's second order method.
    """	
    def __init__(self, nr_categories: int, sigma_min: float = 0.0001, sigma_max: float = 40, rho: float = 7, nr_steps: int = 32):
        self.nr_categories = nr_categories # dimension of the target variable
        self.sigma_min     = sigma_min   # minimum sigma of the diffusion process
        self.sigma_max     = sigma_max   # maximum sigma of the diffusion process
        self.rho           = rho         # rho parameter, steepness of the time steps
        self.nr_steps      = nr_steps    # number of steps to take, more steps means more accurate but also more expensive

    def sample(self, N: int, score_function: callable, X: list[torch.tensor] = None):
        """
        Sample from the score function using Heun's second order method. N is the number of samples to take.
        """
        # create an initial sample
        y = torch.randn(N, self.nr_categories) * self.sigma_max 

        # create empty tensors to store results
        y_over_time = torch.zeros(self.nr_steps+1, N, self.nr_categories)
        y_over_time[0] = y

        # loop over time steps
        for i in range(self.nr_steps):
            # get current sample
            y_i = y_over_time[i]

            # time and sigmas
            t_i        = self.get_t(i)
            t_i_p1     = self.get_t(i+1)
            sigma_i    = self.get_sigma(t_i)
            sigma_i_p1 = self.get_sigma(t_i_p1)

            # we need to expand them to the same shape as y
            t_i        = self.expand_scalar_to_tensor(t_i, y)
            t_i_p1     = self.expand_scalar_to_tensor(t_i_p1, y)
            sigma_i    = self.expand_scalar_to_tensor(sigma_i, y)
            sigma_i_p1 = self.expand_scalar_to_tensor(sigma_i_p1, y)

            # get the score at the current time step
            score = score_function(y_i, sigma_i, X)

            # update rule for next i
            d_i = - score * sigma_i
            y_i_p1 = y_i +  (t_i_p1 - t_i) * d_i

             # apply second order correction if sigma at ti_p1 is not 0 (this is the case for the last sample)
            if (i+1) != self.nr_steps:
                # get the score at the next time step
                score_p1 = score_function(y_i_p1, sigma_i_p1, X)

                # trapezoide rule
                d_i_p1 = score_p1 * -sigma_i_p1
                y_i_p1 = y_i + 0.5 * (t_i_p1 - t_i) * (d_i + d_i_p1)

            # store sample
            y_over_time[i+1] = y_i_p1.detach()

        return y_over_time
    
    # helper functions
    def get_t(self, i):
        """
        get time step t_i for i in {0,...,N-1}
        note that when sampling i increases monotonically, but the time t decreases monotonically
        Addition: if i = self.nr_Steps, then we should return an exact 0
        """
        if i == self.nr_steps:
            return 0.0
        else:
            return (self.sigma_max ** (1/self.rho) + i/(self.nr_steps-1) * (self.sigma_min ** (1/self.rho) - self.sigma_max ** (1/self.rho))) ** self.rho
    
    def get_sigma(self,t):
        """get sigma for time step t, which in our case is the same as the time step"""
        return t
    
    def expand_scalar_to_tensor(self, scalar, y):
        """
        Expand a scalar to the same shape as y
        """
        return torch.tensor(scalar).expand_as(y)


# %% example to run it
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from score_estimator import ScoreToCategoricalDistribution
    from FSDM_algorithm import FSDM

    # params
    sigma_min = 1e-4 # minimum sigma of the diffusion process
    sigma_max = 40   # maximum sigma of the diffusion process 
    rho       = 7    # rho parameter, steepness of the time steps
    nr_steps  = 32   # number of steps to take, more steps means more accurate but also more expensive
    N = 100          # batch size
    nr_trials = 256  # number of trials to take, repeating the sampling process to get the majority class, more trials means more accurate but also more expensive

    # create the models
    score_prior        = ScoreToCategoricalDistribution(nr_categories=4, x_influences=None)
    score_likelihood_1 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([ 1, 1,-1,-1]))
    score_likelihood_2 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([-1, 1,-1, 1]))
    fdsm = FSDM(score_prior, [score_likelihood_1, score_likelihood_2], [score_prior, score_prior])

    # create the sampler
    sampler = HeunSampler(nr_categories=4, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, nr_steps=nr_steps)

    # %% create the data
    x = torch.ones(N)
    x[N//2:] = -1
    X = [x, x]

    correct_class = torch.zeros(N)
    correct_class[:N//2] = 1
    correct_class[N//2:] = 2

    # %% sample a couple of times to get a majority class
    end_estimate_prior       = []
    end_estimate_likelihood1 = []
    end_estimate_likelihood2 = []
    end_estimate_posterior   = []

    for i in range(nr_trials):
        # sample
        y_over_time_prior       = sampler.sample(N, score_prior)
        y_over_time_likelihood1 = sampler.sample(N, score_likelihood_1, X[0])
        y_over_time_likelihood2 = sampler.sample(N, score_likelihood_2, X[1])
        y_over_time_posterior   = sampler.sample(N, fdsm, X)

        # store the end estimate
        end_estimate_prior.append(y_over_time_prior[-1].argmax(dim=1))
        end_estimate_likelihood1.append(y_over_time_likelihood1[-1].argmax(dim=1))
        end_estimate_likelihood2.append(y_over_time_likelihood2[-1].argmax(dim=1))
        end_estimate_posterior.append(y_over_time_posterior[-1].argmax(dim=1))

    # get the majority class
    majority_vote_prior       = torch.stack(end_estimate_prior).mode(dim=0).values
    majority_vote_likelihood1 = torch.stack(end_estimate_likelihood1).mode(dim=0).values
    majority_vote_likelihood2 = torch.stack(end_estimate_likelihood2).mode(dim=0).values
    majority_vote_posterior   = torch.stack(end_estimate_posterior).mode(dim=0).values

    # %% plot the results
    fig, ax = plt.subplots(5,1, figsize=(10,8))
    ax[0].plot(majority_vote_prior, label="prior")
    ax[0].set_title("prior")
    ax[0].grid()
    ax[0].set_ylim(-0.2,3.2)
    ax[3].set_ylabel("class")

    ax[1].plot(majority_vote_likelihood1, label="likelihood 1")
    ax[1].set_title("likelihood 1")
    ax[1].grid()
    ax[1].set_ylim(-0.2,3.2)
    ax[3].set_ylabel("class")

    ax[2].plot(majority_vote_likelihood2, label="likelihood 2")
    ax[2].set_title("likelihood 2")
    ax[2].grid()
    ax[2].set_ylim(-0.2,3.2)
    ax[3].set_ylabel("class")

    ax[3].plot(majority_vote_posterior, label="posterior")
    ax[3].set_title("posterior")
    ax[3].grid()
    ax[3].set_ylim(-0.2,3.2)
    ax[3].set_ylabel("class")

    ax[4].plot(correct_class, label="correct class")
    ax[4].set_title("correct class")
    ax[4].grid()
    ax[4].set_ylim(-0.2,3.2)
    ax[4].set_xlabel("sample")
    ax[4].set_ylabel("class")

    plt.savefig("result.png",dpi=300,bbox_inches="tight")
    plt.tight_layout()
    plt.show()


    