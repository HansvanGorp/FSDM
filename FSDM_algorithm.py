"""
Below is a code snippet for the FSDM algorithm. 

The FSDM algorithm allows for the calculation of the posterior score using the following formula:
nabla_y log p(y|X) = nabla_y log p(y) + 1/N Sum [ nabla_y log p(y|x)- nabla_y log p(y) ]

Or in other words:
Posterior = Global prior + average over signals [ individual likelihood - individual prior ]
"""

# %% imports
import numpy as np
import torch


# %% FSDM algorithm
class FSDM(torch.nn.Module):
     """
     This class implements the FSDM algorithm for calculating the posterior score.
     """
     def __init__(self, global_prior_score_model: torch.nn.Module, likelihood_score_models: list[torch.nn.Module], prior_score_models: list[torch.nn.Module]):
          """
          global_prior_score_model: torch.nn.Module           the global prior score model, we expect it to take as inputs y, sigma and return nabla_y log p(y)
          likelihood_score_models:  list[torch.nn.Module]     list of likelihood score models, each model should take as inputs y, X[i], sigma and return nabla_y log p(y|x)
          prior_score_models:       list[torch.nn.Module]     list of prior score models, each model should take as inputs y and return nabla_y log p(y)
          """
          super(FSDM, self).__init__()
          self.global_prior_score_model = global_prior_score_model
          self.likelihood_score_models = likelihood_score_models
          self.prior_score_models = prior_score_models

     def forward(self, y: torch.tensor, sigma: torch.tensor, X: list[torch.tensor]):
          """
          Implements the FSDM algorithm for calculating the posterior score.

          Args:
          y: torch.tensor         tensor of shape (batch_size, output_dim), the target variable w.r.t. we calculate the score
          X: list[torch.tensor]   each element of the list is a tensor of shape (batch_size, input_dim), inputs should be in the same order as the likelihood_score_models and prior_score_models
          sigma: torch.tensor     tensor of shape (batch_size, 1), the standard deviation of the diffusion trajectory. 

          Returns:
          torch.tensor            tensor of shape (batch_size, output_dim), the posterior score
          """
          # assert all lists have the same length
          assert len(X) == len(self.likelihood_score_models) == len(self.prior_score_models)

          # get the number of signals
          N = len(X)

          # calculate the global prior score
          global_prior = self.global_prior_score_model(y, sigma)

          # calculate the individual likelihood scores
          likelihoods = [self.likelihood_score_models[i](y, sigma, x = X[i]) for i in range(N)]
          likelihood = torch.stack(likelihoods, dim=0).sum(dim=0)

          # calculate the individual prior scores
          priors = [self.prior_score_models[i](y, sigma) for i in range(N)]
          prior = torch.stack(priors, dim=0).sum(dim=0)

          # calculate the posterior score
          posterior = global_prior + 1/N * (likelihood - prior)

          # using the posterior score, calculate the end-estimate
          resulting_end_estimate = posterior * sigma**2 + y

          # renormalize the end-estimate over the channels, which is the second dimension
          resulting_end_estimate = resulting_end_estimate / resulting_end_estimate.sum(dim = 1, keepdim = True)

          # get the resulting score from the renormalized end estimate
          resulting_score = (resulting_end_estimate - y) / sigma**2

          return resulting_score

# %% example to run it
if __name__ == "__main__":
     """
     In this example, we show what the FSDM does in a simple case.
     Signal x1 only tells us if we should look at the first two or last two classes.
     Signal x2 only tells us if we should look at the first and third class or the second and fourth class.
     
     Only by combining the information of both signals, can we get the correct posterior score.

     We will check this by printing it out.
     """

     from score_estimator import ScoreToCategoricalDistribution

     score_prior        = ScoreToCategoricalDistribution(nr_categories=4, x_influences=None)
     score_likelihood_1 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([ 1, 1,-1,-1]))
     score_likelihood_2 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([-1, 1,-1, 1]))
     fdsm = FSDM(score_prior, [score_likelihood_1, score_likelihood_2], [score_prior, score_prior])

     # test the score functions for a batch size of 2, with given ys and xs
     y = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
     sigma = torch.tensor([1 , 2]).unsqueeze(1).repeat(1,4)
     x = torch.tensor([10, -10])

     # calculate the scores
     print(score_prior(y, sigma, None))
     print(score_likelihood_1(y, sigma, x))
     print(score_likelihood_2(y, sigma, x))
     print(fdsm(y, sigma, [x, x]))
