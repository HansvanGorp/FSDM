"""
Below is a code snippet for a model that calculates the score to a categorical distribution of 4 dimensions.
"""

# %% imports
import numpy as np
import torch

# create a model that calculates the score to a a categorical distribution
class ScoreToCategoricalDistribution(torch.nn.Module):
    def __init__(self, nr_categories: int = 4, x_influences: torch.tensor = None):
        super(ScoreToCategoricalDistribution, self).__init__()
        self.nr_categories = nr_categories
        self.x_influences = x_influences

        self.positive_indices = torch.arange(self.nr_categories)[self.x_influences > 0]
        self.negative_indices = torch.arange(self.nr_categories)[self.x_influences < 0]

    def forward(self, y: torch.tensor, sigma: torch.tensor, x: torch.tensor = None):
        """
        Given a y of a shape (batch_size, classes) calculate the score of y w.r.t. a categorical distribution.
        This we do if we know something about the classes according to x. If x is None, we assume all classes are equally likely.
        We try to find which y is highest, that one is attracted to 1, the others to 0.
        However, x can influence this attraction, making some classes be ignored in the selection process.
        """
        batch_size = y.shape[0]
        y_to_select_from = y.clone()

        # if x is not None, we only consider the classes that are influenced by x
        if x is not None:
            y_to_select_from[:, self.positive_indices] = y_to_select_from[:, self.positive_indices] - (x.unsqueeze(1)<0)*1e32
            y_to_select_from[:, self.negative_indices] = y_to_select_from[:, self.negative_indices] - (x.unsqueeze(1)>0)*1e32

        # for each batch, select the highest y
        max_y = torch.argmax(y_to_select_from, dim=1)

        # then create an end-estimate, which is a one-hot encoded vector with 1 at the max_y
        end_estimate = torch.zeros_like(y)
        end_estimate[torch.arange(batch_size), max_y] = 1

        # calculate the score using Tweedies formula.
        score = (end_estimate - y) / sigma**2

        return score
    
# %% Test it
if __name__ == "__main__":
    score_prior        = ScoreToCategoricalDistribution(nr_categories=4, x_influences=None)
    score_likelihood_1 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([ 1, 1,-1,-1]))
    score_likelihood_2 = ScoreToCategoricalDistribution(nr_categories=4, x_influences=torch.tensor([-1, 1,-1, 1]))

    # test the score functions for a batch size of 2, with given ys and xs
    y = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    sigma = torch.tensor([1 , 2]).unsqueeze(1).repeat(1,4)
    x = torch.tensor([10, -10])

    # calculate the scores
    print(score_prior(y, sigma, None))
    print(score_likelihood_1(y, sigma, x))
    print(score_likelihood_2(y, sigma, x))

