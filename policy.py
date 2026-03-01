import numpy as np
import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations: torch.Tensor) -> ptd.Distribution:
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations: np.ndarray, filter_actions: np.ndarray = None, return_log_prob: bool = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        raise NotImplementedError()
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############
        distributions: ptd.Distribution = self.action_distribution(observations)
        sampled_actions = distributions.sample()
        log_probs = distributions.log_prob(sampled_actions).detach().numpy()
        sampled_actions = sampled_actions.detach().numpy()
        #######################################################
        #########          END YOUR CODE.          ############
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


# class CategoricalPolicy(BasePolicy, nn.Module):
#     def __init__(self, network):
#         nn.Module.__init__(self)
#         self.network = network

#     def action_distribution(self, observations):
#         """
#         Args:
#             observations: torch.Tensor of shape [batch size, dim(observation space)]
#         Returns:
#             distribution: torch.distributions.Categorical where the logits
#                 are computed by self.network

#         See https://pytorch.org/docs/stable/distributions.html#categorical
#         """
#         #######################################################
#         #########   YOUR CODE HERE - 1-2 lines.    ############
#         distribution = ptd.Categorical(logits=self.network(observations))
#         #######################################################
#         #########          END YOUR CODE.          ############
#         return distribution
    
#     def act(self, observations: np.ndarray, filter_actions: np.ndarray = None, return_log_prob: bool = False):
#         """
#         Args:
#             observations: np.array of shape [batch size, dim(observation space)]
#             filter_actions: np.array of shape [batch size, dim(observation space)] limiting valid actions
#         Returns:
#             sampled_actions: np.array of shape [batch size, *shape of action]
#             log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

#         TODO:
#         Call self.action_distribution to get the distribution over actions,
#         then sample from that distribution. Compute the log probability of
#         the sampled actions using self.action_distribution. You will have to
#         convert the actions and log probabilities to a numpy array, via numpy(). 

#         You may find the following documentation helpful:
#         https://pytorch.org/docs/stable/distributions.html
#         """
#         observations = np2torch(observations)
#         #######################################################
#         #########   YOUR CODE HERE - 1-4 lines.    ############
#         distributions: ptd.Categorical = self.action_distribution(observations)

#         if filter_actions is not None:
#             filter_actions = np2torch(filter_actions)
#             probs: torch.Tensor = distributions.probs
#             probs[~filter_actions] = 0
#             probs /= probs.sum(-1, keepdim=True)
#             distributions = ptd.Categorical(probs)
        
#         sampled_actions = distributions.sample()
#         log_probs = distributions.log_prob(sampled_actions).detach().numpy()
#         sampled_actions = sampled_actions.detach().numpy()
#         #######################################################
#         #########          END YOUR CODE.          ############
#         if return_log_prob:
#             return sampled_actions, log_probs
#         return sampled_actions


class CategoricalPolicyValue(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations, filter_actions: np.ndarray = None) -> tuple[ptd.Categorical, torch.FloatTensor]:
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        logits, value = self.network(observations, return_value=True)
        distribution = ptd.Categorical(logits=logits)

        if filter_actions is not None:
            filter_actions = np2torch(filter_actions)
            probs: torch.Tensor = distribution.probs
            probs[~filter_actions] = 0
            probs /= probs.sum(-1, keepdim=True)
            distribution = ptd.Categorical(probs)
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution, value
    
    # def act(self, observations: np.ndarray, filter_actions: np.ndarray = None, return_log_prob: bool = False) -> tuple[np.ndarray, np.ndarray]:
    #     observations = np2torch(observations)
    #     distributions, values = self.action_distribution(observations, filter_actions=filter_actions)
    #     sampled_actions = distributions.sample()
    #     log_probs = distributions.log_prob(sampled_actions).detach().numpy()
    #     sampled_actions = sampled_actions.detach().numpy()
    #     if return_log_prob:
    #         return sampled_actions, log_probs
    #     return sampled_actions, values
    
    def best_action(self, observations: np.ndarray, filter_actions: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            observations = np2torch(observations[None])
            distributions, values = self.action_distribution(observations, filter_actions=filter_actions[None])
            value = float(values.item())
            best_action = np.argmax(distributions.probs.numpy())
            return best_action, value
