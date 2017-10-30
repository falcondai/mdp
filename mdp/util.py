import scipy.signal
import numpy as np

# reward processing
def discount(rewards, gamma):
    # magic formula for computing gamma-discounted rewards
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]

# returns processing
def n_step_return(rewards, values, gamma, bootstrap_value, n_step=1):
    ''' computes n-step TD return '''
    n_step = min(n_step, len(rewards))
    returns = np.concatenate((values[n_step:], [bootstrap_value] * (n_step + 1)))
    for dt in xrange(n_step):
        returns[:-(n_step-dt)] = rewards[n_step-dt-1:] + gamma * returns[:-(n_step-dt)]
    return returns[:-1]

def td_return(rewards, values, gamma, bootstrap_value):
    ''' computes TD return, i.e. n-step TD return with n = 1'''
    return rewards + gamma * np.concatenate((values[1:], [bootstrap_value]))

def mc_return(rewards, gamma, bootstrap_value):
    ''' computes infinity-step return, i.e. MC return, with bootstraping state value.
    equivalent to setting n to larger than len(rewards) in n-step return '''
    return discount(np.concatenate((rewards, [bootstrap_value])), gamma)[:-1]

def lambda_return(rewards, values, gamma, td_lambda, bootstrap_value):
    td_error = td_return(rewards, values, gamma, bootstrap_value) - values
    lambda_error = discount(td_error, gamma * td_lambda)
    return lambda_error + values

def default_ooe_value(default_value):
    ''' use a constant for OOE value in tabular models '''
    return lambda _, __: default_value

def random_ooe_value(low, high):
    ''' fill in a uniformly random value for OOE entries in tabular models '''
    def _fill(ob, value_dict):
        v = np.random.uniform(low, high)
        value_dict[ob] = v
        return v
    return _fill
