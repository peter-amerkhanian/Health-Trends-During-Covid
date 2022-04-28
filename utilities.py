import pandas as pd
import numpy as np

def bootstrap_sample(data, k, n):
    """
    Performs bootstrap sampling on data to obtain k samples of size n.
    
    Arguments:
        data - Dataset contained as a Pandas DataFrame 
        k - Number of randomly drawn samples
    
    Returns:
        samples - List containing k Pandas DataFrames of size n each
                  corresponding to each sample  
    """
    samples = []
    for _ in range(k):
        samples.append(data.sample(n=n, replace=True)) 
    return samples

def extract_coefs(models, include_intercept = True):
    """
    NOTE: This function has already been implemented. You do not need to modify this!
    
    Extracts coefficients of all the linear regression models in models, and returns
    it as a NumPy array with one model's coefficients as each row.
    
    Arguments:
        models - Contains k sklearn LinearRegression models, each with p + 1 coefficients
        include_intercept - Whether to include intercept in returned coefficients
    
    Returns:
        coef_array - Coefficients of all k models, each with p + 1 coefficients (if intercept
                     enabled, otherwise p). Returned object is k x (p + 1) NumPy array.
    """
    coef_array = np.zeros(shape = (len(models), len(models[0].coef_[0]) + 1))
    for i, m in enumerate(models):
        coef_array[i, 0] = m.intercept_
        coef_array[i, 1:] = m.coef_[0]
    if include_intercept:
        return coef_array 
    return coef_array[:, 1:]

def confidence_interval(coefs):
    """
    Calculates confidence intervals for each theta_i based on coefficients of 
    bootstrapped models. Returns output as a list of confidence intervals.
    
    Arguments:
        coefs - Output of extract_coefs, a k x (p + 1) or k x p NumPy array containing
                coefficients of bootstrapped models
    
    Returns:
        cis - Confidence intervals of each parameter theta_i in the form of a 
              list like this: [(0.5, 0.75), (0.2, 0.4), ...]
    """
    cis = []
    
    for i in range(coefs.shape[1]):
        theta_i_values = coefs[:, i]
        theta_i_lower_ci, theta_i_upper_ci = np.percentile(theta_i_values, 2.5), np.percentile(theta_i_values, 97.5)
        cis.append((theta_i_lower_ci, theta_i_upper_ci))
    
    return cis