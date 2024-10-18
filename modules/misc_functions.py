# -*- coding: utf-8 -*-
"""
Created on Thursday October 2nd 2024

@author: Felipe Segundo Abril Bermúdez
"""

# Libraries ----
import numpy as np  # type: ignore

from tqdm import tqdm  # type: ignore
from multiprocessing import Pool, cpu_count


# Estimation of p-norm ----
def estimate_p_norm(x: np.ndarray, y: np.ndarray, p: float):
    r"""Estimation of p-norm defined as:
    :math:`\lim_{q\to p}{\left(\frac{1}{N}\sum_{k=1}^{N}|x_{k}-y_{k}|^{q}\right)^{1/q}}` 

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Arbitrary vector of real values of the same size of y
    y : float or numpy array dtype float
        Arbitrary vector of real values of the same size of x
    p : float
        Exponent for estimation of p-norm

    Returns:
    ---------------------------------------------------------------------------
    z : float
        p-norm estimation between x and y weighted by the size of vectors N
    """  # noqa: 501

    if p == 0:
        z = np.exp(0.5 * np.mean(np.log(np.power(np.abs(x-y), 2))))
    else:
        z = np.power(np.mean(np.power(np.abs(x - y), p)), 1 / p)
    return z


# Estimation of coefficient of determination R2 ----
def estimate_coefficient_of_determination(y: np.ndarray, y_fitted: np.ndarray):
    """Estimation of coefficient of determination R2:

    Args:
    ---------------------------------------------------------------------------
    y : float or numpy array dtype float
        Arbitrary vector of real values of the same size of y_fitted
    y_fitted : float or numpy array dtype float
        Arbitrary vector of real values of the same size of y

    Returns:
    ---------------------------------------------------------------------------
    float
        R2 value (percentage of variance explained by y_fitted)
    """

    return 1 - np.sum(np.power(y - y_fitted, 2)) / np.sum(np.power(y - np.mean(y), 2))  # noqa: 501


# Translate vector values to probabilities ----
def get_probabilities(x: np.ndarray, bins: int = 10, density: bool = True):
    r"""Estimate probabilities of vector x from the histogram data:

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Vector of probabilities or positive values of a sample
    bins : int
        Number of numerical ranges in which the histogram data is aggregated
        (default value 10)
    density : bool
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1
        (default value True)

    Returns:
    ---------------------------------------------------------------------------
    bins_ : float or numpy array dtype float
        Return the bin edges with size of len(hist_) + 1
    bins_midpoint_ : float or numpy array dtype float
        Return the bin edges with the same size of hist_
    hist_ : float or numpy array dtype float
        The values of the histogram
    """

    # Histogram data
    hist_, bins_ = np.histogram(x, bins=bins, density=density)

    # Midpoint per bin
    bins_midpoint_ = np.zeros(len(hist_))
    for k in np.arange(0, len(hist_), 1):
        bins_midpoint_[k] = 0.5 * (bins_[k] + bins_[k+1])

    return bins_, bins_midpoint_, hist_


# Estimate Shannon entropy over a vector of probabilities ----
def estimate_renyi_entropy(x: np.ndarray, p: float):
    r"""Estimation of Renyi entropy defined as:
    :math:`\lim_{q\to p}{\frac{1}{1-q}\log{\left(\sum_{k=1}^{N}p_{k}^{q}\right)}}`

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Vector of probabilities or positive values of a sample
    p : float
        Exponent for estimation of Renyi entropy

    Returns:
    ---------------------------------------------------------------------------
    renyi_entropy : float
        Renyi entropy of the sample
    """  # noqa: 501

    # Normalized array
    x_normalized = x / np.sum(x)
    x_normalized = x_normalized[x_normalized > 0]

    # Hartley or max-entropy
    if p == 0:
        renyi_entropy = np.log(len(x))

    # Min-entropy
    elif np.isinf(p) is True:
        renyi_entropy = -np.log(np.max(x_normalized))

    # Shannon entropy
    elif p == 1:
        renyi_entropy = -np.sum(x_normalized * np.log(x_normalized))

    # Renyi entropy
    else:
        renyi_entropy = np.log(np.sum(np.power(x_normalized, p))) / (1 - p)

    return renyi_entropy


# Estimate mutual information over a two vectors of probabilities ----
def estimate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    joint_xy: np.ndarray
):
    """Estimation of mutual information between two vectors:

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Vector of probabilities or positive values of a sample (marginal). The
        length of this vector is M
    y : float or numpy array dtype float
        Vector of probabilities or positive values of a sample (marginal). The
        length of this vector is N
    joint_xy : float or numpy array dtype float
        Vector of joint probabilities or positive values of a sample (joint
        probability). The length of this vector is M * N

    Returns:
    ---------------------------------------------------------------------------
    mutual_information : float
        Mutual information of the samples
    """

    # Mutual information
    hx = estimate_renyi_entropy(x=x, p=1)
    hy = estimate_renyi_entropy(x=y, p=1)
    hxy = estimate_renyi_entropy(x=joint_xy, p=1)

    mutual_information = hx + hy - hxy
    if len(x) == len(y):
        if (np.array(x) == np.array(y)).all():
            mutual_information = hx

    return mutual_information


# Estimate shared information distance over a two vectors of probabilities ----
def estimate_shared_information_distance(
    x: np.ndarray,
    y: np.ndarray,
    joint_xy: np.ndarray
):
    """Estimation of shared information distance between two vectors:

    Args:
    ---------------------------------------------------------------------------
    x : float or numpy array dtype float
        Vector of probabilities or positive values of a sample (marginal)
    y : float or numpy array dtype float
        Vector of probabilities or positive values of a sample (marginal)
    joint_xy : float or numpy 2D array dtype float
        Matrix of joint probabilities or positive values of a sample (joint)

    Returns:
    ---------------------------------------------------------------------------
    shared_information : float
        Shared information distance of the samples
    """

    shared_information = estimate_renyi_entropy(x=joint_xy, p=1)
    shared_information -= estimate_mutual_information(x=x, y=y, joint_xy=joint_xy)  # noqa: 501

    return shared_information


# Deployment of parallel run in function of arguments list ----
def parallel_run(
    fun,
    arg_list,
    tqdm_bar: bool = False
):
    """Implement parallel run in arbitrary function with input arg_list:

    Args:
    ---------------------------------------------------------------------------
    fun : function
        Function to implement in parallel
    arg_list : list of tuples
        List of arguments to pass in function
    tqdm_bar : bool
        Progress bar (default value is False)

    Returns:
    ---------------------------------------------------------------------------
    m : list of objects
        Function evaluation in all possible combination of tuples
    """

    if tqdm_bar:
        m = []
        with Pool(processes=cpu_count()) as p:
            with tqdm(total=len(arg_list), ncols=60) as pbar:
                for _ in p.imap(fun, arg_list):
                    m.append(_)
                    pbar.update()
            p.terminate()
            p.join()
    else:
        p = Pool(processes=cpu_count())
        m = p.map(fun, arg_list)
        p.terminate()
        p.join()

    return m


# Extract scientific notation in Python string ----
def extract_sci_notation(number: float, significant_figures: int = 2):
    """Extract scientific notation (mantissa and exponent) in Python number:

    Args:
    ---------------------------------------------------------------------------
    number : float
        float number
    significant_figures : int
        Number of significant figures in number (default value 2)

    Returns:
    ---------------------------------------------------------------------------
    mantissa : float
        Mantissa of scientific notation rounded with significant figures
    exponent : int
        Exponent in scientific notation without '+' sign
    """

    z = "{0:.{1:d}e}".format(number, significant_figures)
    mantissa, exponent = z.split("e")
    return mantissa, int(exponent)  # Remove leading "+" and strip leading zero


# Define scientific notation in LaTeX string ----
def define_sci_notation_latex(number: float, significant_figures: int = 2):
    """Define scientific notation of Python float number as LaTeX string:

    Args:
    ---------------------------------------------------------------------------
    number : float
        float number
    significant_figures : int
        Number of significant figures in number (default value 2)

    Returns:
    ---------------------------------------------------------------------------
    z : string
        LaTeX string for scientific notation of the number
    """

    mantissa, exponent = extract_sci_notation(
        number=number,
        significant_figures=significant_figures
    )
    z = "$" + str(mantissa) + " \\times 10^{" + str(exponent) + "}$"
    return z


# Define scientific notation in a vector as Physical Review format ----
def define_sci_notation_latex_vectorize(
    x: np.ndarray,
    significant_figures: int = 2
):
    """Define scientific notation of Python numpy array number as LaTeX string:

    Args:
    ---------------------------------------------------------------------------
    x : numpy array dtype float
        Numpy array of floats number
    significant_figures : int
        Number of significant figures in number (default value 2)

    Returns:
    ---------------------------------------------------------------------------
    z : string
        Vector of LaTeX strings for scientific notation of the numbers
    """

    x = sorted(x)
    mantissa_min, exponent_min = extract_sci_notation(
        number=np.min(np.abs(x)),
        significant_figures=significant_figures
    )
    del mantissa_min

    z = []
    for i in np.arange(len(x)):
        z_ = str(round(x[i] / np.float_power(10, exponent_min), significant_figures))  # noqa: 501
        if i + 1 == len(x):  # Only last element with sci_notation
            z.append("$" + str(z_) + " \\times 10^{" + str(exponent_min) + "}$")  # noqa: 501
        else:  # Only mantissa
            z.append("$" + str(z_) + "$")

    return z
