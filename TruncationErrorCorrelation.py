#
#  Truncation Error Correlation
#   
#  Created by: Nathan Carter, Summer 2023
#  Last Updated by: Nathan Carter, Oct 13 2025
#  
#
#  Imports -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import time
import emcee
from math import exp, log
from scipy.optimize import minimize
import corner
from math import pi

# just for plot resolution
plt.rcParams['savefig.dpi'] = 300 
# save to working directory
savedir = "./"

#  -------------------------------------
#  PULLING DATA FROM DATAFILE

# getData - pulls from a file any number of columns of numerical data
#     filename - the name of the file you are pulling from
#     columns - a 1D array of ints, the number, from left to right, of columns you want
# returns - a 2-D array of the columns in the order they appear in the parameter 'columns'
def getData(filename: str, columns: list):
    file = open(filename, 'r')
    listforfile = []
    data = []
    for line in file:
        if (line.startswith('#') == False):
            listforfile += [line.split()]
    [data.append([float(x[y]) for x in listforfile]) for y in columns]
    file.close()
    return data
    
starttime = time.time()
x, data, stdv = getData('D1_c_5.dat', [0, 1, 2])
# stdv = np.multiply(data, 0.01)
#  --------------------------


# FUNCTION DEFINITIONS ---------------------------------------------------------------
# 
# FUNCTION: create_model
# 
# PARAMS:
#    - n : order of desired polynomial
#
# returns a python function which itself 
# returns a polynomial of order n
#
# the function it creates will take the x
# input from domain, then for each order it
# will need a parameter, so n+2 parameters overall
# like in an order n = 5 polynomial, 
# 6 parameters + x domain, so 7 parameters total
def create_model(n):

    def polynomial(x, *coefficients):

        if len(coefficients) != n + 1:
            raise ValueError(f"Expected {n + 1} coefficients, got {len(coefficients)}")
        
        result = 0
        for i, coeff in enumerate(coefficients):
            power = n - i  # Highest order first
            result += coeff * (x ** power)
        return result
    
    return polynomial
#
# 
# 
# FUNCTION: sample_at_k
# 
# PARAMS:
#    - num_of_parameters: number of parameters for your model
#    - model            : function which takes {num_of_parameters} parameters as input,
#                           plus of course domain input i.e. x
#    - samples          : number of desired samples
#
# Samples for a given number of parameters.
def sample_at_k(num_of_parameters, model, samples):

    # Defines the Chi^2 function for a given model.
    def chi2_function(x, y_data, stdv, x_data):
        return sum([((((l - model(n, *x))/m))**2) for l, m, n in zip(y_data, stdv, x_data)])

    # Defines the likelihood with a naturalness prior for the given model
    # includes the naturalness prior inside it already.
    def log_prob(x, y_data, stdv, x_data):
        return -1/2 * (((chi2_function(x,  y_data, stdv, x_data))) + (sum((x**2)/R**2)))

    # does the minimization to get input for emcee
    res = minimize(chi2_function, x0=np.ones(num_of_parameters), args=(data, stdv, x))
    cov_matrix = np.diag((2*res.hess_inv), 0)

    # gets the number of walkers and dimension
    pos = res.x + 1e-5 * np.random.randn(50, num_of_parameters)
    nwalkers, ndim = pos.shape

    # actually running the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(data, stdv, x))
    sampler.run_mcmc(pos, samples, progress=True)
    flat_samples = sampler.get_chain(discard=1500, thin = 15, flat=True)

    corner.corner(flat_samples, quantiles=[0.16, 0.5, 0.84], show_titles=True, labels=["a", "b", "c", "d", "e"], truths=res.x)
    plt.savefig(savedir + "corner.png")

    # return the chain (flattened)
    return flat_samples
# ------------------------------------------------------------------------------------


# PARAMETERS FOR SPECIFIC RUN --------------------------------------------------------
# 
# MODIFY THESE
#

# Determine naturalness parameter for the prior
R = 5

# HOW MANY PARAMETERS DO YOU WANT?
# Paper uses up to quartic term ()
num_of_parameters = 5

# HOW MANY PARAMETERS DO YOU WANT TO TRUNCATE?
truncation = 1

# This creates the polynomial model
model = create_model(num_of_parameters - 1)
# Sample!
# All of the actual EMCEE sampling is done in this function. See above for more info.
flat_samples = sample_at_k(num_of_parameters, model, 10000)

# TRUNCATION ERROR STUFF ---------------------------------------------------------------------
# 

# Now that we have samples, 
# calculate the medians and get ready for truncation
# Get the medians for each parameter
medians = [np.median([i[j] for i in flat_samples]) for j in range(num_of_parameters)]
#
# We need our domain for calculations.
#
domain = np.linspace(1e-5, 0.4, 200)
#
# everything we need to store the data
prediction = []
prediction_error = []
#
trunc_pred = []
trunc_pred_error = []
#
low_pred = []
low_pred_error = []
#
low_data = []
trunc_data = []
#
# For every value in the domain
for x_value in domain:

    # calculate the truncated model values for each sample
    trunc_model_values = [model(x_value, *i[:truncation:], *medians[truncation::]) for i in flat_samples]
    # calculate the full model values for each sample
    model_values = [model(x_value, *i) for i in flat_samples]
    # calculate the low order model values for each sample
    low_pred_model_values = [model(x_value, *np.zeros(truncation), *i[truncation::]) for i in flat_samples]

    # store the low order data and truncation data for later use
    # in the redone errors (quadrature vs. using correlation)
    low_data.append([model(x_value, *np.zeros(truncation), *i[truncation::]) for i in flat_samples])
    trunc_data.append([model(x_value, *i[:truncation:], *medians[truncation::]) for i in flat_samples])

    # store the needed values in each array
    prediction.append(np.median(model_values))
    prediction_error.append(np.std(model_values))
    trunc_pred.append(np.median(trunc_model_values))
    trunc_pred_error.append(np.std(trunc_model_values))
    low_pred.append(np.median(low_pred_model_values))
    low_pred_error.append(np.std(low_pred_model_values))

# This part calculates the correlation coefficient between the
# low order and truncation data
Rvalues = []
ho_std_array = []
lo_std_array = []
for i, j in zip(trunc_data, low_data):
    ho_mean = np.mean(i)
    ho_std = np.std(i)
    ho_std_array.append(ho_std)
    lo_mean = np.mean(j)
    lo_std = np.std(j)
    lo_std_array.append(lo_std)
    # stores the correlation coefficients for error recombination later
    Rvalues.append(np.divide(np.mean(np.multiply(np.subtract(j, lo_mean), np.subtract(i, ho_mean))), ho_std*lo_std))


# saveData - saves a 2D or higher array to a file.
#     filename - the name of the file you wish to create
#     data     - the array of data you wish to save
#     sig      - the number of significant figures after the decimal point (always Scientific Notation)
# returns  - creates a datafile with the values in columns, non-csv (whitespace delimiter)
def saveData(filename: str, data: list, sig: int):
    columns = len(data)
    print(columns)
    if(columns < 2):
        raise Exception("Saving a 1 column (# of columns = " + columns + ") array. Save aborted.")
    else:

        stringformat = ""
        for i in range(columns):
            stringformat += "{" + str(i) + ":" + str(sig) + "e}" + "\t"
        
        values = len(data[0])
        file = open(filename, 'w')
        for i in range(values):
            file.write(stringformat.format(*[data[v][i] for v in range(columns)]))
            if (i != (values - 1)):
                file.write("\n")

    file.close()

    return
    
print("Saving data...")
saveData(savedir + "prediction.dat", [domain, prediction, prediction_error], sig=8)
saveData(savedir + "truncation_prediction.dat", [domain, trunc_pred, ho_std_array], sig=8)
saveData(savedir + "lower_order_prediction.dat", [domain, low_pred, lo_std_array], sig=8)
saveData(savedir + "correlation.dat", [domain, Rvalues], sig=8)
print(f"{'\033[92m'}SUCCESS{'\033[0m'}")

# -----------------------------------------------------------------------------------------------

# PLOT CREATION ---------------------------------------------------------------------------------

# MOST OF THESE CAN BE IGNORED, ONLY CREATE_FIGURE1 THROUGH 4 ARE USED FOR PAPER

# Full model prediction plot
def create_figure1():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    ax.tick_params(which='major', length=6, width=1, direction="in")
    ax.tick_params(which='minor', length=4, width=1, direction="in")
    ax.minorticks_on()
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    ax.plot(domain, prediction, color='blue', linestyle='-')
    ax.fill_between(domain, np.add(prediction, prediction_error), np.subtract(prediction, prediction_error), color='blue', alpha=0.25)

    plt.savefig(savedir + "ModelWithPrior.png", bbox_inches="tight")
    # plt.show(block=False)

# PPD plot of just the truncation and parameter uncertainty
def create_figure2():

    color1 = "red"
    color2 = "green"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    ax.tick_params(which='major', length=6, width=1, direction="in")
    ax.tick_params(which='minor', length=4, width=1, direction="in")
    ax.minorticks_on()
    # plots the data points
    ax.scatter(x, data, color='black', zorder=6)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=5)

    # LOW ORDER BAND
    ax.plot(domain, low_pred, color=color1, linestyle='-', zorder=2)
    ax.fill_between(domain, np.add(low_pred, lo_std_array), np.subtract(low_pred, lo_std_array), color=color1, label="Parametric uncertainty", alpha=0.3, zorder=1)
    # TRUNCATED ORDER BAND
    ax.plot(domain, trunc_pred, color=color2, linestyle='-', zorder=3)
    ax.fill_between(domain, np.add(trunc_pred, ho_std_array), np.subtract(trunc_pred, ho_std_array), color=color2, label="Truncation uncertainty", alpha=0.3, zorder=4)
    
    plt.legend(loc="upper left")
    plt.savefig(savedir + "Truncation_vs_ParamUncertainities.png", bbox_inches="tight")
    # plt.show(block=False)

# Pearson Correlation Coefficient plot
def create_figure3():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'$\rho (x)$', rotation='horizontal')
    ax.set_xlabel("x")
    ax.set_xlim(0, 0.4)
    ax.tick_params(which='major', length=6, width=1, direction="in")
    ax.tick_params(which='minor', length=4, width=1, direction="in")
    ax.minorticks_on()

    ax.scatter(domain, Rvalues, color='black', s=7)

    plt.savefig(savedir + "Correlation.png", bbox_inches="tight")
    # plt.show(block=False)

# PPD of the full model uncertainty, then the full model uncertainty but not considering correlations.
# note that the full model uncertainty in this plot uses the pearson correlation coefficient to recalculate it,
# and that it is the same as the output from EMCEE, while the model uncertainty not taking into account correlations is different.
def create_figure4():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    ax.tick_params(which='major', length=6, width=1, direction="in")
    ax.tick_params(which='minor', length=4, width=1, direction="in")
    ax.minorticks_on()
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    ax.plot(domain, prediction, color='blue', linestyle='-')
    redone_errors = [np.sqrt(i**2 + 2*r*i*j + j**2) for i,j,r in zip(ho_std_array, lo_std_array, Rvalues)]
    redone_errors_quadrature = [np.sqrt(i**2 + j**2) for i,j in zip(ho_std_array, lo_std_array)]
    ax.fill_between(domain, np.add(trunc_pred, redone_errors), np.subtract(trunc_pred, redone_errors), color='blue', alpha=0.25, label='Correlated uncertainties')
    ax.fill_between(domain, np.add(trunc_pred, redone_errors_quadrature), np.subtract(trunc_pred, redone_errors_quadrature), color='green', alpha=0.25, label='Uncorrelated uncertainties')

    plt.legend(loc="upper left")

    plt.savefig(savedir + "Model_Quad_vs_WithPrior.png", bbox_inches="tight")
    # plt.show(block=False)

# plot for just the data 
def create_figure5():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    plt.savefig(savedir + "data.png")
    # plt.show(block=False)

# Same as figure 4, just with 1 sigma
def create_figure6():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    ax.plot(domain, prediction, color='blue', linestyle='-')
    redone_errors = [np.sqrt(i**2 + 2*r*i*j + j**2) for i,j,r in zip(ho_std_array, lo_std_array, Rvalues)]
    redone_errors_quadrature = [np.sqrt(i**2 + j**2) for i,j in zip(ho_std_array, lo_std_array)]
    ax.fill_between(domain, np.add(trunc_pred, redone_errors), np.subtract(trunc_pred, redone_errors), color='blue', alpha=0.25, label=r'$1\sigma$ With Prior')
    ax.fill_between(domain, np.add(trunc_pred, redone_errors_quadrature), np.subtract(trunc_pred, redone_errors_quadrature), color='green', alpha=0.25, label=r'$1\sigma$ In Quadrature')

    plt.legend(loc="upper left")

    plt.savefig(savedir + "QuadvsPrior_1sigma.png")
    # plt.show(block=False)

# Same as figure 4, just with 2 sigma
def create_figure7():

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    ax.plot(domain, prediction, color='blue', linestyle='-')
    redone_errors = [np.sqrt(i**2 + 2*r*i*j + j**2) for i,j,r in zip(ho_std_array, lo_std_array, Rvalues)]
    redone_errors_quadrature = [np.sqrt(i**2 + j**2) for i,j in zip(ho_std_array, lo_std_array)]

    ax.fill_between(domain, np.add(trunc_pred, np.multiply(2, redone_errors)), np.subtract(trunc_pred, np.multiply(2, redone_errors)), color='blue', alpha=0.25, label=r'$2\sigma$ With Prior')
    ax.fill_between(domain, np.add(trunc_pred, np.multiply(2, redone_errors_quadrature)), np.subtract(trunc_pred, np.multiply(2, redone_errors_quadrature)), color='green', alpha=0.25, label=r'$2\sigma$ In Quadrature')

    plt.legend(loc="upper left")

    plt.savefig(savedir + "QuadvsPrior_2sigma.png")
    # plt.show(block=False)

# Same as figure 4, just with 3 sigma
def create_figure8():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)

    ax.plot(domain, prediction, color='blue', linestyle='-')
    redone_errors = [np.sqrt(i**2 + 2*r*i*j + j**2) for i,j,r in zip(ho_std_array, lo_std_array, Rvalues)]
    redone_errors_quadrature = [np.sqrt(i**2 + j**2) for i,j in zip(ho_std_array, lo_std_array)]

    ax.fill_between(domain, np.add(trunc_pred, np.multiply(3, redone_errors)), np.subtract(trunc_pred, np.multiply(3, redone_errors)), color='blue', alpha=0.25, label=r'$3\sigma$ With Prior')
    ax.fill_between(domain, np.add(trunc_pred, np.multiply(3, redone_errors_quadrature)), np.subtract(trunc_pred, np.multiply(3, redone_errors_quadrature)), color='green', alpha=0.25, label=r'$3\sigma$ In Quadrature')

    plt.legend(loc="upper left")

    plt.savefig(savedir + "QuadvsPrior_3sigma.png")
    # plt.show(block=False)

# Plotting the full ppd with the actual underlying function used to generate the data
def create_figure9():
    
    f = lambda x : (0.5 + np.tan((pi/2)*x))**2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.15, 1.65)
    # plots the data points
    ax.scatter(x, data, color='black', zorder=3)
    ax.errorbar(x, data, stdv, color='black', linestyle='', zorder=2)
    ax.plot(domain, prediction, color='blue', linestyle='-')

    ax.plot(domain, [f(i) for i in domain], color='red')

    redone_errors = [np.sqrt(i**2 + 2*r*i*j + j**2) for i,j,r in zip(ho_std_array, lo_std_array, Rvalues)]
    redone_errors_quadrature = [np.sqrt(i**2 + j**2) for i,j in zip(ho_std_array, lo_std_array)]
    ax.fill_between(domain, np.add(trunc_pred, redone_errors), np.subtract(trunc_pred, redone_errors), color='blue', alpha=0.25, label='With Correlation')
    ax.fill_between(domain, np.add(trunc_pred, redone_errors_quadrature), np.subtract(trunc_pred, redone_errors_quadrature), color='green', alpha=0.25, label='Quadrature')

    plt.savefig(savedir + "ModelWithTrue.png")
    # plt.show(block=False)

# relative difference plot to compare truncation and parameter uncertainty to the full ppd prediction
def create_comparison_figure():
    size_of_truncation = np.divide(np.subtract(trunc_pred_error, prediction_error), prediction_error)
    size_of_parametric = np.divide(np.subtract(low_pred_error, prediction_error), prediction_error)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation='horizontal')
    ax.tick_params(which='minor', length=4, width=1, direction="in")
    ax.minorticks_on()

    plt.plot(domain, size_of_truncation, color='green', label="Truncation Difference")
    plt.plot(domain, size_of_parametric, color='red', label="Parametric Difference")
    plt.legend(loc='upper left')
    plt.savefig(savedir + "Compare.png", bbox_inches="tight")
    # plt.show(block=False)

print("Saving plots...")
create_figure1()
create_figure2()
create_figure3()
create_figure4()
create_comparison_figure()
print(f"{'\033[92m'}SUCCESS{'\033[0m'}")

plt.close()
#create_figure5()
#create_figure6()
#create_figure7()
#create_figure8()
#create_figure9()

# -----------------------------------------------------------------------------------------------


