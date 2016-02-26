from pystan import StanModel
import matplotlib.pyplot as pyplot
import numpy as np

x_data = 100 * np.random.random(50)
y_data = 0.7 * x_data + 13
N = len(x_data)

# Introduce noise
x_data = np.random.normal(x_data, 7)
y_data = np.random.normal(y_data, 8)

# plot the data
pyplot.plot(x_data, y_data, 'o')

stan_data_mappings = {
  'x': x_data,
  'y': y_data,
  'N': N,
}

model = StanModel(file='models/univariate_regression.stan')

fit = model.sampling(data=stan_data_mappings)

params = fit.extract()
a_pred = params['a']
b_pred = params['b']
sigma_pred = params['sigma']

# Draw 100 points from where x_data is.
xfit = np.linspace(-10 + min(x_data), 10 + max(x_data), 100)

# Number of samples.
M = len(a_pred)

yfit = a_pred.reshape((M, 1)) + b_pred.reshape((M, 1)) * xfit

# Get mean for 100 poinst and std at those points.
mu = yfit.mean(0)
sig = 2 * yfit.std(0)

# Plot the estimate with errors.
pyplot.plot(xfit, mu, '-k')
pyplot.fill_between(xfit, mu - sig, mu + sig, color='lightgray')
pyplot.xlabel('x')
pyplot.ylabel('y')

pyplot.show()
