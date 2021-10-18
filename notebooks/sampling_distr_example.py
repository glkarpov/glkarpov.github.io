import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data file
df = pd.read_csv('zhvf_uc_sfrcondo_tier_0.33_0.67_month.csv', header=0, infer_datetime_format=True,
                 parse_dates=['ForecastedDate'])

# Randomly select 100 data points with replacement
sample_size = 100
df_sample = df.sample(n=sample_size, replace=True)

mu = df['ForecastYoYPctChange'].mean()  # Population mean
population_var = df['ForecastYoYPctChange'].var()

# Print the mean of the sample. This is our point estimate of the population mean mu
print('Estimate of population mean mu=' + str(df_sample['ForecastYoYPctChange'].mean()))


def get_means_vec(data, n_samples, sample_size):
    means = np.empty(n_samples)
    for i in range(n_samples):
        x = data.sample(n=sample_size, replace=False)
        means[i] = x.mean()
    return means


n_draws = 50

# Let us extract (n_draws) samples of the same length from the population. Statistics calculated on each new sample
# may vary, because as we discussed - sample is a random vector, and so every statistic being a function of random
# vector is a random variable itself.

x_means = get_means_vec(df['ForecastYoYPctChange'], n_draws, sample_size)

# Now we have (n_draws) values of sample mean statistic w.r.t. different samples
# Let us plot the histogram of the distribution of the sample mean

# Plot the distribution
plt.hist(x_means, bins=30)
plt.show()

# If we repeat this process but with increased number of samples we will notice a peak in distribution near some value:

n_draws = 8000
x_means = get_means_vec(df['ForecastYoYPctChange'], n_draws, sample_size)

# Plot the distribution
plt.hist(x_means, bins=30)
plt.show()

# Let's compare the expectation of sample means with population mean mu
print('E(x_mean) = {}, mu = {}'.format(x_means.mean(), mu))

# They should be quite similar!

# Let's then compare the population variance and variance of sample mean
# Important! Please note, that variance of sample mean is not th same as variance of random variable from population.
# Sample mean is our new random variable.

var_sample_mean = (x_means ** 2).mean() - (x_means.mean()) ** 2   # The same as:
# var_sample_mean = x_means.var()

print('var(x_mean) = {}, population variance = {}'.format(var_sample_mean, population_var))

# Which should be in accordance with the formula: var(x_mean) = population_var/sample_size

# Then let us construct a new random variable!

z = (x_means - mu)/(np.sqrt(population_var/sample_size))

plt.hist(z, bins=40)
plt.show()

# what do we see?
