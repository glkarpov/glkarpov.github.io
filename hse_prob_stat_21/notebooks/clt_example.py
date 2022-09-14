import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# function makes (n_samples) samples of size (sample_size) from population and computes sample mean for each sample
def get_means_vec(data, n_samples, sample_size):
    means = np.empty(n_samples)
    for i in range(n_samples):
        x = np.random.choice(data, sample_size, replace=False)
        means[i] = x.mean()
    return means


# Generate Population Data
N = 80000

# The main idea of the Central Limit Theorem is that we can sample from absolutely any distribution
# But scaled distribution of sample means ... still will be normal. :)

# Let us generate 3 populations of N objects each w.r.t. different distributions:

data_set_uniform = np.random.randint(0, 100, size=N)  # Uniform
data_set_exp = np.random.exponential(size=N)  # Exponential
data_set_bin = np.random.binomial(n=20, p=0.2, size=N)  # Binomial(20, 0.2)

# Then let us sum up these 3 arrays of populations with changes to each of them.
# As a result we will get a population of absolutely strange nature

data_set = data_set_exp ** 2 + (5 * data_set_bin) + np.sin(data_set_uniform)

# Find out global properties of population:
population_mu = np.mean(data_set)
population_var = np.var(data_set)

# Generate Sample Means
sample_size = 200  # keep fixed for the first time
n_draws = 5000
x_means = get_means_vec(data_set, n_draws, sample_size)

# Plot Sample Means
# Change (n_draws) and look how the picture is changing

plt.figure(figsize=(20, 10))
sns.histplot(x_means, bins=30).grid()
plt.title('Sampling Distribution of Sample Mean ({} samples of size {})'.format(n_draws, sample_size))
plt.axvline(x=np.mean(x_means), label='Mean of Sample Means')
plt.legend()
plt.show()

# Now let us scale our sampling distribution to the form in CLT:

z = (x_means - population_mu) / (np.sqrt(population_var / sample_size))
x_space = np.linspace(-3, 3)

plt.figure(figsize=(20, 10))
sns.histplot(z, bins=40).grid()
counts, _, _ = plt.hist(z, bins=40, alpha=0.5)  # just in order to find out the scaling coefficient for PDF
# plt.hist(z, bins=40, alpha=0.5)
plt.title('Sampling Distribution of scaled sample means ({} samples of size {})'.format(n_draws, sample_size))
plt.axvline(x=np.mean(z), label='Mean of Sample Means')

# scaling of normal PDF is needed, because histogram has large values on y-axis, and we need to fit them

plt.plot(x_space, np.max(counts) * stats.norm.pdf(x_space, 0, 1) * np.sqrt(2 * np.pi), label='Normal')
plt.legend()
plt.show()
