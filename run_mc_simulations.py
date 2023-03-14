from TDA_utils import *



alphas = np.linspace(1.6,2,9)
amplitudes = np.linspace(5,9,9)

mc_iterations = 1000
avg_bcs = []
acc_threshs = np.zeros_like(alphas, dtype = float)
print("running {} mc simulations for the tda approach")
test_powers_alpha = compute_tests_powers(alphas, amplitudes, mc_iterations, metric = "l1", method = "tda")
plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha)

print("running {} mc simulations for the cvb approach")
test_powers_alpha = compute_tests_powers(alphas, amplitudes, mc_iterations, metric = "l1", method = "cvb")
plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha)