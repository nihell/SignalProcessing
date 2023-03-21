from TDA_utils import *



alphas = np.linspace(1.6,2,9)
amplitudes = np.linspace(5,9,9)

mc_iterations = 1000
print("running {} mc simulations for the tda approach".format(mc_iterations))
test_powers_alpha = compute_test_powers(alphas, amplitudes, mc_iterations, metric = "l1", method = "tda")
#print(test_powers_alpha.shape)
np.savetxt("{}iterations_l1-metric_tda_test_powers.txt".format(mc_iterations),np.array(test_powers_alpha))
f,ax = plt.subplots(figsize = (len(amplitudes),len(alphas)))
plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha, "tda", ax=ax)
plt.savefig("normalized_weighted_tda-powers{}.pdf".format(mc_iterations))
plt.show()
plt.cla()
#plt.clf()
#plt.close()
print("running {} mc simulations for the cvb approach".format(mc_iterations))
test_powers_alpha = compute_test_powers(alphas, amplitudes, mc_iterations, metric = "l1", method = "cvb")
#print(test_powers_alpha.shape)
np.savetxt("{}iterations_l1-metric_cvb_test_powers.txt".format(mc_iterations),np.array(test_powers_alpha))

#f,ax = plt.subplots(figsize = (len(amplitudes),len(alphas)))
plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha, "cvb", ax=ax)
plt.savefig("normalized_weighted_cvb-powers{}.pdf".format(mc_iterations))
plt.show()
plt.clf()
plt.close()