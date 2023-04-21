from TDA_utils_Copy1 import *

### Main script to reproduce simulations on synthetic data as displayed in the paper

alphas = np.linspace(1.6,2,9)
amplitudes = np.linspace(5,9,9)

mc_iterations = 1000

functions = [betticurve, cvb_selector, infogram,infogram]
args = [{"dim":2499, "delay":1, "skip":1}, {}, {"gn":True}, {"gn":False}]
names = ["TDA", "CVB", "GNInfogram", "Infogram"]

create_data=True

for f, a, n  in zip(functions,args,names):
    print(n)
    print("running {} mc simulations".format(mc_iterations))
    test_powers = compute_test_powers(alphas, amplitudes, mc_iterations, f, create_data=create_data, metric = "l1", **a)
    create_data = False
    #print(test_powers_alpha.shape)
    np.savetxt("{}iterations_l1-metric_{}_test_powers.txt".format(mc_iterations, n),np.array(test_powers))

    f,ax = plt.subplots(figsize = (len(amplitudes),len(alphas)))
    plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers, n, ax=ax)
    plt.savefig("{}-powers{}.pdf".format(n,mc_iterations))
    plt.show()
    plt.clf()
    plt.close()
