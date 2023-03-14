
import numpy as np
import gudhi as gd
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.point_cloud.timedelay import TimeDelayEmbedding
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.representations import BettiCurve
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import levy_stable
from symulacja_py import impsim
from tqdm import tqdm


def compute_diagram(data, dim, delay, skip, normalize = True, weighted = True, point_cloud_size = 100):
    tde = TimeDelayEmbedding(dim = dim, delay=delay, skip=skip)
    point_clouds = tde.transform([data])[0]
    if len(point_clouds>250):
        skip = max(len(point_clouds)//250,1)
        tde = TimeDelayEmbedding(dim = dim, delay=delay, skip=skip)
        point_clouds = tde.transform([data])[0] 
    #point_clouds = levy_stable.rvs(alpha,0,0, size=(100,2))

    if (normalize):
        point_clouds = point_clouds-np.mean(point_clouds,1)[:, None]
        point_clouds = point_clouds/np.sqrt(np.sum(point_clouds**2, 1))[:, None]

    pc = point_clouds
    #print(len(pc))

    if weighted:
        dist = cdist(pc,pc)
        dtm = DistanceToMeasure(5, dim = 10, q=2, metric="precomputed")
        r = dtm.fit_transform(dist)
        ac = WeightedRipsComplex(distance_matrix=dist,weights = 1/r)
    else:
        ac = gd.RipsComplex(points=pc)
    
    st = ac.create_simplex_tree(max_dimension = 2)
    st.compute_persistence()
    pd = st.persistence_intervals_in_dimension(1)
    return pd

def create_signal(alpha, impulse_amplitude, seed):
    fs = 25000
    fmod = 30
    f_center = 5000
    bandwidth = 1500
    shift = 0

    amp_imp = impulse_amplitude
    alfa = alpha
    skala = 1

    varsize = 1*fs
    tt_ts = np.linspace(1/fs,varsize/fs,varsize)
    #szum = np.random.normal(0,skala,varsize) #szum Gaussowski

    szum = levy_stable.rvs(alfa,0,0,skala,varsize, random_state=seed)  #szum alfa-stabilny

    signal_l = 2*impsim(fs,varsize,fmod,amp_imp,f_center,bandwidth,shift)+szum
    return signal_l

def generate_data_compute_bc(alpha, amplitude, seed, dim, delay, skip, normalize = True, weighted = False, method = "tda"):
    if method == "tda":

        data = create_signal(alpha, amplitude, seed)
        #dim * delay should roughly equal len(time_series)/numer_of_periods
        dim = 3*833#417# half period
        delay = len(data)//dim
        #print("delay", delay)
        skip = 1#200#0#100
        #print(dim*delay)
        #print(len(data[0])/24)

        #print("===============computing SWE====================")
        tde = TimeDelayEmbedding(dim = dim, delay=delay, skip=skip)
        point_clouds = tde.transform([data])[0]
        #point_clouds = levy_stable.rvs(alpha,0,0, size=(100,2))

        if (normalize):
            point_clouds = point_clouds-np.mean(point_clouds,1)[:, None]
            point_clouds = point_clouds/np.sqrt(np.sum(point_clouds**2, 1))[:, None]

        pc = point_clouds
        if weighted:
            dist = cdist(pc,pc)
            dtm = DistanceToMeasure(5, dim = 10, q=2, metric="precomputed")
            r = dtm.fit_transform(dist)
            ac = WeightedRipsComplex(distance_matrix=dist,weights = 1/r)
        else:
            ac = gd.RipsComplex(points=pc)

        st = ac.create_simplex_tree(max_dimension = 2)
        st.compute_persistence()
        pd = st.persistence_intervals_in_dimension(1)

        bc = BettiCurve(predefined_grid=grid)
        betti_curve = bc.fit_transform([pd])
        return betti_curve[0]
    
    elif method == "cvb":
        x = create_signal(alpha, amplitude, seed)
        freqs, t, Pxx = spectrogram(x, fs=25000, nfft=512, window="hamming", nperseg= 256, noverlap= int(np.floor(0.85*256)),detrend = False, mode="magnitude")
        return np.array([cvb(np.abs(Pxx[i])) for i in range(0,len(freqs))])

def compute_tests_powers(alphas, amplitudes, mc_iterations, metric = "l1", method = "tda"):
    acc_threshs = np.zeros_like(alphas, dtype = float)

    avg_bcs=[]
    for i in tqdm(range(0,len(alphas))):
        alpha=alphas[i]
        betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, 0, k, dim, delay, skip, method = method) for k in range(0,mc_iterations))
        avg_bc = np.mean(betti_curves, axis=0)
        avg_bcs.append(avg_bc)
        new_betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, 0, mc_iterations+k, dim, delay, skip, method = method) for k in range(0,mc_iterations))
        if metric == "l1":
            acc_thresh = np.quantile(pairwise_distances([avg_bc],new_betti_curves, "minkowski", p=1, n_jobs=-1)[0],0.95)
        elif metric == "max":
            acc_thresh = np.quantile(pairwise_distances([avg_bc],new_betti_curves, "chebyshev",n_jobs=-1)[0],0.95)
        acc_threshs[i] = acc_thresh

    test_powers_alpha = np.zeros((len(alphas),len(amplitudes)), dtype=float)
    for j in range(0,len(amplitudes)):
        for i in tqdm(range(0,len(alphas))):
            amp = amplitudes[j]
            alpha=alphas[i]
            #print("alpha = ",alpha,", amp = ", amp)
            seed = int(2*mc_iterations*(amp+1))
            test_betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, amp, seed+k, dim, delay, skip, method=method) for k in range(0,mc_iterations))
            if metric == "l1":
                dists = pairwise_distances([avg_bcs[i]],test_betti_curves, "minkowski",p=1,n_jobs=-1)[0]
            elif metric == "max":
                dists = pairwise_distances([avg_bcs[i]],test_betti_curves, "chebyshev",n_jobs=-1)[0]
            #print("computing test power")
            power = np.sum(dists>acc_threshs[i])/mc_iterations
            test_powers_alpha[i][j] =power
    return test_powers_alpha

def plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha):
    f,ax = plt.subplots(figsize = (len(amplitudes),len(alphas)))
    plt.rc('font', **{'size'   : 20})
    plt.rcParams["axes.labelsize"] = 20
    sns.heatmap(test_powers_alpha, annot=True, cbar=False)

    ax.set_yticklabels(np.round(alphas,2),  fontsize = 16)
    ax.set_xticklabels(np.round(amplitudes,1),  fontsize = 16)

    ax.set_xlabel("amplitude",  fontsize = 20)
    ax.set_ylabel("alpha",  fontsize = 20)



    #ax.scatter(np.linspace(0,20,21), 19*(-1.1+(1.7/((0.1*np.linspace(0,20,21))**(0.2)))), color="red")
    plt.title("TDA Test power estimated via {} MC iterations".format(mc_iterations))
    sns.set(font_scale=50)
    plt.savefig("normalized_unweighted_topotest-powers{}.pdf".format(mc_iterations))
    plt.show()