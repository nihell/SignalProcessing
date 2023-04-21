
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
from scipy.signal import spectrogram



def compute_diagram(data, dim, delay, skip, normalize = True, weighted = True, point_cloud_size = 100):
    """Compute 1d persistent homology from time series

    Uses time delay embedding (normalized by default) and weighted VR complex on a subsample.

    Parameters
    ----------
    data : numpy.array
        time series
    dim : int
        embedding dimension
    delay : int
        delay for time delay embedding
    skip : int
        _description_
    normalize : bool, optional
        normalize to unit sphere after time delay embedding, by default True
    weighted : bool, optional
        use weighted rips, by default True
    point_cloud_size : int, optional
        size of the subsample, by default 100

    Returns
    -------
    gudhi.PersistenceDiagram
        1d persistence diagram 
    """
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
    """creates noisy time series
    cyclic impulses with specified amplitude with added noise, possibly non gaussian
    Parameters
    ----------
    alpha : float
        between 1 and 2: parameter of alpha-stable distribution of the noise. 2-> gaussian
    impulse_amplitude : float
        amplitude of the cyclic impulses
    seed : int
        seed for the noise

    Returns
    -------
    numpy.array
        time series
    """
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

def cvb(S):
    """conditional variance

    Parameters
    ----------
    S : np.array
        absolute values in frequency band

    Returns
    -------
    float
        conditional variance of S
    """
    quantiles = np.concatenate([np.array([float("-inf")]),
                          np.array([np.quantile(S, q) for q in [0.004,
                          0.062,
                          0.308,
                          0.692,
                          0.938,
                          0.996]]),
                        np.array([float("inf")])])
    quantiles_partition = [S[(S>quantiles[i]) & (S <= quantiles[i+1])] for i in range(0,len(quantiles)-1)] 
    C1 = ((np.var(quantiles_partition[2])-np.var(quantiles_partition[3]))/np.var(S) + (np.var(quantiles_partition[4])-np.var(quantiles_partition[3]))/np.var(S))**2
    return C1*np.sqrt(len(S))

def cvb_selector(x):
    """conditional variance based selector

    Introduced by Hebda-Sobkowicz, J.; Zimroz, R.; Pitera, M.; Wylomanska, A. Informative frequency band selection in the presence of non-Gaussian noise—A novel approach based on the conditional variance statistic. Mech. Syst. Signal Proc. 2020,
    
    Parameters
    ----------
    x : numpy.array
        time series

    Returns
    -------
    numpy.array
        cvb selector in frequency domain
    """
    freqs, t, Pxx = spectrogram(x, fs=25000, nfft=512, window="hamming", nperseg= 256, noverlap= int(np.floor(0.85*256)),detrend = False, mode="magnitude")
    return np.array([cvb(np.abs(Pxx[i])) for i in range(0,len(freqs))])

def betticurve(x, dim, delay, skip, normalize = True, weighted = True):
    """betti curve from time series

    uses time delay embedding and 1d persistent homology to compute betti curve

    Parameters
    ----------
    x : numpy.array
        time series
    dim : int
        embedding dimension
    delay : int
        delay for time delay embedding
    skip : int
        _description_
    normalize : bool, optional
        normalize to unit sphere after time delay embedding, by default True
    weighted : bool, optional
        use weighted rips, by default True

    Returns
    -------
    numpy.array
        betti curve in the filtration domain
    """
    dim = 3*833
    delay = len(x)//dim

    skip = 1

    start = 0
    end = 5
    grid = np.linspace(start,end,257)
    #print("===============computing SWE====================")
    tde = TimeDelayEmbedding(dim = dim, delay=delay, skip=skip)
    point_clouds = tde.transform([x])[0]
    

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


def infogram():
    """dummy method
    """
    return

def compute_test_powers(alphas, amplitudes, mc_iterations, func, create_data=True, metric = "l1", **kwargs):
    """computes power of statistical test

    For each alpha, consider the null hypothesis of having time series consisting only of alpha-stable noise.
    Under the null, simulate the distribution of the functional summary.
    As a test statistic, use distance between sample and average functional summary.
    Computes power of this test rejecting time series with cyclic impulses of different amplitudes.

    Parameters
    ----------
    alphas : numpy.array
        list of alpha parameters to be considered for noise
    amplitudes : numpy.array
        list of amplitudes of cyclic impulses
    mc_iterations : int
        number of monte carlo iterations to be used in simulations
    func : function
        function, that produces a functional summary as a numpy array from an input time series (as numpy.array)
    create_data : bool, optional
        whether to create the data first. Only needed on first run, by default True
    metric : str, optional
        which metric to use as test statistic: either "l1" or "max", by default "l1"

    Returns
    -------
    numpy.array
        array of shape=(len(alphas),len(amplitudes)) containing the test powers
    """
    acc_threshs = np.zeros_like(alphas, dtype = float)

    avg_curves=[]
    for i in tqdm(range(0,len(alphas))):
        alpha=alphas[i]
        if create_data:
            data = Parallel(n_jobs=-1)(delayed(create_signal)(alpha, 0, k) for k in range(0,mc_iterations))
            np.savetxt("generated_data/alpha{}training0.csv".format(alpha), np.array(data), delimiter=",")
        data = np.loadtxt("generated_data/alpha{}training0.csv".format(alpha), delimiter=",")
        if func == betticurve or func==cvb_selector:
            curves = Parallel(n_jobs=-1)(delayed(func)(x, **kwargs) for x in data)
        elif func == infogram:
            if kwargs["gn"]:
                curves = np.loadtxt("Gener_data2/gninfogram_alpha{}training0.csv".format(alpha), delimiter = ",")
            else:
                curves = np.loadtxt("Gener_data2/infogram_alpha{}training0.csv".format(alpha), delimiter = ",")
        avg_curve = np.mean(curves, axis=0)
        avg_curves.append(avg_curve)
    
        if create_data:
            data = Parallel(n_jobs=-1)(delayed(create_signal)(alpha, 0, mc_iterations+k) for k in range(0,mc_iterations))
            np.savetxt("generated_data/alpha{}training1.csv".format(alpha), np.array(data), delimiter=",")
        data = np.loadtxt("generated_data/alpha{}training1.csv".format(alpha), delimiter=",")
        if func == betticurve or func==cvb_selector:
            new_curves = Parallel(n_jobs=-1)(delayed(func)(x, **kwargs) for x in data)
        elif func == infogram:
            if kwargs["gn"]:
                new_curves = np.loadtxt("Gener_data2/gninfogram_alpha{}training1.csv".format(alpha), delimiter = ",")
            else:
                new_curves = np.loadtxt("Gener_data2/infogram_alpha{}training1.csv".format(alpha), delimiter = ",")
        if metric == "l1":
            acc_thresh = np.quantile(pairwise_distances([avg_curve],new_curves, "minkowski", p=1, n_jobs=-1)[0],0.95)
        elif metric == "max":
            acc_thresh = np.quantile(pairwise_distances([avg_curve],new_curves, "chebyshev",n_jobs=-1)[0],0.95)
        acc_threshs[i] = acc_thresh

    test_powers_alpha = np.zeros((len(alphas),len(amplitudes)), dtype=float)
    for j in range(0,len(amplitudes)):
        for i in tqdm(range(0,len(alphas))):
            amp = amplitudes[j]
            alpha=alphas[i]
            #print("alpha = ",alpha,", amp = ", amp)
            seed = int(2*mc_iterations*(amp+1))
            if create_data:
                data = Parallel(n_jobs=-1)(delayed(create_signal)(alpha, amp, seed+k) for k in range(0,mc_iterations))
                np.savetxt("generated_data/alpha{}_amp{}.csv".format(alpha, amp), np.array(data), delimiter=",")
            data = np.loadtxt("generated_data/alpha{}_amp{}.csv".format(alpha, amp), delimiter=",")
            if func == betticurve or func==cvb_selector:
                test_curves = Parallel(n_jobs=-1)(delayed(func)(x, **kwargs) for x in data)
            elif func == infogram:
                if kwargs["gn"]:
                    test_curves = np.loadtxt("Gener_data2/gninfogram_alpha{}_amp{}.csv".format(alpha,amp), delimiter = ",")
                else:
                    test_curves = np.loadtxt("Gener_data2/infogram_alpha{}_amp{}.csv".format(alpha,amp), delimiter = ",")
            if metric == "l1":
                dists = pairwise_distances([avg_curves[i]],test_curves, "minkowski",p=1,n_jobs=-1)[0]
            elif metric == "max":
                dists = pairwise_distances([avg_curves[i]],test_curves, "chebyshev",n_jobs=-1)[0]
            #print("computing test power")
            power = np.sum(dists>acc_threshs[i])/mc_iterations
            test_powers_alpha[i][j] =power
    return test_powers_alpha

def plot_test_power_matrix(alphas, amplitudes, mc_iterations, test_powers_alpha, method, ax = None):
    """plot test power matrix

    Uses seaborn to plot test powers

    Parameters
    ----------
    alphas : numpy.array
        list of alpha parameters that was considered for noise
    amplitudes : numpy.array
        list of amplitudes of cyclic impulses
    mc_iterations : int
        number of monte carlo iterations used in simulations
    test_powers_alpha : numpy.array
        Test powers, usually as computed by compute_test_powers
    method : string
        name of the method used in title and file name
    ax : matplotlib.pyplot.axis, optional
        axis on which to plot, by default None
    """
    if ax == None:
        ax = plt.gca()
    #
    plt.rc('font', **{'size'   : 20})
    plt.rcParams["axes.labelsize"] = 20
    sns.set(font_scale=1.4)
    sns.heatmap(test_powers_alpha, annot=True, cbar=False, ax=ax)

    ax.set_xlabel("amplitude",  fontsize = 20)
    ax.set_ylabel("alpha",  fontsize = 20)
    
    ax.set_xticks(np.arange(len(amplitudes))+0.5)
    ax.set_yticks(np.arange(len(alphas))+0.5)
    
    ax.set_yticklabels(np.round(alphas,2),  fontsize = 16)
    ax.set_xticklabels(np.round(amplitudes,1),  fontsize = 16)

    #ax.scatter(np.linspace(0,20,21), 19*(-1.1+(1.7/((0.1*np.linspace(0,20,21))**(0.2)))), color="red")
    plt.title("{} Test power estimated via {} MC iterations".format(method,mc_iterations), fontsize = 20)
    #
    return(ax)
    
    