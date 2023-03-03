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
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import levy_stable, kurtosis
from scipy.signal import spectrogram
from symulacja_py import impsim
from tqdm import tqdm

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

#generate data under null, i.e. only noise
alphas = np.linspace(1.6,2,9)

amplitude = 0
mc_iterations = 1000
avg_bcs = []
acc_threshs = np.zeros_like(alphas, dtype = float)

dim = 3*833#417# half period
delay = 50000//dim
#print("delay", delay)
skip = 1#200#0#100
#print(dim*delay)
#print(len(data[0])/24)

#start = min([np.min(d) for d in pds])-0.005
#end = max([np.max(d) for d in pds])+0.005
#print(start,end)
start = 2.98
end = 3.7
grid = np.linspace(start,end,10000)



def generate_data_compute_bc(alpha, amplitude, seed, dim, delay, skip, normalize = True, weighted = True):
    x = create_signal(alpha, amplitude, seed)
    freqs, t, Pxx = spectrogram(x, fs=25000, nfft=512, window="hamming", nperseg= 256, noverlap= int(np.floor(0.85*256)),detrend = False, mode="magnitude")
    return kurtosis(np.abs(Pxx), axis=1, fisher=False)
    
    
metric = "l1"
avg_bcs=[]
for i in tqdm(range(0,len(alphas))):
    alpha=alphas[i]
    betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, 0, k, dim, delay, skip) for k in range(0,mc_iterations))
    avg_bc = np.mean(betti_curves, axis=0)
    avg_bcs.append(avg_bc)
    new_betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, 0, mc_iterations+k, dim, delay, skip) for k in range(0,mc_iterations))
    if metric == "l1":
        acc_thresh = np.quantile(pairwise_distances([avg_bc],new_betti_curves, "minkowski", p=1, n_jobs=-1)[0],0.95)
    elif metric == "max":
        acc_thresh = np.quantile(pairwise_distances([avg_bc],new_betti_curves, "chebyshev",n_jobs=-1)[0],0.95)
    acc_threshs[i] = acc_thresh

amplitudes = np.linspace(5,9,9)
test_powers_alpha = np.zeros((len(alphas),len(amplitudes)), dtype=float)
for j in range(0,len(amplitudes)):
    for i in tqdm(range(0,len(alphas))):
        amp = amplitudes[j]
        alpha=alphas[i]
        #print("alpha = ",alpha,", amp = ", amp)
        seed = int(2*mc_iterations*(amp+1))
        test_betti_curves = Parallel(n_jobs=-1)(delayed(generate_data_compute_bc)(alpha, amp, seed+k, dim, delay, skip) for k in range(0,mc_iterations))
        if metric == "l1":
            dists = pairwise_distances([avg_bcs[i]],test_betti_curves, "minkowski",p=1,n_jobs=-1)[0]
        elif metric == "max":
            dists = pairwise_distances([avg_bcs[i]],test_betti_curves, "chebyshev",n_jobs=-1)[0]
        #print("computing test power")
        power = np.sum(dists>acc_threshs[i])/mc_iterations
        test_powers_alpha[i][j] =power

f,ax = plt.subplots(figsize = (len(amplitudes),len(alphas)))
ax.imshow(test_powers_alpha)

ax.set_xticks(range(0,len(amplitudes)), labels=np.round(amplitudes,1))
ax.set_yticks(range(0,len(alphas)), labels = np.round(alphas,2))

ax.set_xlabel("amplitude")
ax.set_ylabel("alpha")

plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(alphas)):
    for j in range(len(amplitudes)):
        text = ax.text(j, i, test_powers_alpha[i][j],
                       ha="center", va="center", color="w")


#ax.scatter(np.linspace(0,20,21), 19*(-1.1+(1.7/((0.1*np.linspace(0,20,21))**(0.2)))), color="red")
plt.title("{} Kurtosis Test power estimated via {} MC iterations".format(metric, mc_iterations))
plt.savefig("NEW-alpha-amplitudes-magnitude-kurt-GoF-test-powers{}.pdf".format(mc_iterations))
plt.show()

np.savetxt("{}iterations_{}-metric_magnitude-kurt-test_powers_alpha1.1-2_amplitude0-2.txt".format(mc_iterations,metric),np.array(test_powers_alpha))
