import numpy as np
from scipy.fft import fft
from collections import defaultdict
from scipy.signal import find_peaks

def constant_region_analysis(ratio):
    tol = 1e-8      # adjust if needed

    regions = []

    start = 0
    for i in range(1, len(ratio)):
        if abs(ratio[i]-ratio[i-1]) > tol:
            regions.append((start, i-1))
            start = i

    regions.append((start, len(ratio)-1))

    # print("Constant regions")

    # for s,e in regions:
    #     print(f"{s:6d} -> {e:6d}   length={e-s+1:5d}   value={ratio[s]:.8f}")
    
    return ratio

def dip_analysis(ratio):

    dip_idx, _ = find_peaks(-ratio)

    # print("Dip locations")

    # for d in dip_idx:
    #     print(d, ratio[d])

    return dip_idx

def dip_depth(ratio, dip_idx):
    dip_depths = []
    dict = defaultdict(list)
    for d in dip_idx:

        left = ratio[d-1] if d>0 else ratio[d]
        right = ratio[d+1] if d<len(ratio)-1 else ratio[d]

        local_top = max(left,right)

        depth = local_top-ratio[d]

        dip_depths.append(depth)

        dict[depth].append(d)
    max_depth = max(dict)
    return dict[max_depth]
    # dict2 = defaultdict(list)
    # for key, value in dict.items():
        # value_diff = np.unique(np.diff(value))

        # dict2[key].append(value_diff)
    
    # print(dict)
    # print(np.array(dip_depths))

def local_max_analysis(ratio, dip_idx, window = 20):
    # window = 20

    depths = []

    for d in dip_idx:

        s = max(0,d-window)
        e = min(len(ratio),d+window)

        top = np.max(ratio[s:e])

        depths.append(top-ratio[d])

def diff_btw_dips_analysis(dip_idx):
    periods = np.diff(dip_idx)

    print(periods)

    print("Mean period =", periods.mean())

def piano_block_heights(ratio):
    peak_idx, _ = find_peaks(ratio)

    tops = ratio[peak_idx]

    print(np.unique(np.round(tops,8)))

def oscillation_block_analysis(ratio):
    diff = np.abs(np.diff(ratio))

    moving = diff > 1e-8

    blocks = []

    inside = False

    for i,val in enumerate(moving):

        if val and not inside:
            start=i
            inside=True

        if inside and (not val):
            end=i
            blocks.append((start,end))
            inside=False

    # print(blocks)

    return blocks

def measure_piano_blocks(ratio, blocks):
    for s,e in blocks:

        r = ratio[s:e]

        print()

        print("Block")

        print("start =",s)

        print("end   =",e)

        print("length=",e-s)

        print("top   =",np.max(r))

        print("bottom=",np.min(r))

        print("mean  =",np.mean(r))

        print("std   =",np.std(r))

def oscillation_amplitude(ratio, blocks):
    amplitude = []

    for s,e in blocks:

        r = ratio[s:e]

        amplitude.append(np.max(r)-np.min(r))

    print(amplitude)

def frequency_analysis(ratio):

    x = ratio - ratio.mean()

    Y = np.abs(fft(x))

    freq = np.fft.fftfreq(len(x))

    positive = freq > 0

    freq = freq[positive]
    Y = Y[positive]

    dominant = freq[np.argmax(Y)]

    print("Dominant frequency =", dominant)
    print("Approximate period =", 1/dominant)

def repeated_piano_pattern(ratio):
    corr = np.correlate(
        ratio-ratio.mean(),
        ratio-ratio.mean(),
        mode='full'
    )

    corr = corr[len(corr)//2:]

    peaks,_ = find_peaks(corr)

    print(peaks[:20])

def cluster_unique_ratios(ratio):
    vals = np.round(ratio,8)

    unique, counts = np.unique(vals, return_counts=True)

    for v,c in zip(unique,counts):
        print(v,c)

    # min_ratio = min(unique)
    # max_ratio = max(unique)

    return unique

def analyze_ratio(ratio_list):
    ratio = np.array(ratio_list)
    ratios = cluster_unique_ratios(ratio)
    # constant_region_analysis(ratio)
    # dip_idx = dip_analysis(ratio)
    # max_dip_idx = dip_depth(ratio, dip_idx)
    # local_max_analysis(ratio, dip_idx)
    # diff_btw_dips_analysis(dip_idx)
    # piano_block_heights(ratio)
    # blocks = oscillation_block_analysis(ratio)
    # measure_piano_blocks(ratio, blocks)
    # oscillation_amplitude(ratio, blocks)
    # frequency_analysis(ratio)
    # repeated_piano_pattern(ratio)
    dict = defaultdict(list)
    for rat in ratios:
        indices = [idx for idx, r in enumerate(ratio) if np.round(r, 8) == np.round(rat, 8)]
        dict[rat] = indices
    return dict
    # max_idx = [idx for idx, rat in enumerate(ratio) if np.round(rat, 8) == max_ratio]

    # return min_idx, max_idx