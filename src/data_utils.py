import pandas as pd
import numpy as np
import json
import os

from collections import defaultdict
from glob import glob
from tqdm import tqdm
tqdm.pandas()

from timing import timeit


def read_vertex_file(path):
    with open(path) as f:
        return json.load(f)


def sample_vertices(vertex_stats, n, random_seed=None):
    rs = np.random.RandomState(seed=random_seed)
    # sample vertex X coordinate
    vertex_x = rs.normal(
        loc=vertex_stats['x']['mean'],
        scale=vertex_stats['x']['std'],
        size=(n, 1))
    # sample vertex Y coordinate
    vertex_y = rs.normal(
        loc=vertex_stats['y']['mean'],
        scale=vertex_stats['y']['std'],
        size=(n, 1))
    # sample vertex Z coordinate
    vertex_z = rs.uniform(
        low=vertex_stats['z']['left'],
        high=vertex_stats['z']['right'],
        size=(n, 1))
    # return stacked coordinates
    return np.hstack([vertex_x, vertex_y, vertex_z])


def get_tracks_with_vertex(df, vertex_stats=None, random_seed=13, train_split=None):
    # extract tracks groups

    groupby = df.groupby(['event', 'track'])
    n_stations = groupby.size().max()
    n_stations_min = groupby.size().min()
    bins = None

    # if we are splitting tracks by lengths distributions,
    # initialize bins with [min_tracks]*n_stations
    if train_split:
        is_equal_distr = len(set(train_split)) == 1
        v_cnts = groupby.size().value_counts()
        if not is_equal_distr:
            # assert False and "TODO: custom split"
            bins = [v_cnts[val] if val >= (n_stations + 1 - len(train_split)) else 0 for val in range(n_stations + 1)]
            for ind, val in enumerate(range((n_stations + 1 - len(train_split)), n_stations+1)):
                if train_split[ind] != -1:
                    bins[val] *= train_split[ind]
        else:
            min_val = v_cnts.min()
            if train_split[0] == -1:
                bins = [v_cnts[station] if station in v_cnts else 0 for station in range(n_stations + 1)]
            else:
                bins = [ min_val if station in v_cnts else 0 for station in range(n_stations + 1)]


    # create result array
    count_of_tracks = np.sum(bins) if bins else groupby.ngroups
    res = np.zeros((count_of_tracks, n_stations, 3))

    if vertex_stats is not None:
        # sample vertex data
        vertex = sample_vertices(vertex_stats, count_of_tracks, random_seed)
        # vertex + n_stations 3D hits
        res = np.zeros((count_of_tracks, n_stations + 1, 3))
        # fill with vertex data
        res[:, 0] = vertex

    # get tracks
    tracks = groupby[['x', 'y', 'z', 'station']].progress_apply(pd.Series.tolist)

    # fill result array
    ind = 0
    broken_cnt = [0]*(n_stations + 1)
    #if bins:
    # TODO: shuffle with info about track_id and event_id
    #np.random.RandomState(seed=random_seed).shuffle(tracks.values)
    if bins:
        copy_bins = np.copy(bins)
    copy_start = 0 if vertex_stats is None else 1
    for i, track in enumerate(tqdm(tracks)):
        track_len = len(track)
        if bins:
            if bins[track_len] > 0:
                # TODO: drop tracks with other way (may be on preprocessing)
                nparray = np.asarray(track)
                if np.all(np.diff(nparray[:, 3]) == 1.) and nparray[:, 3][0] == 0.:
                    res[ind, copy_start:track_len + copy_start] = nparray[:, :3]
                    bins[track_len] -= 1
                    ind += 1
                else:
                    broken_cnt[track_len] += 1
        else:
            nparray = np.asarray(track)
            if np.all(np.diff(nparray[:, 3]) == 1.) and nparray[:, 3][0] == 0.:
                res[ind, 1:track_len + 1] = nparray[:, :3]
                bins[track_len] -= 1
                ind += 1
            else:
                broken_cnt[track_len] += 1
    if bins:
        # this space is left in the res because of appearance of broken tracks
        gap = np.sum(bins)
        if gap > 0:
            res = res[:-gap]
    print("\n Total tracks:", (copy_bins if bins else groupby.size().value_counts()))
    print(" Total broken tracks:", broken_cnt, '\n')
    return res


@timeit
def read_train_dataset(dirpath, vertex_fname=None, random_seed=13, debug=False, train_split=None):
    '''Reads data from directory. Directory must have 
    the following structure:

        dir
        |__file1.tsv
        |__file2.tsv
        |...
        |__fileN.tsv

    where .tsv files contains events hits
    or with corresponding track labels
    '''
    # collect files
    train_files = glob(os.path.join(dirpath, '*.tsv'))

    vertex_stats = None
    if vertex_fname is not None:
        # get vertex statistics
        vertex_file = os.path.join(dirpath, vertex_fname)
        vertex_stats = read_vertex_file(vertex_file)

    # get train data
    length = [None]*len(train_files)
    train = [None]*len(train_files)
    for i, f in enumerate(train_files):
        print("Processing `%s` file..." % f)
        df = pd.read_csv(f, encoding='utf-8', sep='\t')
        # extract only true tracks
        df = df[df.track != -1]
        if debug and debug['debug_size']:
            df = df[df.event < debug['debug_size']]
        # get true tracks array (N, M, 3)
        train[i] = get_tracks_with_vertex(df, vertex_stats, random_seed, train_split)
        length[i] = len(train[i])

    # create result array
    N = sum(length)
    # empty is faster than zeros
    train_arr = np.empty((N, *train[0].shape[1:]), dtype=np.float32)

    # record all to train_arr
    left_idx = 0
    right_idx = 0
    while train:
        right_idx = left_idx + length.pop()
        train_arr[left_idx:right_idx] = train.pop()
        left_idx = right_idx

    return train_arr


def shuffle_arrays(*args, random_seed=None):
    '''Shuffles input lists of arrays if they are equal length

    # Arguments
        args : lists or numpy arrays
        random_seed : int, seed for the RandomState

    # Returns 
        shuffled arrays
    '''
    assert len(set(len(x) for x in args)) == 1, 'Arrays must be the same size'
    # create seed for random generator
    rs = np.random.RandomState(seed=random_seed)
    idx = rs.permutation(len(args[0]))

    if len(args) == 1:
        return args[0][idx]
    # else
    return [x[idx] for x in args]


def train_test_split(data, shuffle = False, test_size=0.3, random_seed=13):
    # shuffle original array
    if shuffle:
        data = shuffle_arrays(data, random_seed=random_seed)
    # calc split index
    idx = int(len(data)*(1-test_size))
    return data[:idx], data[idx:]


def get_part(X, n_hits):
    '''Cuts the input array X with the shape of (N, S, 3) with 
    S points to the n_hits points and set the n_hits+1 point as target.
    Also, concatenates the X[:, i+1, 2], which is actually a z+1 coordinate,
    to the array

    # Arguments
        X : ndarray with dtype=float32 with shape of size 3
        n_hits : int, number of hits to cut off
    '''
    assert len(X.shape) == 3, "Input array must be 3-dimensional, but got {}".format(X.shape)
    assert X.shape[1] > 2, "Input shape is {}, second dimension must be greater than 2".format(X.shape)
    assert X.shape[2] == 3, "Input shape is {}, third dimension must be 3 for (x, y, z) coords".format(X.shape)
    assert n_hits < X.shape[1], "Number of hits to cut off must be smaller than the actual number of hits(%d), but got %d" % (X.shape[1], n_hits)
    x_part = np.zeros(shape=(*X.shape[:2], 4))
    # :2 - we take only (x, y) coords
    target = X[:, n_hits, :2].copy()
    # cut to n_hits
    x_part[:, :n_hits, :3] = X[:, :n_hits]
    # z+1 point
    x_part[:, :n_hits, 3] = X[:, 1:n_hits+1, 2]
    return (x_part, target)


def split_on_buckets(X, shuffle=False, random_seed=None):
    '''Data may contains from tracks with varying lengths.
    To prepare a train dataset in a proper way, we have to 
    split data on so-called buckets. Each bucket includes 
    tracks based on their length, as we can't predict the 
    6'th point of the track with length 4, but we can predict
    3-d point

    # Arguments
        X: ndarray with dtype=float32 with shape of size 3
        random_seed: int, seed for the RandomState
        shuffle: boolean, whether or not shuffle output dataset

    # Returns
        dict {n_points: data_idx}, train_data_size
    '''
    # create random generator
    rs = np.random.RandomState(random_seed)
    # length of each training sample
    # count number of nonzero rows
    tracklens = np.sum(np.count_nonzero(X, axis=-1) > 0, axis=1)
    # all existing track lengths
    utracklen = np.unique(tracklens)
    # result dictionary
    buckets = defaultdict(list)
    # data split by tracks length
    subbuckets = {x: np.where(tracklens==x)[0] for x in utracklen}
    # maximum track length
    maxlen = np.max(utracklen)
    # approximate size of the each bucket
    bsize = len(X) // (maxlen-2)
    # set index
    k = maxlen
    # reverse loop until two points
    for n_points in range(maxlen, 2, -1):
        # while bucket is not full
        while len(buckets[n_points]) < bsize:
            if shuffle:
                rs.shuffle(subbuckets[k])
            # if we can't extract all data form subbucket
            # without bsize overflow
            if len(buckets[n_points])+len(subbuckets[k]) > bsize:
                n_extract = bsize - len(buckets[n_points])
                # extract n_extract samples
                buckets[n_points].extend(subbuckets[k][:n_extract])
                # remove them from original subbucket
                subbuckets[k] = subbuckets[k][n_extract:]
            else:
                buckets[n_points].extend(subbuckets[k])
                # remove all data from the original list
                subbuckets[k] = []
                # decrement index
                k -= 1

    return buckets, bsize*len(buckets)


def get_dataset(X, shuffle=False, random_seed=None):
    '''Creates a dataset to train the tracknet model. 
    The input is cropped tracks and target is the next point in track

    # Arguments
        X: ndarray with dtype=float32 with shape of size 3
        random_seed: int, seed for the RandomState
        shuffle: boolean, whether or not shuffle output dataset
    '''
    # create pseudo random generator 
    rs = np.random.RandomState(seed=random_seed)
    # shuffle input array
    if shuffle:
        rs.shuffle(X)
    # split array on buckets with tracks of different length
    buckets, size = split_on_buckets(X, shuffle=shuffle, random_seed=random_seed)
    # create vars for resulting arrays
    inputs = np.zeros(shape=(size, X.shape[1], 4))
    target = np.zeros(shape=(size, 2))
    # create parts
    for i, (n_hits, idx) in enumerate(buckets.items()):
        n_hits -= 1
        # calculate indices
        i_s = i*len(idx)       # start index 
        i_e = (i+1)*len(idx)   # end index
        # create part
        inputs[i_s:i_e], target[i_s:i_e] = get_part(X[idx], n_hits=n_hits)
    # shuffle arrays
    if shuffle:
        inputs, target = shuffle_arrays(inputs, target, random_seed=random_seed)

    return (inputs, target)