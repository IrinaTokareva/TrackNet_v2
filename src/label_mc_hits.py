import pandas as pd
import numpy as np
import warnings
import plac
import time
import os

from glob import glob
from tqdm import tqdm

from timing import timeit

warnings.simplefilter(action='ignore', category=FutureWarning)
tqdm.pandas()


def euclidean(x, y):
    '''Compute the euclidean distance 
    between two vectors or 2d arrays
    '''
    return np.sum(np.square(x-y), axis=-1)


def brute_force_1nn(x, y, return_distance=False):
    '''K-nearest neighbors algorithm for k=1
    using brute-force
    '''
    distances = [None]*len(y)
    indices = [None]*len(y)
    for i, y_ in enumerate(y):
        dist = euclidean(x, y_)
        # add the smallest distance
        indices[i] = np.argmin(dist)
        distances[i] = dist[indices[i]]

    if return_distance:
        return indices, distances
    # else return only indices
    return indices


def match_hits_to_mc_points(event_id,
                            event_data,
                            event_mc,
                            verbose=False):
    '''For each hit finds the most closest Monte-Carlo
    point and assigns to it's track_id to this hit.
    
    Note: 
        one hit may contain two mc_points, 
        such hit will be duplicated

    # Arguments
        event_id: int, identifier of the event
        event_data: pd.DataFrame, hits
        event_mc: pd.DataFrame, Monte-Carlo points

    # Returns
        array with shape=(N, 5), where columns are:
        event_id, x, y, z, track_id
    '''
    # Faster than KD-Tree and avoids dynamic allocations
    usecols = ['x', 'y', 'z']
    event_hits = event_data[usecols].values
    event_mc_points = event_mc[usecols].values

    # TODO: drop events, where distance is too high
    ii, dist = brute_force_1nn(event_hits, event_mc_points, return_distance=True)

    if np.max(dist) > 1:
        if verbose:
            print('\nEvent id: %d, max distance: %.2f' % (event_id, np.max(dist)))
            print('Drop this event')
        raise ValueError

    # indices of hits without track_id
    ii_ = set(range(len(event_hits))) - set(ii)
    ii_ = list(ii_)
    # event_id, x, y, z, station, track_id
    # track_id = -1, if fake hit
    result_array = np.full((len(ii)+len(ii_), 6), -1, dtype=np.float32)
    # add event_id
    result_array[:, 0] = event_id
    # get hits with track_ids
    result_array[:len(ii), 1:5] = event_data[usecols+['station']].values[ii]
    result_array[:len(ii), -1] = event_mc.track
    # get the remaining hits
    result_array[len(ii):, 1:5] = event_data[usecols+['station']].values[ii_]
    return result_array


@timeit
def read_mc_file(mc_fpath, sep='\t', index_col=None, encoding='utf-8'):
    '''Reads file 'mc_fpath'
    
    # Return
        pandas.DataFrame{event, track, x, y, z, station}
    '''
    usecols = ['event', 'track', 'x_in', 'y_in', 'z_in', 'station']
    dtypes = [np.int32, np.int32, np.float32, np.float32, np.float32, np.int32]
    # read dataframe
    df = pd.read_csv(mc_fpath, sep=sep, encoding=encoding, index_col=index_col,
                     usecols=usecols, dtype=dict(zip(usecols, dtypes)))
    # rename columns
    df = df.rename(columns={'x_in': 'x', 'y_in': 'y', 'z_in': 'z'})
    return df


@timeit
def read_hits_file(hits_fpath, sep='\t', index_col=0, encoding='utf-8'):
    '''Reads file 'mc_fpath'
    
    # Return
        pandas.DataFrame{event, x, y, z, station}
    '''
    return pd.read_csv(hits_fpath,
                       sep=sep,
                       encoding=encoding,
                       index_col=index_col,
                       dtype={'event': np.int32,
                              'x': np.float32,
                              'y': np.float32,
                              'z': np.float32,
                              'station': np.int32})


@timeit
def drop_short_tracks(mc_df, hits_df, n_points=3):
    gp_size = mc_df.groupby(['event', 'track']).size()
    # extract groups with size more than 2
    gp = gp_size[gp_size >= n_points]
    # create multiindex
    mc_df = mc_df.set_index(['event', 'track'])
    # mask dataframe
    mc_df = mc_df.loc[gp.index].reset_index()
    # after cleaning some events may be fully removed, 
    # so remove also in hits_df
    removed_events = set(hits_df.event) - set(mc_df.event)
    hits_df = hits_df[~hits_df.event.isin(removed_events)]
    return mc_df, hits_df


@timeit
def drop_spinning_tracks(mc_df, n_points=1):
    gp_size = mc_df.groupby(['event', 'track']).station.value_counts()
    # exclude tracks with more than n_points per station
    gp = gp_size[gp_size > n_points]
    # create multiindex
    mc_df = mc_df.set_index(['event', 'track'])
    # mask dataframe
    idx = gp.index.droplevel('station').unique()
    mask = mc_df.index.isin(idx)
    return mc_df[~mask].reset_index()


@timeit
def drop_events_by_hits_number(mc_df, hits_df, n_hits=10):
    gp_size = hits_df.groupby(['event', 'station']).size()
    # exclude events with more than n_hits per station 
    gp = gp_size[gp_size > n_hits]
    event_ids = gp.index.get_level_values('event')
    # exclude selected events
    mc_df = mc_df[~mc_df.event.isin(event_ids)]
    hits_df = hits_df[~hits_df.event.isin(event_ids)]
    return mc_df, hits_df



@timeit
def label_hits(mc_df, hits_df):
    # TODO: add station to the file
    hits_with_track_id = []

    for event_id, event_data in tqdm(hits_df.groupby('event')):
        # extract event_mc_points
        event_mc_points = mc_df[mc_df.event==event_id]

        try:
            # match points to hits
            matched = match_hits_to_mc_points(
                event_id, event_data, event_mc_points)
        except ValueError:
            continue

        hits_with_track_id.extend(matched)

    # create dataframe
    hits_with_track_id_df = pd.DataFrame(hits_with_track_id,
                                         columns=['event', 'x', 'y', 'z', 'station', 'track'])

    # data types conversion
    hits_with_track_id_df = hits_with_track_id_df.astype({
        'event': np.int32,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'station': np.int32,
        'track': np.int32})

    return hits_with_track_id_df


@timeit
def merge_mc_with_hits(mc_fpath, hits_fpath):
    print("1. Read data...")
    mc_df = read_mc_file(mc_fpath)
    hits_df = read_hits_file(hits_fpath)
    print("Event number: %d" % hits_df.event.nunique())

    #print("2. Remove events with anomalous number of hits")
    #mc_df, hits_df = drop_events_by_hits_number(mc_df, hits_df)

    print("2. Remove tracks containing less than 3 points")
    mc_df, hits_df = drop_short_tracks(mc_df, hits_df)
    print("Event number: %d" % hits_df.event.nunique())

    print("3. Drop spinning tracks")
    mc_df = drop_spinning_tracks(mc_df)
    print("Event number: %d" % hits_df.event.nunique())

    print("4. Set labels to hits")
    hits_with_track_id_df = label_hits(mc_df, hits_df)
    print("Event number: %d" % hits_with_track_id_df.event.nunique())

    return hits_with_track_id_df


@timeit
@plac.annotations(
    datapath=("Path to the directory with root files", "positional", None, str))
def main(datapath):
    mc_files = sorted(glob(os.path.join(datapath, 'evetest*')))
    hits_files = sorted(glob(os.path.join(datapath, 'bmndst*')))

    for mc_fpath, hits_fpath in zip(mc_files, hits_files):
        print("Files:\n\t%s\n\t%s" % (mc_fpath, hits_fpath))
        df = merge_mc_with_hits(mc_fpath, hits_fpath)
        # create name of the file to save
        save_fname = os.path.split(mc_fpath)[1]
        save_fname = save_fname.split("evetest_")[1]
        save_fname = os.path.join(datapath, save_fname)
        # save to the same location with different name
        print("Save data into `%s`" % save_fname)
        df.to_csv(save_fname, encoding='utf-8', index=None, sep='\t')
        print("--- OK ---\n")


if __name__ == "__main__":
    plac.call(main)