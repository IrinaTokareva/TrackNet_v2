import itertools
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import yaml
import sys

sys.path.append(os.path.realpath('.'))

from losses import custom_loss
from metrics import point_in_ellipse
from metrics import point_in_ellipse_numpy
from metrics import circle_area
from data_utils import sample_vertices
from data_utils import read_vertex_file
from tensorflow.keras.models import load_model
from visualizer import Visualizer, revert_types
from data_utils import sample_vertices
from metrics import point_in_ellipse_numpy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)


def drop(x):
    return np.all(np.diff(x.station.values) == 1.) and x.station.values[0] == 0.

def dropBroken(df, preserve_fakes = True, drop_full_tracks = False):
    if not preserve_fakes:
        df = df[df.track != -1]
    ret = df.groupby('track', as_index=False).filter(
        lambda x: drop(x) or preserve_fakes and x.track.values[0] == -1
        # if preserve_fakes == False, we are leaving only matched events, no fakes
    )
    if drop_full_tracks:
        ret = ret.groupby('track', as_index=False).filter(
            lambda x: x.station.nunique() < 6 or preserve_fakes and x.track.values[0] == -1)
    return ret

def getEvents(config_df, hits_df):
    eventIdsArr = config_df['event_ids']
    def parseSingleArrArg(arrArg):
        if '..' in arrArg:
            args = arrArg.split('..')
            assert len(args) == 3 and "It should have form '%num%..%num%' ."
            return np.arange(int(args[0]), int(args[2]))
        if ':' in arrArg:
            return -1
        return [int(arrArg)]

    res = np.array([])
    for elem in eventIdsArr:
        toAppend = parseSingleArrArg(elem)
        if toAppend == -1:
            return hits_df
        res = np.append(res, toAppend)

    hits = hits_df[hits_df.event.isin(res)]
    return hits

def parseDf(config_df):
    df_path = config_df['df_path']
    if config_df['read_only_first_lines']:
        nrows = config_df['read_only_first_lines']
        return pd.read_csv(df_path, encoding='utf-8', sep='\t', nrows=nrows)
    return pd.read_csv(df_path, encoding='utf-8', sep='\t')

def visualize_2d(event_df, withNN = False):
    pass

# pnt_to_pred is an array of following shape:
#[
#   [
#       0:[x0,y0,z0,z1]
#       1:[x1,y1,z1,z2]
#       2:[x2,y2,z2,z3]
#       ...
#   ]
#]
def get_nn(nn_config):
    path_to_model = os.path.realpath(os.path.join(os.getcwd(),nn_config['network_path']))
    custom_objects = {
        "_tracknet_loss": custom_loss,
        "point_in_ellipse": point_in_ellipse,
        "circle_area": circle_area}
    print("PATH" + path_to_model)
    return load_model(path_to_model, custom_objects=custom_objects)

def get_seeds_with_index_with_vertex(hits, vertex_stats, stations_z):
    # first station hits
    st0_hits = hits[hits.station==0][['x', 'y', 'z']].values
    vertex = sample_vertices(vertex_stats, n=1)
    # create seeds
    seeds = np.zeros((len(st0_hits), 2, 4))
    seeds[:, 0, :-1] = vertex
    seeds[:, 1, :-1] = st0_hits
    seeds[:, :2, -1] = stations_z[:2]
    return seeds, hits[hits.station==0].index.values.reshape(-1, 1)

def get_seeds_with_index(hits, stations_z):
    st0_hits = hits[hits.station==0][['x', 'y', 'z']]
    st1_hits = hits[hits.station==1][['x', 'y', 'z']]
    # all possible combinations
    idx0 = st0_hits.index.values
    idx1 = st1_hits.index.values
    idx_comb = np.array(np.meshgrid(idx0, idx1)).T.reshape(-1,2)
    # create seeds array
    seeds = np.zeros((len(idx_comb), 2, 4))
    seeds[:, 0, :-1] = st0_hits.loc[idx_comb[:,0]].values
    seeds[:, 1, :-1] = st1_hits.loc[idx_comb[:,1]].values
    seeds[:, :2, -1] = stations_z[1:3]
    return seeds, idx_comb

def get_seeds(hits, stations_z):
    st0_hits = hits[hits.station==0][['x', 'y', 'z']].values
    st1_hits = hits[hits.station==1][['x', 'y', 'z']].values
    # all possible combinations
    idx0 = range(len(st0_hits))
    idx1 = range(len(st1_hits))
    idx_comb = itertools.product(idx0, idx1)
    # unpack indices
    idx0, idx1 = zip(*idx_comb)
    idx0 = list(idx0)
    idx1 = list(idx1)
    # create seeds array
    seeds = np.zeros((len(idx0), 2, 4))
    seeds[:, 0, :-1] = st0_hits[idx0]
    seeds[:, 1, :-1] = st1_hits[idx1]
    seeds[:, :2, -1] = stations_z[1:3]
    return seeds


def get_extreme_points(xcenter,
                       ycenter,
                       width,
                       height):
    # width dependent measurements
    half_width = width / 2
    left = xcenter - half_width
    right = xcenter + half_width
    # height dependent measurements
    half_height = height / 2
    bottom = ycenter - half_height
    top = ycenter + half_height
    return (left, right, bottom, top)


def is_ellipse_intersects_module(ellipse_params,
                                 module_params):
    # calculate top, bottom, left, right
    ellipse_bounds = get_extreme_points(*ellipse_params)
    module_bounds = get_extreme_points(*module_params)
    # check conditions
    if ellipse_bounds[0] > module_bounds[1]:
        # ellipse left > rectangle rigth
        return False

    if ellipse_bounds[1] < module_bounds[0]:
        # ellipse rigth < rectangle left
        return False

    if ellipse_bounds[2] > module_bounds[3]:
        # ellipse bottom > rectangle top
        return False

    if ellipse_bounds[3] < module_bounds[2]:
        # ellipse top < rectangle bottom
        return False

    return True


def is_ellipse_intersects_station(ellipse_params,
                                  station_params):
    module_intersections = []
    for module in station_params:
        ellipse_in_module = is_ellipse_intersects_module(
            ellipse_params, module)
        # add to the list
        module_intersections.append(ellipse_in_module)
    return any(module_intersections)


def reconstruct_event(single_event_hits, nn, n_stations, stations_z, stations_sizes, vertex_stats=None):
    st = 1
    paths_start = 1
    j = 2
    short_tracks = []
    short_tracks_idx = []
    short_tracks_ellipses = []
    # prepare seeds
    if vertex_stats is None:
        st = 2
        paths_start = 0
        seeds, paths = get_seeds_with_index(single_event_hits, stations_z)
        batch_shape = (len(seeds), n_stations, 4)
        path_shape = (len(seeds), n_stations)

    else:
        seeds, paths = get_seeds_with_index_with_vertex(single_event_hits, vertex_stats, stations_z)
        batch_shape = (len(seeds), n_stations + 1, 4)
        path_shape = (len(seeds), n_stations + 1)

    batch_xs = np.zeros(batch_shape, dtype=np.float32)
    batch_paths = np.full(path_shape, -1)
    batch_paths[:, paths_start:2] = paths
    batch_ellipses = np.zeros_like(batch_xs)
    batch_xs[:, :2] = seeds
    MIN_LENGTH = 4 if vertex_stats is None else 3
    track_lost = []
    track_lost_last_ellipse = []
    while st < n_stations:
        y_pred = nn.predict(batch_xs)
        batch_xs_new = []
        batch_paths_new = []
        for i, ellipse in enumerate(y_pred):
            batch_ellipses[i, st, :] = ellipse
            if not is_ellipse_intersects_station(ellipse, stations_sizes[st]):
                if st >= MIN_LENGTH:
                    short_tracks.append(batch_xs[i])
                    short_tracks_idx.append(batch_paths[i])
                    short_tracks_ellipses.append(batch_ellipses[i])
                continue
            # ellipse intersection doesn't work!!!
            next_hits = single_event_hits[single_event_hits.station == st][['x', 'y', 'z']]
            next_hits_idx = single_event_hits[single_event_hits.station == st].index
            next_hits = next_hits.values.astype('float32')
            ellipses = np.repeat([ellipse], len(next_hits), axis=0)
            mask = point_in_ellipse_numpy(next_hits, ellipses)
            masked_hits = next_hits[mask]
            if len(masked_hits) == 0:
                track_ids = single_event_hits.loc[batch_paths[i][paths_start:j]].track.values
                if track_ids[0] != -1 and (track_ids == track_ids[0]).all():
                    track_lost.append(batch_paths[i, paths_start:])
                    track_lost_last_ellipse.append((j - paths_start, ellipse))
            masked_idx = next_hits_idx[mask]

            extensions = np.zeros((len(masked_hits), batch_xs.shape[1], 4))
            extensions[:] = batch_xs[i:i + 1]
            extensions[:, j, :-1] = masked_hits

            extensions_idx = np.full((len(masked_idx), batch_paths.shape[1]), -1)
            extensions_idx[:] = batch_paths[i:i+1]
            extensions_idx[:, j] = masked_idx

            if st < n_stations - 1:
                # for the last row we have not next z
                extensions[:, j, -1] = stations_z[st + 1]

            # add to the result array
            batch_xs_new.extend(extensions)
            batch_paths_new.extend(extensions_idx)

        if len(batch_xs_new) == 0:
            print('Alert!!!')
            break

        # prepare for the next iteration
        batch_xs = np.asarray(batch_xs_new)
        batch_paths = np.asarray(batch_paths_new)
        st += 1
        j += 1

    if len(short_tracks_idx) > 0:
        batch_paths = np.vstack([batch_paths, short_tracks_idx])
        short_tracks_idx = np.copy(np.array(short_tracks_idx)[:, paths_start:])

    if len(short_tracks) > 0:
        batch_xs = np.vstack([batch_xs, short_tracks])


    # only remove z+1
    ret0 = batch_xs[:, :, :-1]
    if vertex_stats is not None:
        # remove vertex and z+1
        ret0 = batch_xs[:, 1:, :-1]
        batch_paths = batch_paths[:, paths_start:]

    return ret0, batch_paths, short_tracks, short_tracks_idx, short_tracks_ellipses, track_lost, track_lost_last_ellipse

def visualize_3d(config, event_df, withNN = False, vertex_stats = None):
    cfg_vis = config['visualize']
    assert cfg_vis['mode'] == '3d'

    visualizer_all_tracks = Visualizer(event_df, 'ALL TRACKS')
    visualizer_lost_tracks = Visualizer(event_df, 'LOST TRACKS')
    visualizer_found_tracks = Visualizer(event_df, 'FOUND TRACKS')
    visualizer_all_tracks.add_coord_planes(config['stations_sizes'])
    visualizer_lost_tracks.add_coord_planes(config['stations_sizes'])
    visualizer_found_tracks.add_coord_planes(config['stations_sizes'])
    if withNN:
        event_df_tracks = event_df[event_df.track != -1]
        batch_tracks_hits, batch_track_idx, short_tracks, short_tracks_idxs, \
        short_track_ellipses, lost_tracks, track_lost_last_ellipse = reconstruct_event(event_df, get_nn(config['network']),
                                                                                       6, config['z_stations'], config['stations_sizes'], vertex_stats=vertex_stats)
        #visualizer.init_draw(reco_tracks=batch_track_idx)
        visualizer_lost_tracks.init_draw(reco_tracks=lost_tracks, draw_all_hits=True)
        for ind, (last_index, ell) in enumerate(track_lost_last_ellipse):
            visualizer_lost_tracks.add_nn_pred(last_index, lost_tracks[ind][last_index-1], ell[:2], ell[2:])

        visualizer_found_tracks.init_draw(reco_tracks=batch_track_idx)
        visualizer_all_tracks.init_draw(draw_all_tracks_from_df=True)
        visualizer_found_tracks.draw(False)
        visualizer_lost_tracks.draw(False)
        visualizer_all_tracks.draw(True)
    else:
        visualizer = Visualizer(event_df, 'ALL TRACKS')
        visualizer.init_draw(draw_all_tracks_from_df=True)
        visualizer.draw()

def main(config_path='configs/visualize_basic_with_vertex.yaml'):
    config = load_config(config_path)
    df = getEvents(config['df'], parseDf(config['df']))
    if config['df']['drop_broken_tracks']:
        df = dropBroken(df, preserve_fakes=True, drop_full_tracks=True)
    #reconstruct_event(df, get_nn(config['network']),6, config['z_stations'], config['stations_sizes'])
    if config['visualize']:
        vertex_stats = None
        if config['with_vertex']:
            vertex_stats = read_vertex_file(config['with_vertex']['vertex_fname'])

        visualize_3d(config, df, withNN=True, vertex_stats=vertex_stats)



if __name__ == "__main__":
    main()


