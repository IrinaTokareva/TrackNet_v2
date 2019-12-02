import numpy as np
import plac
import os
import yaml
import sys

sys.path.append(os.path.realpath('.'))

from data_utils import read_train_dataset
from data_utils import train_test_split
from data_utils import get_dataset


def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)

@plac.annotations(
    config_path=("Path to the config file", "option", None, str))
def main(config_path='configs/prepare_train_data_equal_distribution.yaml',
         ):
    config = load_config(config_path)

    # reading the config
    train_dir = config['train_dir']
    fname_to_save = config['fname_to_save']
    val_size = config['val_size']
    random_seed = config['random_seed']
    debug = config['debug']
    vertex = config['vertex'] if config['vertex'] != 'None' else None
    distributions = config['distribution']
    len3 = distributions['len3']
    len4 = distributions['len4']
    len5 = distributions['len5']
    len6 = distributions['len6']

    print("Read train data")
    train_data = read_train_dataset(
        train_dir,
        vertex_fname=vertex,
        random_seed=random_seed,
        debug=debug,
        train_split=(len3, len4, len5, len6))

    print("\nSplit on train and validation")
    train, validation = train_test_split(
        train_data,
        shuffle=False,
        test_size=val_size,
        random_seed=random_seed)
    print("\nTrain shape: {}".format(train.shape))
    print("\nValidation shape: {}".format(validation.shape))

    print("\nPrepare data for full validation")
    tracklens = np.count_nonzero(validation, axis=1)[:, -1]
    full_val_data = validation[tracklens == validation.shape[1]]

    print("\nPrepare data as input to NN")
    train = get_dataset(train, shuffle=False, random_seed=random_seed)
    validation = get_dataset(validation, shuffle=False, random_seed=random_seed)

    print("\nSave to the file `%s`" % fname_to_save)
    np.savez(fname_to_save,
             x_train=train[0],
             y_train=train[1],
             x_val=validation[0],
             y_val=validation[1],
             full_val=full_val_data)


if __name__ == "__main__":
    plac.call(main)