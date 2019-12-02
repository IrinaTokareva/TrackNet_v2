import torch
import torch.utils.data as utils
import numpy as np
import plac
import yaml
import os

from torch.optim import Adam
from time import gmtime, strftime

# tracknet utils
from tracknet import TrackNet
from losses import tracknet_loss
from metrics import circle_area
from metrics import point_in_ellipse
from metrics import MetricsCallback


def load_config(config_file):
    with open(config_file) as f:
        return yaml.load(f)


def create_data_loader(data_x, data_y, batch_size):
    tensor_x = torch.stack([torch.Tensor(i).t() for i in data_x])
    tensor_y = torch.stack([torch.Tensor(i).t() for i in data_y])
    train_dataset = utils.TensorDataset(tensor_x, tensor_y)
    return utils.DataLoader(train_dataset, batch_size=batch_size)


def train(tracknet, train_loader, device, optimizer, epoch):
    tracknet.train()
    hits_efficiency = 0
    circle = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = tracknet.forward(x)
        loss = tracknet_loss(y, y_pred)
        in_ellipse = point_in_ellipse(y, y_pred)
        hits_efficiency += torch.sum(in_ellipse) / len(train_loader.dataset)
        circle = (circle + torch.mean(circle_area(y, y_pred))) / 2
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, point_in_ellipse: {:.4f}, circle_area: {:.4f}'.format(
            epoch + 1, (batch_idx + 1) * len(x), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), hits_efficiency, torch.mean(circle)))
    return hits_efficiency


def validate(tracknet, validation_loader, device):
    tracknet.eval()
    hits_efficiency = 0
    circle = 0
    loss = 0
    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            y_pred = tracknet.forward(x)
            loss = (loss + tracknet_loss(y, y_pred).item()) / 2
            in_ellipse = point_in_ellipse(y, y_pred)
            hits_efficiency += torch.sum(in_ellipse) / len(validation_loader.dataset)
            circle = (circle + torch.mean(circle_area(y, y_pred))) / 2
            # print("Circle area: {}".format(circle))
            print("Hits efficiency: {}".format(hits_efficiency))
    print('After validation : loss: {:.6f}, point_in_ellipse: {:.4f}, circle_area: {:.4f}'.format(loss, hits_efficiency, circle))
    return hits_efficiency, circle


def save_model(autosave, datetime, epoch, val_hits_efficiency, val_circle, tracknet):
    print("Save model's weights")
    file_prefix = autosave['file_prefix']
    directory = autosave['output_dir']
    res_name = directory + datetime
    if not os.path.exists(res_name):
        os.makedirs(res_name)
    filepath = "{}/{}_init-{:02d}-.val_point_in_ellipse.{:.2f}.-val_circle_area.{:.2f}.pt".format(
        res_name, file_prefix, epoch + 1, val_hits_efficiency, val_circle)
    torch.save(tracknet.state_dict(), filepath)


@plac.annotations(
    config_path=("Path to the config file", "option", None, str))


def main(config_path='../configs/train_init.yaml'):
    config = load_config(config_path)
    random_seed = config['random_seed']
    data_path = config['data_path']
    batch_size = config['batch_size']
    autosave = config['autosave']
    epochs = config['epochs']
    # set random seed for reproducible results
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Read data")
    data = np.load(data_path)
    print("Train size: %d" % len(data['x_train']))
    print("Validation size: %d" % len(data['x_val']))

    train_loader = create_data_loader(data['x_train'], data['y_train'], batch_size)
    validation_loader = create_data_loader(data['x_val'], data['y_val'], batch_size)

    tracknet = TrackNet()
    print(tracknet)
    # TODO: clipnorm in pytorch
    optimizer = Adam(tracknet.parameters())

    metrics_cb = MetricsCallback(test_data=data['full_val'])

    device = torch.device("cpu")
    print("Training...")
    # train the network
    # TODO: change to metric from config
    best_point_in_ellipse = 0
    metrics_cb.on_train_begin()
    datetime = strftime("_%Y-%m-%d__%H.%M.%S", gmtime())
    for epoch in range(epochs):
        hits_efficiency = train(tracknet, train_loader, device, optimizer, epoch)
        if (1 - hits_efficiency) ** 2 < (1 - best_point_in_ellipse) ** 2:
            print('Value of metric point_in_ellipse has improved from {} to {}'.format(best_point_in_ellipse, hits_efficiency))
            best_point_in_ellipse = hits_efficiency
        else:
            print('Value of metric point_in_ellipse did not improve. The best value is {}'.format(best_point_in_ellipse))
        val_hits_efficiency, val_circle = validate(tracknet, validation_loader, device)
        metrics_cb.on_epoch_end(epoch, tracknet, batch_size, device)
        if autosave and autosave['enabled']:
            save_model(autosave, datetime, epoch, val_hits_efficiency, val_circle, tracknet)


if __name__ == "__main__":
    plac.call(main)