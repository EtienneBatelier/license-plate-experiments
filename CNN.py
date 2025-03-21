import numpy as np
import matplotlib.pyplot as plt
import pandas

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset, random_split

from torchinfo import summary

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)


### Specify a network architecture

class ConvActPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvActPool, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = out_channels,
                              kernel_size = (13, 13),
                              padding = 'same')
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = (2,2))

    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))

class LPCharactersModel(nn.Module):
    def __init__(self, num_classes_):
        super(LPCharactersModel, self).__init__()
        sizes = [(1,32),
                 (32,64),
                 (64,128),
                 (128,256)]
        self.conv_act_pool = nn.Sequential(*[ConvActPool(in_, out_) for in_, out_ in sizes])

        self.output = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(2*1*256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes_))
    def forward(self, x):
        val = self.conv_act_pool(x)
        val = torch.flatten(val, start_dim = 1)
        return self.output(val)


### Build, train and save a neural network

def create_CNN():
    ### Arrange the data into a DataModule

    num_classes = 35
    X_tensor = torch.load("./pytorch_files/X_tensor.pt", weights_only=True)
    Y_tensor = torch.load("./pytorch_files/Y_tensor.pt", weights_only=True)
    print('Shape X: ', X_tensor.shape)
    print('Shape Y: ', Y_tensor.shape)
    lp_characters_ds = TensorDataset(X_tensor, Y_tensor)
    lp_characters_train_ds, lp_characters_test_ds = random_split(lp_characters_ds, [48000, 8535])
    lp_characters_dm = SimpleDataModule(lp_characters_train_ds,
                                        lp_characters_test_ds,
                                        validation = 0.2,
                                        batch_size = 128,
                                        num_workers = rec_num_workers())
    print(Y_tensor.min(), Y_tensor.max())


    ### Check the shape of typical batches in the data loader.

    for idx, (X_ ,Y_) in enumerate(lp_characters_dm.train_dataloader()):
        print('Batch shape X: ', X_.shape)
        print('Batch shape Y: ', Y_.shape)
        if idx >= 1:
            break


    ### Instantiate a model

    lp_characters_model = LPCharactersModel(num_classes)
    lp_characters_optimizer = RMSprop(lp_characters_model.parameters(), lr=0.001)
    lp_characters_module = SimpleModule.classification(lp_characters_model,
                                                       num_classes = num_classes,
                                                       optimizer = lp_characters_optimizer)
    summary(lp_characters_model,
            input_data = X_,
            col_names = ['input_size', 'output_size', 'num_params'])


    ### Set a logger

    lp_characters_logger = CSVLogger('./pytorch_files/logs', name = 'LPCharacters')


    ### Train the model and save it

    max_epochs = 6
    lp_characters_trainer = Trainer(deterministic = False,
                                    max_epochs = max_epochs,
                                    logger = lp_characters_logger,
                                    enable_progress_bar = True,
                                    callbacks = [ErrorTracker()])

    lp_characters_trainer.fit(lp_characters_module, datamodule = lp_characters_dm)
    torch.save(lp_characters_model.state_dict(), "./pytorch_files/saved_models/lp_characters_trained.pt")
    lp_characters_model.eval()


    ### Load the model (if needed)

    #lp_characters_model = LPCharactersModel(num_classes)
    #lp_characters_model.load_state_dict(torch.load("./pytorch_files/saved_models/lp_characters_trained.pt", weights_only=True))
    #lp_characters_model.eval()


    ### Evaluate the model

    lp_characters_trainer.test(lp_characters_module, datamodule = lp_characters_dm)

    def summary_plot(results, ax_,
                     col = 'loss',
                     valid_legend = 'Validation',
                     training_legend = 'Training',
                     y_label = 'Loss'):
        for (column, color, label) in zip([f'train_{col}_epoch', f'valid_{col}'],
                                           ['black', 'red'],
                                           [training_legend, valid_legend]):
            results.plot(x='epoch', y=column, label=label, marker='o', color=color, ax=ax_)
        ax_.set_xlabel('Epoch')
        ax_.set_ylabel(y_label)
        return ax_

    log_path = lp_characters_logger.experiment.metrics_file_path
    lp_characters_results = pandas.read_csv(log_path)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    summary_plot(lp_characters_results, ax, col = 'accuracy', y_label = 'Accuracy')
    ax.set_xticks(np.linspace(0, max_epochs).astype(int))
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    plt.show()
    plt.close()

#create_CNN()