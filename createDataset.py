# import sys

import utils.read as read
import utils.get as get

import os
import numpy as np
import h5py

# from sklearn.utils import shuffle


def main():
    # sample size in current dataset
    height = 64
    width = 64
    channels_input = 4
    # channels_output = 4

    # reference viscosity
    viscosity = 1e-4

    # 1 for turbulent regime, 0 for laminar regime
    turbulence = 1

    shape = (None, height, width, channels_input)

    # path where to save the dataset
    save_path = './datasets/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # name of the dataset
    names = ['test', 'train', 'validation']
    ncases = [1, 3, 1]
    for name, number_of_cases in zip(names,ncases):
        # name = "validation_dataset"

        file_path = save_path + '/' + name + 'dataset.h5'

        # benchmark = "ellipse"
        grid = "1b_rect_grid"

        # number_of_cases = 1

        for i in range(1, number_of_cases + 1):
            case = "case_" + str(i)
            data = "data"
            dim = np.array([height, width, 0.0, float(viscosity)])

            train_x_addrs, train_y_addr = read.addrs(data, "poiseuilleFlow/" + name, case, grid)

            coordinates = 0

            train_x = []
            train_y = []
            get.case_data(train_x_addrs, train_y_addr, coordinates, dim, grid, turbulence, train_x, train_y)

            train_x = np.float32(np.asarray(train_x))
            train_y = np.float32(np.asarray(train_y))

            print("x size ", train_x.shape)
            print("y size ", train_y.shape)

            # If dataset does not exist, create it
            if i == 1:
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('x_train_dataset', data=train_x, compression="lzf", chunks=True, maxshape=shape)
                    hf.create_dataset('y_train_dataset', data=train_y, compression="lzf", chunks=True, maxshape=shape)

            # if it exists, augment it
            else:
                with h5py.File(file_path, 'a') as hf:

                    hf["x_train_dataset"].resize((hf["x_train_dataset"].shape[0] + train_x.shape[0]), axis=0)
                    hf["x_train_dataset"][-train_x.shape[0]:] = train_x

                    hf["y_train_dataset"].resize((hf["y_train_dataset"].shape[0] + train_y.shape[0]), axis=0)
                    hf["y_train_dataset"][-train_y.shape[0]:] = train_y
    print('Finished')


if __name__ == '__main__':
    main()
