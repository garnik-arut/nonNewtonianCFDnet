import utils.loadDataset as loadDataset
import numpy as np
import matplotlib.pyplot as plt
import io

def image_grid_full(x, y, label_x='x', label_y='y'):
    """Return a 4x4 grid of the ux, uy, p, nu images as a matplotlib figure."""
    # Create a figure to contain the plot.
    fig, axs = plt.subplots(2, 6)
    fig.set_size_inches(18, 4)
    [axi.set_axis_off() for axi in axs.ravel()]
    x_ux = axs[0, 0].pcolormesh(x[:, :, 0])
    axs[0, 0].set_title(label_x + ' ux')
    x_uy = axs[0, 1].pcolormesh(x[:, :, 1])
    axs[0, 1].set_title(label_x + ' uy')
    x_p = axs[1, 0].pcolormesh(x[:, :, 2])
    axs[1, 0].set_title(label_x + ' p')
    x_nu = axs[1, 1].pcolormesh(x[:, :, 3])
    axs[1, 1].set_title(label_x + ' nu')
    y_ux = axs[0, 2].pcolormesh(y[:, :, 0])
    axs[0, 2].set_title(label_y + ' ux')
    y_uy = axs[0, 3].pcolormesh(y[:, :, 1])
    axs[0, 3].set_title(label_y + ' uy')
    y_p = axs[1, 2].pcolormesh(y[:, :, 2])
    axs[1, 2].set_title(label_y + ' p')
    y_nu = axs[1, 3].pcolormesh(y[:, :, 3])
    axs[1, 3].set_title(label_y + ' nu')
    d_ux = axs[0, 4].pcolormesh(y[:, :, 0] - x[:, :, 0])
    axs[0, 4].set_title('diff ux')
    d_uy = axs[0, 5].pcolormesh(y[:, :, 1] - x[:, :, 1])
    axs[0, 5].set_title('diff uy')
    d_p = axs[1, 4].pcolormesh(y[:, :, 2] - x[:, :, 2])
    axs[1, 4].set_title('diff p')
    d_nu = axs[1, 5].pcolormesh(y[:, :, 3] - x[:, :, 3])
    axs[1, 5].set_title('diff nu')

    fig.colorbar(x_ux, ax=axs[0, 0])
    fig.colorbar(x_uy, ax=axs[0, 1])
    fig.colorbar(x_p, ax=axs[1, 0])
    fig.colorbar(x_nu, ax=axs[1, 1])
    fig.colorbar(y_ux, ax=axs[0, 2])
    fig.colorbar(y_uy, ax=axs[0, 3])
    fig.colorbar(y_p, ax=axs[1, 2])
    fig.colorbar(y_nu, ax=axs[1, 3])
    fig.colorbar(d_ux, ax=axs[0, 4])
    fig.colorbar(d_uy, ax=axs[0, 5])
    fig.colorbar(d_p, ax=axs[1, 4])
    fig.colorbar(d_nu, ax=axs[1, 5])
    return fig


def image_grid_comp(x, y, label_x='x', label_y='y'):
    """Return a 4x4 grid of the ux, uy, p, nu images as a matplotlib figure."""
    # Create a figure to contain the plot.
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(16, 4)

    [axi.set_axis_off() for axi in axs.ravel()]
    x_ux = axs[0, 0].pcolormesh(x[:, :, 0])
    axs[0, 0].set_title(label_x + ' ux')
    x_uy = axs[0, 1].pcolormesh(x[:, :, 1])
    axs[0, 1].set_title(label_x + ' uy')
    x_p = axs[1, 0].pcolormesh(x[:, :, 2])
    axs[1, 0].set_title(label_x + ' p')
    x_nu = axs[1, 1].pcolormesh(x[:, :, 3])
    axs[1, 1].set_title(label_x + ' nu')
    y_ux = axs[0, 2].pcolormesh(y[:, :, 0])
    axs[0, 2].set_title(label_y + ' ux')
    y_uy = axs[0, 3].pcolormesh(y[:, :, 1])
    axs[0, 3].set_title(label_y + ' uy')
    y_p = axs[1, 2].pcolormesh(y[:, :, 2])
    axs[1, 2].set_title(label_y + ' p')
    y_nu = axs[1, 3].pcolormesh(y[:, :, 3])
    axs[1, 3].set_title(label_y + ' nu')

    fig.colorbar(x_ux, ax=axs[0, 0])
    fig.colorbar(x_uy, ax=axs[0, 1])
    fig.colorbar(x_p, ax=axs[1, 0])
    fig.colorbar(x_nu, ax=axs[1, 1])
    fig.colorbar(y_ux, ax=axs[0, 2])
    fig.colorbar(y_uy, ax=axs[0, 3])
    fig.colorbar(y_p, ax=axs[1, 2])
    fig.colorbar(y_nu, ax=axs[1, 3])
    return fig


def image_grid_diff(x, y):
    """Return a 4x4 grid of the ux, uy, p, nu images as a matplotlib figure."""
    # Create a figure to contain the plot.
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(4, 4)

    [axi.set_axis_off() for axi in axs.ravel()]
    dx = axs[0, 0].pcolormesh(y[:, :, 0] - x[:, :, 0])
    axs[0, 0].set_title('diff ux')
    dy = axs[0, 1].pcolormesh(y[:, :, 1] - x[:, :, 1])
    axs[0, 1].set_title('diff uy')
    dp = axs[1, 0].pcolormesh(y[:, :, 2] - x[:, :, 2])
    axs[1, 0].set_title('diff p')
    dnu = axs[1, 1].pcolormesh(y[:, :, 3] - x[:, :, 3])
    axs[1, 1].set_title('diff nu')

    fig.colorbar(dx, ax=axs[0, 0])
    fig.colorbar(dy, ax=axs[0, 1])
    fig.colorbar(dp, ax=axs[1, 0])
    fig.colorbar(dnu, ax=axs[1, 1])
    return fig

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    return buf


def main():
    # load training dataset
    train_dataset_path = './datasets/train/train_dataset.h5'
    x_train, y_train = loadDataset.load_train_dataset(train_dataset_path)
    x_train, y_train = np.float32(x_train), np.float32(y_train)

    samples = x_train.shape[0]

    val_dataset_path = './datasets/validation/validation_dataset.h5'
    x_val, y_val = loadDataset.load_train_dataset(val_dataset_path)
    x_val, y_val = np.float32(x_val), np.float32(y_val)

    test_dataset_path = './datasets/test/test_dataset.h5'
    x_test, y_test = loadDataset.load_train_dataset(test_dataset_path)
    x_test, y_test = np.float32(x_test), np.float32(y_test)

    save_dir = './data/images'
    count = 0
    for x,y in zip(x_train,y_train):
        temp = image_grid_full(x, y)
        temp.savefig(save_dir + '/train/' + str(count)+'.png')
        plt.close(temp)
        count += 1
    print('Train set done')

    count = 0
    for x,y in zip(x_val,y_val):
        temp = image_grid_full(x, y)
        temp.savefig(save_dir + '/validation/' + str(count)+'.png')
        plt.close(temp)
        count += 1
    print('Validation set done')


if __name__ == '__main__':
    main()
