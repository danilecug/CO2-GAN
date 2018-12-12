import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import imageio
import seaborn as sns
import matplotlib.gridspec as gridspec
from skimage.measure import compare_ssim  as ssim
sns.set_style('darkgrid')

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result(input, target, gen_image, epoch, training=True, save=False, save_dir='results/', show=False, fig_size=(17, 5), dpi=300):
    
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)

    fig, axes = plt.subplots(1, 4, figsize=fig_size, dpi=300)
    #gs = gridspec.GridSpec(1, 4, wspace=0.03, hspace=0.03, width_ratios=[1, 1, 1.2, 1.2])
    imgs = [input, gen_image, target, gen_image-target]
    

    for i, (ax, img) in enumerate(zip(axes.flatten(), imgs)):
        ax.axis('off')
        ax.set_adjustable('box-forced')
        # Scale to 0-255
        #img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        #img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().astype(np.uint8)
        #ax.imshow(img, cmap=None, aspect='equal')
        if i==0: 
            sns.heatmap(img, cmap='RdYlBu',cbar=True, vmin=0, vmax=1, ax = ax)
        elif i==1: 
            sns.heatmap(img, cmap='YlGnBu', cbar=False, vmin=0, vmax=1,ax = ax)
        elif i==2: 
            sns.heatmap(img, cmap='YlGnBu', cbar=True, vmin=0, vmax=1,ax = ax)
        elif i==3: 
            sns.heatmap(img, cmap='seismic', cbar=True, vmin=-0.1, vmax=0.1,ax = ax)
        left = 0.03 
        right = 0.95
        bottom = 0.1
        top = 0.95
        plt.subplots_adjust(left=left, 
                            right = right,
                            bottom = bottom,
                            top = top,
                            wspace=0.1,
                            hspace=0.1)
    

    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + '.png'
            fig.subplots_adjust(bottom=0.1)
            fig.subplots_adjust(top=0.95)
            fig.subplots_adjust(right=0.95)
            fig.subplots_adjust(left=0.03)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_pix2pix_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)
