from os.path import join
import argparse
import numpy as np
from tqdm import trange
import hydra
import torch
import torchvision
import seaborn as sns
from torchvision import models, transforms
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv
import argparse

from utils import *
from models import load_models

#code from https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e


def imshow(img, title):
    """Custom function to display the image using matplotlib"""

    # define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    # define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    # convert the tensor img to numpy img and de normalize
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction

    # plot the numpy image
    plt.figure(figsize=(batch_size * 4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()
    return img


def show_batch_images(dataloader):
    images, _ = next(iter(dataloader))

    # run the model on the images
    outputs = model(images)

    # get the maximum class
    _, pred = torch.max(outputs.data, 1)

    # make grid
    img = torchvision.utils.make_grid(images)

    # call the function
    imshow(img, title=[classes[x.item()] for x in pred])

    return images, pred

def occlusion(model, image, label, occ_size=32, occ_stride=1, occ_pixel=0.5):
    # get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    # setting the output image width and height
    output_height = int(np.ceil((height - occ_size) / occ_stride))
    output_width = int(np.ceil((width - occ_size) / occ_stride))

    # create a white image of sizes we defined
    heatmap = torch.zeros((output_height, output_width))

    # iterate all the pixels in each column
    for h in range(0, height):
        for w in range(0, width):

            h_start = h * occ_stride
            w_start = w * occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)

            if (w_end) >= width or (h_end) >= height:
                continue

            input_image = image.clone().detach()

            # replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:, :, w_start:w_end, h_start:h_end] = occ_pixel

            # run inference on modified image
            output = model(input_image)
            output = nn.functional.softmax(output, dim=1)
            prob = output.tolist()[0][label]

            # setting the heatmap location to probability value
            heatmap[h, w] = prob

    return heatmap
@hydra.main(config_name='conf/conf_print_images.yaml')
def main(config):
    train_loader, val_loader, test_loader, train_data, val_data, test_data = get_datasets(config, noise=False,
                                                                                          joint=False,
                                                                                          return_dataset=True)

    model = load_models.load_model('resnet50',config.num_classes,config.pretrained)
    checkpoint = torch.load(config.save_dir.format(**config) + '/model/' + 'resnet50_final.pt')
    model.load_state_dict(checkpoint['state_dict'])
    image = Image.open('/data/common/pacs/cartoon/dog/pic_187.jpg')
    img = np.array(image)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,

    ])
    img = transform(image)


    img = img.unsqueeze(0)
    model.cuda()
    img = img.to('cuda')
    outputs = model(img)


    # get the maximum class
    prob_no_occ, pred = torch.max(outputs.data, 1)

    # get the first item
    prob_no_occ = prob_no_occ[0].item()


    # define the transforms


    heatmap = occlusion(model, img, pred[0].item(), 32, 5)
    #displaying the image using seaborn heatmap and also setting the maximum value of gradient to probability
    imgplot = sns.heatmap(heatmap, xticklabels=False, yticklabels=False, vmax=prob_no_occ)
    figure = imgplot.get_figure()
    figure.savefig('pac_no_pretraining.png', dpi=400)
if __name__ == '__main__':
    main()




