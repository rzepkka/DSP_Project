import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm
import PIL
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import make_grid
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz
from captum.attr import Occlusion


# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# Target Classes
target_name = {0:"NonDemented", 1:"VeryMildDemented", 2:"MildDemented", 3:"ModerateDemented"}
target_map = {"NonDemented": 0, "VeryMildDemented": 1, "MildDemented": 2, "ModerateDemented": 3}

# Load Models
vgg16 = models.vgg16(pretrained=True)


#def import_model(path="classifier/model/export.pkl"):
#    return load_learner(Path(path))


def load_checkpoint(filepath="classifier/model/checkpoint_vgg.pth"):
    """
    Loads a model from a file.
    :param filepath: a string specifying the path to the file containing the model
    :return: model:
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)

    if checkpoint['arch'] == 'vgg16':
        model = vgg16
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")

    model.class_to_idx = checkpoint['class_to_idx']

    # Build a Custom Classifier for our problem
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.2)),
                                            ('fc2', nn.Linear(4096, 4)),
                                            ('output', nn.Softmax(dim=1))]))

    model.classifier = classifier

    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def imshow(img):
    """
    Displays Image.
    :param img: RGB Image Tensor -> 3xNxN.
    :return: None
    """
    img = img / 2 + 0.5     # unnormalize
    img = img.permute(1, 2, 0)
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()


def PILtoNormalizedTensor(input):
    input_tensor = transforms.PILToTensor()(input)  # PIL Image to Tensor
    input_tensor = input_tensor[None, :, :, :]
    transform = transforms.Compose(
        [transforms.ConvertImageDtype(torch.float),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Apply Image Normalization
    return transform(input_tensor)


def make_prediction(model, input):
    """

    :param model: torch model
    :param input: PIL Image -> shape: [1, 3, N, N]
    :return: string with predicted class
    """
    input_tensor = PILtoNormalizedTensor(input)
    outputs = model(input_tensor)
    #imshow(make_grid(input_tensor))
    print('Predicted: ', ' '.join('%5s' % target_name[torch.argmax(outputs[j]).detach().cpu().item()]
                                  for j in range(len(outputs))))
    return target_name[torch.argmax(outputs[0]).detach().cpu().item()]


def explain_image(input, prediction, model, method=None):
    """
    Explains the model image with a heatmap according to the chosen method.
    :param input: PIL Image
    :param prediction: predicted class
    :param model: model object
    :param method: one of the following explainable methods ->
     ["gradient_magnitudes", "integrated_gradients", "deeplift", "gradcam"]
    :return: original image and explainable image.
    """

    input_tensor = PILtoNormalizedTensor(input)  # PIL Image to Tensor
    input_sample = (input_tensor.squeeze() / 2 + 0.5).unsqueeze(0)  # Denormalized Tensor
    input_sample.requires_grad = True
    original_image = np.transpose((input_sample[0].cpu().detach().numpy() / 2) + 0.5, (1, 2, 0)) # Array with denorm Img
    if method == "Gradient Magnitudes":
        saliency = Saliency(model)
        grads = saliency.attribute(input_sample, target=target_map[prediction])
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        # Visualize
        x = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="absolute_value",
                                     show_colorbar=True, title="Overlayed Gradient Magnitudes")
    elif method == "Integrated Gradients":
        ig = IntegratedGradients(model)
        attr_ig, delta = attribute_image_features(model, ig, input_sample, target=target_map[prediction],
                                                  baselines=input_sample * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print('Approximation delta: ', abs(delta))

        # Visualize
        x = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map", sign="absolute_value",
                                     cmap="Reds",
                                     show_colorbar=True, title="Overlayed Integrated Gradients")
    elif method == "Deep Lift":
        dl = DeepLift(model)
        attr_dl = attribute_image_features(model, dl, input_sample, target=target_map[prediction], baselines=input_sample * 0)
        attr_dl = np.transpose(attr_dl.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # Visualize
        x = viz.visualize_image_attr(attr_dl, original_image, method="blended_heat_map", sign="all", show_colorbar=True,
                                     title="Overlayed DeepLift")
    elif method == "Grad-Cam":
        # GuidedGradCAM.
        guided_gc = GuidedGradCam(model, model.features[28])
        attribution = guided_gc.attribute(input_sample, target=target_map[prediction],)
        attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # Visualize
        x = viz.visualize_image_attr(attribution, original_image, method="blended_heat_map", sign="all",
                                     show_colorbar=True,
                                     title="GradCam")

    elif method == "Occlusion":
        occlusion = Occlusion(model)

        attribution = occlusion.attribute(input_sample, strides=(3, 8, 8), target=target_map[prediction],
                                          sliding_window_shapes=(3, 15, 15), baselines=0)
        attribution = np.transpose(attribution.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # Visualize
        x = viz.visualize_image_attr(attribution, original_image, method="blended_heat_map", sign="all",show_colorbar=True, title="Occlusion")

    return x[0]


def attribute_image_features(model, algorithm, input, **kwargs):
    """
    :param algorithm:
    :param input:
    :param kwargs:
    :return: tensor attributions
    """
    model.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              **kwargs)
    return tensor_attributions


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == "__main__":
    # Read Image
    img_path = '../images/mildDem0.jpg'
    img = PIL.Image.open(img_path).convert('RGB') # PIL Image
    # Import Model and Make a Prediction
    model = load_checkpoint('model/checkpoint_vgg.pth')
    prediction = make_prediction(model, img)
    # Explainable Method
    xplane_method = ["gradient_magnitudes", "integrated_gradients", "deeplift", "gradcam"]
    # Call Explainable Method
    img = explain_image(img, prediction, model, method=xplane_method[0])
    # Display Explanation
    print("Explained.")
    #print('Predicted: ', prediction)
