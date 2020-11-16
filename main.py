import cv2
import torch
import numpy as np
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms

import model

# imagenet transform
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

# load the image
dataset = datasets.ImageFolder(root='./images/', transform=transform)

# dataloader
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=False,
                                         batch_size=1)

# model
my_vgg = model.VGG()

# evaluation mode
my_vgg.eval()

# get the image
img, _ = next(iter(dataloader))

# likelihood distribution
distribution = my_vgg(img)
# should be 386 for the elephant
pred = distribution.argmax(dim=1)
print(pred)

# gradient of the output with respect to the model parameters
distribution[:, 386].backward()

# gradients that we hooked (gradients of the last conv)
gradients = my_vgg.get_activation_gradient()
# print(gradients.shape)

# pool the gradients across the channel
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# print(len(pooled_gradients))

# activations of the last conv layer
activations = my_vgg.get_activation(img).detach()

# weight the channels by corresponding gradients
for i in range(len(pooled_gradients)):
    activations[:, i, :, :] *= pooled_gradients[i]

# average all channels of the weighted activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu to obtain only positive effect
heatmap = np.maximum(heatmap, 0)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# show the heatmap
plt.matshow(heatmap)
# plt.show()

# read the input image
img = cv2.imread('./images/elephant/elephant.jpeg')
heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./heatmap_elephant.jpg', superimposed_img)


