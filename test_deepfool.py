import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from Deepfool import deepfool
import os


net = models.resnet34(pretrained=True)

# Switch to evaluation mode
net.eval()

im_orig = Image.open('test_im2.jpg')

mean = [0.485,0.456,0.406]
std = [ 0.229, 0.224, 0.225 ]

# Remove the mean

im = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,std = std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im,net)

labels = open(os.path.join('synset_words.txt'),'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ",str_label_orig)
print("Perturbed label = ",str_label_pert)

def clip_tensor(A,minv,maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x,0,255)

tf = transforms.Compose([transforms.Normalize(mean=[0,0,0],std=map(lambda x: 1 / x, std)),
                        transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

plt.figure()
x = np.transpose(pert_image[0].numpy(),[1,2,0])
r_map = np.transpose(r[0],[1,2,0])
def Normalize(data):
    mx = data.max()
    mn = data.min()
    return (data - mn) / (mx - mn)
x = Normalize(x)
r_map = Normalize(r_map)
plt.imshow(x)
plt.title(str_label_pert)
plt.imshow(r_map)
plt.title("r")
plt.show()