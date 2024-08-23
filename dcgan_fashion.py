import torch.utils
from imagesearch.dcgan import Generator
from imagesearch.dcgan import Discriminator 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from torchvision import transforms
from sklearn.utils import shuffle 
from imutils import build_montages 
from torch.optim import Adam 
from torch.nn import BCELoss 
from torch import nn 
import torch
import numpy as np 
import cv2 
import os 

def weights_init(model):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

init_lr = 0.0002
betas = (0.5, 0.999)
num_epochs = 50
batch_size =128 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

trainData = MNIST(root='data', train=True, download=False, transform=dataTransforms)
testData = MNIST(root='data', train=False, download=False, transform=dataTransforms)
data = torch.utils.data.ConcatDataset([trainData, testData])

dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)
stepsPerEpochs = len(dataloader.dataset) // batch_size 

gen = Generator(inputDim=100, outputChannels=1)
gen.apply(weights_init)
gen.to(device)

disc = Discriminator(depth=1)
disc.apply(weights_init)
disc.to(device)

genopt = Adam(gen.parameters(), lr=init_lr, betas=betas, weight_decay=init_lr/num_epochs)
discopt = Adam(disc.parameters(), lr=init_lr, betas=betas, weight_decay=init_lr/num_epochs)

criterion = BCELoss()
benchmarkNoise = torch.randn(256, 100, 1, 1, device=device)

realLabel = 1 
fakeLabel = 0 

for epoch in range(num_epochs):
    print('[INFO] starting epoch {} of {} ...'.format(epoch+1, num_epochs))

    epochLossG = 0 
    epochLossD = 0

    for x in dataloader:

        disc.zero_grad()

        images = x[0]
        images = images.to(device)

        bs = images.size(0)
        labels = torch.full((bs,), realLabel, dtype=torch.float, device=device)

        output = disc(images).view(-1)

        errorReal = criterion(output, labels)
        errorReal.backward(retain_graph=True)

        noise = torch.randn(bs, 100, 1, 1, device=device)
        fake = gen(noise)
        labels.fill_(fakeLabel)

        output = disc(fake).view(-1)
        errorfake = criterion(output, labels)

        errorfake.backward(retain_graph=True)

        errorD = errorfake + errorReal 
        discopt.step()

        gen.zero_grad()

        labels.fill_(realLabel)
        output = disc(fake).view(-1)

        errorG = criterion(output, labels)
        errorG.backward(retain_graph=True)

        genopt.step()

        epochLossD +=errorD
        epochLossG +=errorG

	# display training information to disk
    print("[INFO] Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
		epochLossG / stepsPerEpochs, epochLossD / stepsPerEpochs))
	# check to see if we should visualize the output of the
	# generator model on our benchmark data
    if (epoch + 1) % 2 == 0:
		# set the generator in evaluation phase, make predictions on
		# the benchmark noise, scale it back to the range [0, 255],
		# and generate the montage
        gen.eval()
        images = gen(benchmarkNoise)
        images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]
		# build the output path and write the visualization to disk
        p = os.path.join('./outputs', "epoch_{}.png".format(
			str(epoch + 1).zfill(4)))
        cv2.imwrite(p, vis)
		# set the generator to training mode
        gen.train()