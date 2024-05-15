import torch
from torch.utils.data import DataLoader
import torchvision

from pytools.data.datasetS2 import DatasetS2Random
from pytools.logging.images import make_grid
from pytools.nn.projectors import (
    PCA, PCAProjector, RandomProjector)

# General
ROOT = "/scratchm/sollivie/data/sentinel2/random100/train"
NBANDS = [1,2,3,4,5,6,7,9,10]
HEIGHT = 256
WIDTH = 256
MODE = "multispectral"
TRESHOLD = 80
NIMG_PLOT = 8
IMG_PATH = "/scratchm/sollivie/PyTools/test/img/projections.png"

# PCA
NIMG = 128
NPIXELS = 1000000
PLOT_PATH = "/scratchm/sollivie/PyTools/test/img/pca_9b.png"
HIST_PATH = "/scratchm/sollivie/PyTools/test/img/pca_9b_histograms.png"
NBINS = 100
PCA_PATH = "/scratchm/sollivie/PyTools/test/save/pca_9b.pt"

# Random
IN_CHANNELS = 11
BATCH_SIZE = 16

# ---------------------------------- PCA ----------------------------------- #
dataset = DatasetS2Random(
    ROOT, NBANDS, HEIGHT, WIDTH, MODE, TRESHOLD)
loader = DataLoader(dataset, batch_size=NIMG)

data = next(iter(loader))
pca = PCA(data, NPIXELS, 3)
pca.train()
pca.normalize()
pca.plot(PLOT_PATH)
pca.plot_histograms(HIST_PATH, nbins=NBINS)
pca.save(PCA_PATH)

x = data[:NIMG_PLOT]
y = ((x.transpose(1,3) - pca.mu) 
     @ pca.principal_components).div(pca.eigenvalues).transpose(1,3)
torchvision.utils.save_image(
    make_grid(x[:,1:4], y, xrange=(-1, 1.))["rgb"], IMG_PATH)


pca_proj = PCAProjector(**torch.load(PCA_PATH))
y2 = pca_proj(x)

print("y == y2", torch.equal(y, y2))
print("y ~~ y2", torch.allclose(y, y2))
torchvision.utils.save_image(
    make_grid(x[:,1:4], y2, xrange=(-1, 1.))["rgb"], IMG_PATH[:-4] + "2.png")

# --------------------------------- Random --------------------------------- #
# x = torch.randn(IN_CHANNELS,128,128)
# y = torch.randn(8,IN_CHANNELS,128,128)

# print("Random")
# print("\nOriginal random projector")
# random_projector = RandomProjector(IN_CHANNELS,3)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)

# print("\nBatched random projector n = 1")
# random_projector = BatchRandomProjector(IN_CHANNELS, 3, batch_size=1)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)

# print(f"\nBatched random projector n = {BATCH_SIZE}")
# random_projector = BatchRandomProjector(IN_CHANNELS, 3, batch_size=BATCH_SIZE)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)

# print("\n\nDeterminist")
# print("\nOriginal random projector")
# random_projector = RandomProjector(IN_CHANNELS, 3, determinist=True)
# print("channels", random_projector.channels)
# for _ in range(IN_CHANNELS * (IN_CHANNELS - 1) * (IN_CHANNELS - 2)):
#     random_projector.generate()
# print("channels", random_projector.channels)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)

# print("\nBatched random projector n = 1")
# random_projector = BatchRandomProjector(
#     IN_CHANNELS, 3, batch_size=1, determinist=True)
# print("channels", random_projector.channels)
# for _ in range(IN_CHANNELS * (IN_CHANNELS - 1) * (IN_CHANNELS - 2)):
#     random_projector.generate()
# print("channels", random_projector.channels)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)

# print(f"\nBatched random projector n = {BATCH_SIZE}")
# random_projector = BatchRandomProjector(
#     IN_CHANNELS, 3, batch_size=16, determinist=True)
# print("channels", random_projector.channels)
# for _ in range(IN_CHANNELS * (IN_CHANNELS - 1) * (IN_CHANNELS - 2) // BATCH_SIZE + 1):
#     random_projector.generate()
# print("channels", random_projector.channels)
# print("x shape", random_projector(x).shape)
# print("y shape", random_projector(y).shape)