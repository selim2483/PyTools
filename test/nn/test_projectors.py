from torch.utils.data import DataLoader
import torchvision

from pytools.data.datasetS2 import DatasetS2Random
from pytools.logging.images import make_grid
from pytools.nn.projectors import PCA

# General
ROOT = "/scratchm/sollivie/data/sentinel2/random100/train"
NBANDS = 11
HEIGHT = 256
WIDTH = 256
MODE = "multispectral"
TRESHOLD = 0.8
NIMG_PLOT = 8
IMG_PATH = "/scratchm/sollivie/PyTools/test/img/projections.png"

# PCA
NIMG = 128
NPIXELS = 1000000
PLOT_PATH = "/scratchm/sollivie/PyTools/test/img/pca.png"
HIST_PATH = "/scratchm/sollivie/PyTools/test/img/pca_histograms.png"
NBINS = 100

# ---------------------------------- PCA ----------------------------------- #
dataset = DatasetS2Random(
    ROOT, NBANDS, HEIGHT, WIDTH, MODE, TRESHOLD)
loader = DataLoader(dataset, batch_size=NIMG)

data = next(iter(loader))
pca = PCA(data, NPIXELS, 3)
pca.train()
pca.plot(PLOT_PATH)
pca.plot_histograms(HIST_PATH, nbins=NBINS)

x = data[:NIMG_PLOT]
y = (x.transpose(1,3) - pca.mu) @ pca.principal_components
print("x", x.min(), x.max())
print("y", y.min(), y.max())
torchvision.utils.save_image(make_grid(x[:,1:4], y)["rgb"], IMG_PATH)



