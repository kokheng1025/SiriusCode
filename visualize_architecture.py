from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization graph to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet.png", show_shapes=True)

modelVGG = MiniVGGNet.build(28, 28, 1, 10)
plot_model(modelVGG, to_file="modelVGG.png", show_shapes=True)