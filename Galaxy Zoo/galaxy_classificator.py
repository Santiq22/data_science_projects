################################### Packages ######################################
# --- Torch tools
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets, transforms
import torch.optim as optim

# --- Data analysis
import numpy as np
import pandas as pd
#import sklearn.metrics as met

import os
import sys
from tqdm import tqdm
###################################################################################

################################## Load data ######################################
probabilities = pd.read_csv('training_solutions_rev1.csv', sep=',')

# Create a list to store the image tensors, their IDs and their probabilities
image_list = []
ID_list = []
probabilities_list = []

# CenterCrop object to crop the image
center_crop = transforms.CenterCrop(256)

# Path to dataset
print("PWD: ", os. getcwd())
pat = "/mnt/c/Programacion/Datasets/images_training_rev1"

# Iterate over the images in the folder
for filename in tqdm(os.listdir(pat)):
    # Load each image using the default loader from torchvision
    image = default_loader(os.path.join(pat, filename))
    
    # Crop the image to the given dimension
    cropped_image = center_crop(image)
    
    # Convert the image to a tensor
    tensor_image = torch.tensor(np.array(cropped_image, dtype=np.single), dtype=torch.float)/255.0
    
    # Reshape the tensor
    tensor_image = tensor_image.reshape(3, 256, 256)
    
    # Append the tensor to the image list
    image_list.append(tensor_image)
    
    # Append the ID to the ID list
    ID_temp = int(filename[:-4])
    ID_list.append(ID_temp)
    
    # Append the probability to the probabilities list
    probabilities_list.append(probabilities.loc[probabilities['GalaxyID'] == ID_temp].to_numpy(dtype=np.single).reshape(-1)[1:])

# Concatenate the list of image tensors into a single tensor
image_list = torch.stack(image_list)
# Convert the list of IDs into a Torch tensor of integers
ID_list = torch.tensor(ID_list, dtype=torch.int32)
# Convert the list of probability distributions into a Torch tensor
probabilities_list = torch.tensor(np.array(probabilities_list), dtype=torch.float)
print("¡Listo!")
sys.exit()
###################################################################################

################## Building training and validation datasets ######################
# Let us check the dimenions of the tensors
print(image_list.shape, "\n")
print(ID_list.shape, "\n")
print(probabilities_list.shape)

# We need to define the training and validation dataset. To do so, we will consider the validation 
# set composed of the 15% of the images of the original dataset, while the other 85% will remain
# as the training dataset.

prop = 0.15
testing_size = int(len(ID_list)*prop)
training_size = len(ID_list) - testing_size

# Check the sizes
print("Validation size: ", testing_size, "\n")
print("Training size: ", training_size)

# Let us generate a sequence of random integers that are the indexes of the images and their IDs.
testing_indexes = np.random.randint(0, len(ID_list) - 1, testing_size)
training_indexes = np.delete(np.linspace(0, len(ID_list), len(ID_list), endpoint=False, dtype=int),
                             testing_indexes)

# Training IDs and images
training_images = image_list[training_indexes]
ID_training = ID_list[training_indexes]
training_probabilities = probabilities_list[training_indexes]

# Validation IDs and images
testing_images = image_list[testing_indexes]
ID_testing = ID_list[testing_indexes]
testing_probabilities = probabilities_list[testing_indexes]

# Let us check the dimension of the arrays
print(training_images.shape, "\n")
print(testing_images.shape)
###################################################################################

""" Once loaded the data needed to train the model, let us define a custom image class to 
instance the images and probabililites as these objects. This way, we will be able to 
manipulate the dataset easier. """

################################# Custom class ####################################
# We create a custom Dataset class to work the images
class CustomImageDataset(Dataset):
    def __init__(self, images, ID, probabilities):
        """ The super() builtin returns a proxy object (temporary object of the superclass)
        that allows us to access methods of the base class."""
        super().__init__()
        self.images = images                      # Torch tensor
        self.ID = ID                              # ID array
        self.probabilities = probabilities        # Pandas dataframe
        
    # We redefine the __len__() method
    def __len__(self):
        return len(self.images)
    
    # We redefine the __getitem__() method
    def __getitem__(self, i):
        image = self.images[i]
        probabilities = self.probabilities[i]
        return image, probabilities

# We have to instantiate two different objects, one for the training images and the other for the testing images.
# Training
train_data = CustomImageDataset(training_images, ID_training, training_probabilities)

# Testing
test_data = CustomImageDataset(testing_images, ID_testing, testing_probabilities)
###################################################################################

""" Once we have loaded the CustomImageDataset objects, it is time to instantiate the 
Dataloader objects in order to get the proper inputs to the CNN. Also, it is possible 
to train the CNN in batches if the inputs are Dataloader objects. """

################################## DataLoader #####################################
# Size of the batch
batch_size = 1000

# Training DataLoader object
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# Validation DataLoader object
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
###################################################################################

""" Now we have loaded the datasets and we transformed them into appropiate Dataloader 
objects, it is time to define the model we will train to classificate the input galaxies. 
As said before, the model is a CNN, and the architecture of such a network will be 
explained right now. First, let us define the *hyperparametrs* of the model. """

#
#
#
#
#
#

# We define a function to compute the width and height of an output convolutional layer
def output_size(W, K, P, S):
    # W is the input width and height
    # K is the kernel size
    # P is the padding
    # S is the stride
    return ((W - K + 2*P)/S) + 1

print(output_size(124, 3, 1, 3))


# In[ ]:


# Model parameters
n_inputs = 12348
n_hidden_1 = 6174                 # 12348/2 = 6174
n_hidden_2 = 3087                 # 6174/2 = 3087
n_outputs = 37                    # Number of the components of the target vector
in_channels = 3
out_channels_1 = 5
out_channels_2 = 7
kernel_size_1 = 4
kernel_size_2 = 3
p_dropout = 0.1                   # Dropout probability
lr = 1e-3                         # Learning rate
n_epochs = 100                    # Number of epochs
#n_epochs = 300                   # Number of epochs


# In[ ]:


# Model definition
class Model(nn.Module):
    # Define model elements
    def __init__(self):
        super().__init__()
        # Sequence of transformations implemented by the layers of the network
        self.cnn = nn.Sequential(
            # Convolution layer. Convolution applyed to the input image. Stride = 1 and no padding
            nn.Conv2d(in_channels, out_channels_1, kernel_size_1, stride=2),
            # Activation function applyed to the convolutioned map
            nn.ReLU(),
            # Pooling layer. Max pooling function applyed to the ReLU-convolutioned map. No padding
            nn.MaxPool2d(kernel_size_1, stride=1),
            # Convolution layer. Convolution applyed to the maxpooled layer before. Stride = 1 and no padding
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size_2, stride=3, padding=1),
            # Activation function applyed to the convolutioned map
            nn.ReLU(),
            # Flattens a contiguous range of dims into a tensor
            nn.Flatten(),
            # Linear transformation of the flattened layer
            nn.Linear(n_inputs, n_hidden_1),
            # Activation function applyed to the convolutioned map
            nn.ReLU(),
            # Linear transformation applyed to the ReLU-transformed layer
            nn.Linear(n_hidden_1, n_hidden_2),
            # Rndomly zeroes some of the elements of the input tensor with probability p_dropout
            nn.Dropout(p_dropout),
            # Activation function applyed to the dropped out layer
            nn.ReLU(),
            # Linear transformation applyed to the ReLU-transformed layer 
            nn.Linear(n_hidden_2, n_outputs),
            # Softmax function applied to the linear transformed layer
            nn.Softmax()
        )
        
    # Method to transform inputs in outputs considering the internal structure of the network
    def forward(self, X):
        output = self.cnn(X)
        return output
    
# Now we can create a model and send it at once to the device
model = Model().to(device)
# We can also inspect its parameters using its state_dict() method
print(model.state_dict())
# We can check the architecture this way
print(model.parameters)


# <figure>
#     <img src="cnn_architecture.jpg" style="width:100%">
# </figure>

# Te architecture of the network is based in partially-connected layers (convolutional and pooling layers) and a fully-connected part. In the convolutional part, a *filter* composed of various *kernels* will move across the input, while doing the convolution (to learn more about convolutions see [here](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1#:~:text=Each%20filter%20in%20a%20convolution,a%20processed%20version%20of%20each.)). There will be one kernel per input channel and one filter per output channel. After each convolution, a ReLU function is applied to all the pixels in the different channels, except when the max pooling operation is applyed. After the last convolution, the 7 channels of 42x42 will be flattened, to enter the fully-connected part of the network. Then, after a series of linear, ReLUs and one dropping transformations, the last layer of 37 components is evaluted using a *softmax* function to get the desired probability distributions.

# Now, we have to define the function that will perform the training and testing of the CNN.

# In[ ]:


# We define the training function
def train_loop(dataloader, model, loss_fn, optimizer):
    #size = int(len(dataloader.dataset)/1000)
    size = int(len(dataloader.dataset)/batch_size)
    tmp = []

    # We iterate over batches
    for batch, (X, y) in enumerate(dataloader):
        # We calculate the model's prediction
        print(X.dtype)
        pred = model(X)
        # With the model's prediction we calculate the loss function
        loss = loss_fn(pred, y)

        # We apply the backpropagation method
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Training progress
        loss, current = loss.item(), batch
        tmp.append(loss)
        print(f"Actual batch = {current} | Loss = {loss:>7f} | Processed samples: [{current:>2d}/{size:>2d}]")
    
    tmp = np.array(tmp)
    loss_avg = tmp.sum()/len(tmp)
    return loss_avg

# We define the test function
def test_loop(dataloader, model, loss_fn):
    #size = int(len(dataloader.dataset)/1000)
    size = int(len(dataloader.dataset)/batch_size)
    test_loss = 0
    j = 0
    
    # To test, we need to deactivate the calculation of the gradients
    with torch.no_grad():
        # We iterate over batches
        for X, y in dataloader:
            # Model's prection
            pred = model(X)
            # Corresponding errors, which we acumulate in a total value
            test_loss += loss_fn(pred, y).item()
            j += 1
            
    # We calculate the total loss and print it
    test_loss /= j
    print(f"Test Error: Avg loss = {test_loss:>8f} \n")
    return test_loss


# In order to train the model, we need to instanciate an optimizer object and a loss function object. Let us do this.

# In[ ]:


# Loss function object. It is a Medium Squared Error.
loss_fn = nn.MSELoss()

# We instantiate an optimizer. In this case we choose an Adam optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08, weight_decay=0, amsgrad=False)


# In[ ]:


# Print model's state_dict size to gain some perspective about the model
print("Model's state_dict size:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# We will plot the loss function against the epochs, so, we need to save its value after each epoch of training is concluded.

# In[ ]:


# We define a loss array to plot the training loss function and the testing loss function
loss_to_plot = []
loss_to_plot_test = []


# We are ready to train the model. Let us train it during $n_{epochs}$ epochs, as defined above.

# In[ ]:


# We train the model iterating over the different epochs
for t in tqdm(range(n_epochs)):
    print(f"Epoch {t+1}\n=============================================")
    loss_to_plot.append(train_loop(train_dl, model, loss_fn, optimizer))
    loss_to_plot_test.append(test_loop(test_dl, model, loss_fn))
print("Done!")


# Since we have our model trained, now we need to write a function able to convert the probability distributions predicted by the network in words. The 11 questions shown at the beggining used to define the decision tree has each one a set of answers. Let us convert the predicted probabilites in those answers.

# In[ ]:


# We save both loss functions
np.savetxt('loss_to_plot.txt', loss_to_plot)
np.savetxt('loss_to_plot_test.txt', loss_to_plot_test)

# We choose an image and calculate the corresponding prediction generated by the model
for (X, y) in test_dl:
    pred_cpu = model(X)
    image_cpu = X[7]
    target = y
    break

pred_cpu = pred_cpu[7].detach().numpy()
target = target[7].detach().numpy()

# We plot the image to be predicted and as a title the corresponding prediction
fig, ax = plt.subplots(1, 1, dpi=280)
fig.set_size_inches(4.0, 4.0)
ax.axis("off")
#plt.title(labels_map[np.argmax(pred_cpu)])
ax.imshow(image_cpu.squeeze().reshape(256, 256, 3), cmap="gray")

print("Prediction", "\t", "Target", "\n", "_________________________", "\n")
for (pred, tar) in zip(pred_cpu, target):
    print(pred, "\t", tar)


# Let us plot the loss function as a function of the epochs.

# In[ ]:


# Load both loss functions
lp = np.loadtxt('loss_to_plot.txt')
lp_test = np.loadtxt('loss_to_plot_test.txt')

# Let us plot both loss functions
fig, ax = plt.subplots(1, 1, figsize=(7,7), dpi=200)
ax.plot([i for i in range(3, n_epochs+1)], lp[2:], color='darkblue', lw=1.5, label='Training error')
ax.plot([i for i in range(3, n_epochs+1)], lp_test[2:], ls=':', color='darkblue', lw=1.5, label='Test error')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average loss function')
#ax.set_xticks([0, 75, 150, 225, 300])
#ax.set_xticklabels(['0', '75', '150', '225', '300'])
#y_ticks = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045]
#ax.set_yticks(y_ticks)
#ax.set_yticklabels([str(y_ticks[i]) for i in range(len(y_ticks))])
plt.legend()
plt.show()
#plt.savefig('loss.jpg', bbox_inches='tight')

