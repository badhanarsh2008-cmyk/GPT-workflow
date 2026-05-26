import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device :",device)

transform = transforms.ToTensor()

train_data = MNIST(
    root="data",
    download=True,
    train=True,
    transform = transform
)

test_data = MNIST(
    root="data",
    download=True,
    train=False,
    transform = transform
)

train_loader = DataLoader(train_data, 
                shuffle=True, 
                batch_size=64)

test_loader = DataLoader(train_data, 
                shuffle=True, 
                batch_size=64)

#Brain of model.
class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
                #input layer
                nn.Linear(28 * 28, 512),
                nn.ReLU(),  #activation layer
                
                nn.Linear(512, 256), #hidden layer ( for compressing data ) neural network
                nn.ReLU(),          #hidden layer
                
                nn.Linear(256,128), #extra added layer by myself
                nn.ReLU(),
                
                nn.Linear(128,64),
                nn.ReLU(),
                
                nn.Linear(64,10),   #we are telling it predict 1-9
                nn.ReLU()           #output layer
            )
    def forward(self,x):
        return self.net(x)
    
model = mymodel().to(device)

#after model creation lets devlare some imp things like loss_fn ( for loss calc )  and optimizer ( to frequently change the weight )
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

#lets start training the model.

epochs = 7 #gonna train model 5 times


#model training
for epoch in range(epochs):
    model.train() #start training
    total_loss = 0
    for images, labels in train_loader:     #to get image and label from data
        images = images.view(images.size(0), -1).to(device)     #taking images
        labels = labels.to(device)
        
        optimizer.zero_grad()   #reseting gradient to zero 
        outputs= model(images)  #taking output
        loss = loss_fn(outputs,labels)  #calc loss 
        loss.backward()     #checking which value is cause of wrong output
        optimizer.step()    #change the valur of learning rate and weight acc to gradient ( loss backward gives gradient )
        
        total_loss += loss.item()       #calc total loss
        
    print(f"Epoch {epoch + 1}/{epochs} | Loss : {loss.item():.2f}")        #printing iterations
    
model.eval()
correct = 0 
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        prediction = outputs.argmax(dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)
        
    accuracy = 100 * (correct/total)
    print(f"Accuracy = {accuracy:.2f}")
    

index = random.randint(0, 2000)
image, true_label = test_data[index]

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Actual label : {true_label}")
plt.axis("off")
plt.show()

image_flat = image.view(1, -1).to(device)

with torch.no_grad():
    output = model(image_flat)
    predicted_label = output.argmax(dim=1).item()

print("User picked picture index :", index)
print("Actual label :",true_label)
print("predicted label :",predicted_label)
print("Accuracy :",accuracy)