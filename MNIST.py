import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,input):
        x=self.flatten(input)
        logits=self.linear_relu(x) #cross entropy applies softmax then takes NLL
        return logits


class Initialize:
    def __init__(self,batch_size=64,):
        
        self.batch_size=batch_size
        #initialize any transforms and augmentation
        self.transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
        #initialize train and test data 
        self.train_data=torchvision.datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=self.transforms
        )
        self.test_data=torchvision.datasets.MNIST(
            root="data",
            train=False,
            transform=self.transforms,
            download=True
        )
        #initialize dataloaders
        self.train_dataloader=DataLoader(self.train_data,batch_size=self.batch_size,shuffle=True,num_workers=0)
        self.test_dataloader=DataLoader(self.test_data,batch_size=self.batch_size,shuffle=False,num_workers=0)
        #initialization stats
        print(f"Training Set size = {len(self.train_data)}\nTest Set Size = {len(self.test_data)}")
        print(f"Number of batches in training set = {len(self.train_dataloader)}\nNumber of batches in test set = {len(self.test_dataloader)}")

        #look at image sizes
        for X,y in self.test_dataloader:
            print(f"Shape of X [N,C,H,W]: {X.shape}")
            print(f"Shape of y: {y.shape} {y.dtype}")
            break

class Control(Initialize):
    def train(self,model,epochs):  
        

        loss_fn=nn.CrossEntropyLoss()
        opt=torch.optim.Adam(model.parameters(),lr=0.001)
        
        for epoch in range(epochs):
            
            running_loss=0.

            for input,labels in self.train_dataloader:
                input,labels=input.to(device),labels.to(device)
                opt.zero_grad()
                logits=model(input)
                loss=loss_fn(logits,labels)
                loss.backward()
                opt.step()

                running_loss+=loss.item()
            
            print(f"Loss after epoch {epoch+1}, Loss : {running_loss/len(self.train_dataloader):.4f}")


    def evaluate(self,model):
        model.eval()
        loss_fn=nn.CrossEntropyLoss()
        running_test_loss=0.
        total=0
        correct=0
        with torch.no_grad():
            for test_input,test_labels in self.test_dataloader:
                #load test data onto gpu
                test_input,test_labels=test_input.to(device),test_labels.to(device)
                #model inference
                model_preds=model(test_input)
                preds=model_preds.argmax(0)
                #compute loss for test set
                test_loss=loss_fn(preds,test_labels)
                #add to running test loss
                running_test_loss+=test_loss.item()
                #count total number of examples processed till now
                total+=test_labels.shape[0]
                #count correct examples out of this batch
                correct += (model_preds==test_labels).sum().item()

        test_accuracy=(correct/total)*100
        avg_test_loss=running_test_loss/len(self.test_dataloader)
        print(f"Avg accuracy on test set = {test_accuracy:.2f}\nAvg test loss : {avg_test_loss:.2f}")


if __name__ == "__main__":

    model= Network().to(device)

    print(model)

    print(f"Total network params = {sum(p.numel() for p in model.parameters())}")