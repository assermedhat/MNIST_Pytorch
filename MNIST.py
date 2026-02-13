import torch
import torchvision
import torch.nn as nn
# from torch.utils.tensorboard import Summarywriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt 
import numpy as np
import torchmetrics


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
        # self.writer=Summarywriter()
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
    
    def __init__(self,model,lr=0.001, retrain=False,chpoint_path="mnist.pth"):
        super().__init__()
        self.model=model
        self.retrain=retrain
        self.loss_fn=nn.CrossEntropyLoss()
        self.opt=torch.optim.Adam(model.parameters(),lr=lr)
        self.avg_train_loss=0
        self.checkpoint=chpoint_path
        self.metric=torchmetrics.Accuracy(task="multiclass",num_classes=10).to(device)


    def train(self,epochs):  
        if self.retrain:  
            losses=[]  
            for epoch in range(epochs):
                self.model.train()
                running_loss=0.

                for input,labels in self.train_dataloader:
                    input,labels=input.to(device),labels.to(device)
                    self.opt.zero_grad()
                    logits=self.model(input)
                    loss=self.loss_fn(logits,labels)
                    loss.backward()
                    self.opt.step()

                    running_loss+=loss.item()
                epoch_loss=running_loss/len(self.train_dataloader) #total batches loss / total no of batches
                losses.append(epoch_loss)
                
                print(f"Loss after epoch {epoch+1}, Loss : {epoch_loss:.4f}")
            self.avg_train_loss=float(np.mean(losses))
            print(f"Avg training loss = {self.avg_train_loss:.2f}")
            #saves model params for later use
            torch.save(self.model.state_dict(),self.checkpoint)
        else:
            
            print("Model already trained and ready for evaluation")
            #laods model parameters from the path that it was saved to
            self.model.load_state_dict(torch.load(self.checkpoint,map_location=device))


    def evaluate(self,model):
        model.eval()
        loss_fn=nn.CrossEntropyLoss()
        running_test_loss=0.
        total=0
        correct=0

        with torch.no_grad(): #disables gradient calculation for any operation within its scope
            for test_input,test_labels in self.test_dataloader:
                #load test data onto gpu
                test_input,test_labels=test_input.to(device).float(),test_labels.to(device)
                #model inference
                model_preds=self.model(test_input)
                preds=model_preds.argmax(1)
                #compute loss for test set
                test_loss=loss_fn(model_preds,test_labels)
                #add to running test loss
                running_test_loss+=test_loss.item()
                #count total number of examples processed till now
                total+=test_labels.shape[0]
                #count correct examples out of this batch
                correct += (preds==test_labels).sum().item()
                self.metric.update(model_preds,test_labels)
            test_accuracy_manual=(correct/total)*100
            
            avg_test_loss=running_test_loss/len(self.test_dataloader)
            print(f"Average Test accuracy manual = {test_accuracy_manual:.2f}")
            print(f"Avg accuracy on test set using metric = {self.metric.compute().item()*100:.2f}\nAvg test loss : {avg_test_loss:.2f}")
        
        


if __name__ == "__main__":

    model= Network().to(device)
    Orchestrator=Control(model,lr=0.001)
    Orchestrator.train(epochs=10)
    Orchestrator.evaluate(model)

   