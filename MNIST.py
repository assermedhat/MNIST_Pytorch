import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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
        self.writer=SummaryWriter()
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
        images,labels=next(iter(self.train_dataloader))
        self.writer.add_images('training images',images,0)
        self.writer.close()
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
        self.opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
        self.avg_train_loss=0
        self.checkpoint=chpoint_path
        self.metric=torchmetrics.Accuracy(task="multiclass",num_classes=10).to(device)

#one iteration over training set
    def train_one_epoch(self):  
        self.model.train()
        running_loss=0.0
        self.metric.reset()
        for input,labels in self.train_dataloader:
            #load training data onto gpu
            input,labels=input.to(device),labels.to(device)
            #reset gradients 
            self.opt.zero_grad()
            #forward pass
            logits=self.model(input)
            #compute loss
            loss=self.loss_fn(logits,labels)
            #backprop
            loss.backward()
            #GD step
            self.opt.step()
            #accuracy
            self.metric.update(logits,labels)
            #running loss for each batch for average loss over all batches calulcation
            running_loss+=loss.item()
        
        epoch_loss=running_loss/len(self.train_dataloader) #total batches loss / total no of batches
        train_acc=self.metric.compute().item()
        return epoch_loss,train_acc
        
#one evaluation iteration over test set
    def evaluate(self):
        model.eval()
        loss_fn=nn.CrossEntropyLoss()
        running_test_loss=0.
        self.metric.reset()

        with torch.no_grad(): #disables gradient calculation for any operation within its scope
            for i,(test_input,test_labels) in enumerate(self.test_dataloader):
                #load test data onto gpu
                test_input,test_labels=test_input.to(device).float(),test_labels.to(device)
                #model inference
                model_preds=self.model(test_input)
                #compute loss for test set
                test_loss=loss_fn(model_preds,test_labels)
                #add to running test loss
                running_test_loss+=test_loss.item()
                #compute accuracy
                self.metric.update(model_preds,test_labels)

        avg_test_loss=running_test_loss/len(self.test_dataloader)
        test_acc=self.metric.compute().item()
        return avg_test_loss,test_acc
    
    def fit(self,epochs):
        if not self.retrain:
            print("Model already trained and ready for evaluation")
            self.model.load_state_dict(torch.load(self.checkpoint,map_location=device))
            return
        train_losses=[]
        for epoch in range(epochs):
            train_loss,train_acc=self.train_one_epoch()
            test_loss,test_acc=self.evaluate()
            train_losses.append(train_loss)

            #tensorboard visualization
            self.writer.add_scalar('Loss/train',train_loss,epoch)
            self.writer.add_scalar('Loss/test',test_loss,epoch)
            self.writer.add_scalar('Accuracy/train',train_acc,epoch)
            self.writer.add_scalar('Accuracy/test',test_acc,epoch)

            #print metrics

            print(f"Epoch {epoch+1}/{epochs} : "
                  f"train loss = {train_loss:.4f} , acc train = {train_acc*100:.2f}"
                  f"test loss {test_loss:.4f} , test acc = {test_acc*100:.2f}")
        self.avg_train_loss =  float(np.mean(train_losses))  
        print(f"Average train loss = {self.avg_train_loss:.4f}")
        torch.save(self.model.state_dict(),self.checkpoint)


    def log_test_preds(self,num_imgs=10,step=0):
        self.model.eval()
        images,labels=next(iter(self.test_dataloader))
        images,labels=images.to(device),labels.to(device)

        with torch.no_grad():
            logits=self.model(images)
            preds=logits.argmax(1)

        images=images.cpu()
        preds=preds.cpu()
        labels=labels.cpu()

        fig = plt.figure(figsize=(num_imgs*3,4))

        for i in range(num_imgs):
            ax= fig.add_subplot(1,num_imgs,i+1)
            ax.imshow(images[i].squeeze(),cmap="gray") 
            ax.set_title(f"P:{preds[i].item()} A:{labels[i].item()}")
            ax.axis("off")

        self.writer.add_figure("Test Predictions",fig,global_step=step)       
        


if __name__ == "__main__":

    model= Network().to(device)
    Orchestrator=Control(model,lr=0.001)
    Orchestrator.fit(epochs=8)
    test_loss,test_acc=Orchestrator.evaluate()
    print(F"Final test_loss = {test_loss:.4f}\nFinal test acc = {test_acc*100:.2f}")
    Orchestrator.log_test_preds(step=0)

   