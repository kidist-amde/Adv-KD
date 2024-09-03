from prodm.data.distllation_data import get_transform , DisttlationDataset
from prodm.models.proddpm_student import StudentModel
from torch import nn , optim 
from torch.utils.data import DataLoader 
import torch
from tqdm.auto import tqdm


class SoftBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,y_hat,y):
        batch_size = y.size(0)
        bce_loss = -(y*torch.log(y_hat) +(1-y)*torch.log(1-y_hat))
        return bce_loss.mean()
def log_metric(log_path,iteration,time_step,epoch,loss):
    with open(log_path,mode="a")  as output_file:
        output_file.write("{},{},{},{:.4f}\n".format(iteration,time_step,epoch,loss))

def train(model,data_loader,optimizer,critrion,time_step,epochs,log_path,iteration):
    device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    for epoch in range(epochs):
        epoch_loss= 0
        total = 0
        for inputs,targets in tqdm(data_loader,total=len(data_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs,time_step)           
            loss  = critrion(outputs,targets)
            optimizer.zero_grad()
            loss.backward()
            epoch_loss+=loss.item()*inputs.size(0)
            total+=inputs.size(0)
            optimizer.step()
        print(f"iteration:{iteration} time_step:{time_step} epoch:{epoch} loss:{epoch_loss/total:.4f}" )
        log_metric(log_path,iteration,time_step,epoch,loss.item())
        # scheduler.step()

def main():
    transform = get_transform()
    dataset_path = "../dataset"
    # model = StudentModel()
    model = torch.load("../logs/model-9.pt")
    log_path = "../logs/log.txt"
    with open(log_path,mode="w")  as output_file:
        output_file.write("iteration,time_step,epoch,loss\n")
    for iteration in range(10):
        for time_step in range(1,11):
            dataset = DisttlationDataset(dataset_path,time_step,transform)
            parameters = []
            for i in range(time_step+1):
                parameters.extend(model.blocks[i].parameters())
            # for param in model.parameters():
            #     param.requires_grad=False
            # for param in model.blocks[time_step].parameters():
            #     param.requires_grad=True
            optimizer = optim.Adam(parameters,lr=1e-4)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*(time_step+1), eta_min=1e-5)
            critrion = nn.BCEWithLogitsLoss()
            if time_step<8:
                batch_size=64
            elif time_step<10:
                batch_size=32
            else:
                batch_size=16
            data_loader = DataLoader(dataset,batch_size = batch_size)
            train(model,data_loader,optimizer,critrion,time_step,epochs=5*(time_step+1), log_path=log_path,iteration=iteration)    
            torch.save(model,f"../logs/model-{time_step}.pt")
            
if __name__=="__main__":
    main()
            