import torch
from dkd.data.utkface import get_data
from dkd.models.unet import UNet
from dkd.utils.train import train

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    root_dir = "data"
    image_size=64
    batch_size = 4
    lr = 3e-4
    epochs = 1000
    start_epoch = 0
    log_dir = "./logs"
    dataloader = get_data(root_dir,image_size,batch_size)
    model = UNet(device=device).to(device)
    train(model,dataloader,image_size,lr,epochs,device,start_epoch = start_epoch,log_dir=log_dir)
        
    

if __name__=="__main__":
    main()
    