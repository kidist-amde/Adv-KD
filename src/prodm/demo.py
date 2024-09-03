from  PIL import  Image
import torch 
from torchvision import transforms
import matplotlib.pyplot as plt




def main():
    device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model=torch.load("../logs/model.pt")
    noise=torch.randn(1,3,4,4).to(device)
    image= model(noise,9).squeeze(0)
    transform = transforms.ToPILImage()
    image = transform(image)
    image.save("../image.png")


if __name__=="__main__":
    main()