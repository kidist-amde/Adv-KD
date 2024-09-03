from torch.utils.data import Dataset 
import os 
from PIL import  Image
from torchvision import transforms
from torchvision.transforms import functional as F


def get_transform():
    train_transformation = transforms.Compose([ 
        transforms.ToTensor(),
        # b/n -1 and 1
        transforms.Lambda(lambda t: (t * 2) - 1)   
                                          
        ])
    return train_transformation

class DisttlationDataset(Dataset):
    """create dataset for the student model"""
    def __init__(self,dataset_path,time_step,transform,img_persample=10):
        self.time_step = time_step
        self.dataset_path = dataset_path
        self.transform = transform
        self.img_persample = img_persample
        self.items = os.listdir(dataset_path)
    def __len__(self):
        
        return len(self.items)

    def __getitem__(self,idx):
        item =  self.items[idx]
        curent_size = 2**(self.time_step//2+3)
        input = Image.open(os.path.join(self.dataset_path,item,"0.png"))
        input  = F.resize(input,4)
        input = self.transform(input)
        image = Image.open(os.path.join(self.dataset_path,item,f"{self.time_step+1}.png"))
        image = F.resize(image,curent_size)
        image = self.transform(image)
        
        return input,image


if __name__=="__main__":
    transform = get_transform()
    dataset_path = "../dataset"
    dataset = DisttlationDataset(dataset_path,transform)
    inputs,outputs = dataset[0]
 