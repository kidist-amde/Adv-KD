import PIL
import os
import torch 
import gdown
import numpy as np
import shutil
from torch.utils.data import DataLoader
from dkd.utils.util import get_transforms

UTKFACE_URL = "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk" 


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="../data", train=True, download=False, transforms = None):
        self.transforms = transforms
        self.root_dir = root_dir
        if download:
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)
            archive_path = os.path.join(self.root_dir,"UTKFace.tar.gz")
            gdown.download(UTKFACE_URL,quiet=False,output=archive_path)
            shutil.unpack_archive(archive_path,self.root_dir)
        all_files = os.listdir(os.path.join(self.root_dir,"UTKFace"))
        all_files.sort()
        np.random.seed(42)
        indices = np.arange(len(all_files))
        np.random.shuffle(indices)
        train_size = int(len(all_files)*0.8)
        if train:
            self.file_names = [all_files[i] for i in range(train_size)]
        else:
            self.file_names =  [all_files[i] for i in range(train_size,len(all_files))]

        """
        Initialize other attributes of the dataset as well.
        """

        # Assign race to an index which will be the label for each of the race classes
        self.race2index = {
            "White": 0,
            "Black": 1,
            "Asian": 2,
            "Indian": 3,
            "Others": 4 }
    def __len__(self):
        """This should be the length of the dataset. It is probably the len(image_files)
        """
        return len(self.file_names)
    def __getitem__(self, index):
        """Get the image and target at given index

        Args:
            index (int): The index of the data point
        """
        
        # First get the image file at given index. something like image_file = self.image_files[...]

        image_file = self.file_names[index]
        
        # Now you can get the path to the image. You can use os.path.join to join the self.root_dir and the image_file
        image_path = os.path.join(self.root_dir,"UTKFace",image_file)

        # load the images 
        image = PIL.Image.open(image_path)          

        # You can also get the race from the image 

        if image_file.count("_") == 3:
            age, gender, label, _ = image_file.split("_")
        else:
            label = "0" # If the file is not in the correct format set the race of the image to White(Most frequent faces)

                            # are following the same naming as that of the website you can remove this line


        label = int(label)

        if self.transforms is not None: # If transforms are given apply
            image = self.transforms(image)

        return image, label
def get_data(root_dir,image_size,batch_size):
    transforms = get_transforms(image_size)
    dataset = UTKFaceDataset(root_dir,train=True,download=True,transforms=transforms["train"])
    dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)
    return dataloader