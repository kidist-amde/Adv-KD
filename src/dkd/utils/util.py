# image preprocessing/ transformation / convert the PIL Imnage to tensor format
from torchvision import transforms
import os

def get_transforms(image_size):
    train_transformation = transforms.Compose([ 
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        # convert images to tensor / scales data into [-1 1]
        transforms.ToTensor(),
        # Scale between [-1, 1] inorder to work with Betas 
        transforms.Lambda(lambda t: (t * 2) - 1)                                     
        ])

    test_transformation = transforms.Compose([ 
        transforms.Resize((image_size,image_size)),
        # convert images to tensor / scales data into [-1 1]
        transforms.ToTensor(),
        # Scale between [-1, 1] inorder to work with Betas 
        transforms.Lambda(lambda t: (t * 2) - 1)                                     
        ])
    return {"train":train_transformation,"test":test_transformation}

# setting up the folder fro saving the model and the result 
def setup_logging(log_dir,run_name):
    os.makedirs(os.path.join(log_dir,"models"), exist_ok=True)
    os.makedirs(os.path.join(log_dir,"results"), exist_ok=True)
    os.makedirs(os.path.join(log_dir,"models", run_name), exist_ok=True)
    os.makedirs(os.path.join(log_dir,"results", run_name), exist_ok=True)