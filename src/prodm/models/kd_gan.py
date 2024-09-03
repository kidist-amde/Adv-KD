from .proddpm_student  import StudentModel
from torch import nn
import IPython
import torch

class Discriminator(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        ndf = 32
        blocks = [nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                nn.LeakyReLU(0.2, inplace=True),)]
        i = 0
        output_channels=ndf
        while input_size > 8:
            output_channels=ndf
            if output_channels < 128:
                output_channels *=2
            blocks.append(self.get_single_block(ndf, output_channels))
            input_size = input_size//2
            if ndf<128:
                ndf*=2    
        # blocks.append(nn.Sequential(nn.Conv2d(output_channels, 1, 4, 1, 0, bias=False)))            
        self.layers = nn.Sequential( *blocks)
        self.noise_layers = nn.Sequential(
                nn.Conv2d(3, 32, 3,1,1, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(32, 64, 3, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True))
        self.common_layers = nn.Sequential(
            nn.Linear(output_channels*4*4+64*2*2,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,1))
                       
        
    def get_single_block(self,in_channels,output_channels):
       return nn.Sequential(
                nn.Conv2d(in_channels, output_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self,inputs,noise):
        batch_size = inputs.size(0)
        image_outputs = self.layers(inputs).view(batch_size,-1)
        noise_outputs = self.noise_layers(noise).view(batch_size,-1)
        outputs = self.common_layers(torch.concat([image_outputs,noise_outputs],dim=1)) 
        return outputs
    
class KDGAN(object):
    def __init__(self,student_model,time_step):
        input_size = self.get_input_sizeof_timesteps(time_step)
        self.time_step = time_step
        self.discriminator = self.create_discriminator(input_size)
        self.student_model = student_model
        
    
    def get_input_sizeof_timesteps(self,time_step):
        curent_size = 2**(time_step//2+3)
        return curent_size
         
    def create_discriminator(self,input_size):
        discriminator = Discriminator(input_size)
        return discriminator


if __name__ == "__main__":
    student_model = StudentModel()
    gan = KDGAN(student_model,10)
    IPython.embed()
            

        
        
        