import torch
import logging
from tqdm.auto import tqdm
logging.basicConfig(format= '%(asctime)s - %(levelname)s:(message)s',level = logging.INFO,datefmt="%I:%M%S")

class Diffusion:
      # use the same timestep and beta value as the orginal paper and fair resolution for the image due to compuational limitation 
  def __init__(self,noise_steps = 1000,beta_start =1e-4,beta_end = 0.02,image_size = 64, device = 'cuda'):
    self.noise_steps = noise_steps
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.image_size = image_size
    self.device = device 
    # we use linear schdule 
    self.beta = self.prepare_noise_schedule().to(device)
    self.alpha = 1-self.beta 
    self.alpha_hat  = torch.cumprod(self.alpha,dim=0)

  def prepare_noise_schedule(self):
    return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)

  # to noise images 
  def noise_image(self,x,t):
    sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None] # size (BCHW)
    sqrt_one_minus_alph_hat = torch.sqrt(1-self.alpha_hat[t])[:,None,None,None]
    epslon = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alph_hat *epslon , epslon 

  # sample timesetps 
  def sample_timesteps(self,n):
    return torch.randint(low = 1,high=self.noise_steps,size=(n,))
  # sampling function which takes the model used for sampling and number of image we want to sample  
  # we follow the sampling algorithm from the DDPM paper 

  def sample(self, model,n):
    logging.info(f"sampling {n} new images...")
    model.eval()
    with torch.no_grad():
      x = torch.randn((n,3,self.image_size,self.image_size)).to(self.device)
      # going overall thw 1000 steps in reversed order 
      for i in tqdm(reversed(range(1,self.noise_steps)),total=self.noise_steps):
        t = (torch.ones(n)*i).long().to(self.device)
        # the modelexpect the image and the time step
        predicted_noise = model(x, t)
        alpha = self.alpha[t][:, None, None, None]
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        beta = self.beta[t][:, None, None, None]
        # we don't want to eddit in the last itteration because it will make the final outcome worse
        if i > 1:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
      model.train()
      # put the out put in the valid range of [-1,1] and bring back the value b/n the range of  0 and 1 
      x_max = torch.amax(x,dim=[1,2,3],keepdim=True)
      x_min = torch.amin(x,dim=[1,2,3],keepdim=True)
      x = (x-x_min)/x_max

      # bring them to the valid pixel range and change the data tyope for saving them latter 
      x = (x * 255).type(torch.uint8)
      
      return x

      
