import logging
import torch
import os
from torch import nn , optim
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from .util import setup_logging
from .sampling import save_images
from dkd.models.diffussion import Diffusion
logging.basicConfig(format= '%(asctime)s - %(levelname)s:(message)s',level = logging.INFO,datafmt="%I:%M%S")



def train(model,dataloader,image_size,lr,epochs,device,start_epoch = 0,log_dir="./logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    run_name = "DDPM_UTKFace"
    setup_logging(log_dir,run_name)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", run_name))
    l = len(dataloader)
    # we follow the algorithmn of the DDPM paper 
    for epoch in range(start_epoch,epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            # sample timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # noise the images 
            x_t, noise = diffusion.noise_image(images, t)
            # feed the niosied image to the model to predict the nois added to the images
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging 
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join(log_dir,"logs", f"{epoch}.jpg"))
        # save the model
        torch.save(model.state_dict(), os.path.join(log_dir,"models" ,f"ckpt.pt"))