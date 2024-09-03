from prodm.data.distllation_data import get_transform , DisttlationDataset
from prodm.models.proddpm_student import StudentModel 
from prodm.models.kd_gan import KDGAN  
from torch import nn , optim 
from torch.utils.data import DataLoader 
import torch ,os
import numpy as np
from torch.nn import functional as F
from torch import autograd
from torchvision import transforms

from tqdm.auto import tqdm
# Log in to your W&B account
import wandb
wandb.login()

def compute_gradient_penalty(D,noise, real_samples, fake_samples,device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates,noise)
    fake = autograd.Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def log_metric(log_path,time_step,epoch,discriminator_loss,student_loss):
    with open(log_path,mode="a")  as output_file:
        output_file.write("{},{},{:.4f},{:.4f}\n".format(time_step,epoch,discriminator_loss,student_loss))

def train(kd_gan,data_loader,optimizerD,optimizerG,criterion,time_step,wandb,log_path):
    config = wandb.config
    device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    kd_gan.student_model.to(device)
    kd_gan.discriminator.to(device)
    weight_cliping_limit = 0.1
    # Loss weight for gradient penalty
    lambda_gp = 10
    noise=torch.randn(5,3,4,4).to(device)
    for epoch in range(config["epochs"]):
        discriminator_loss= 0
        student_loss = 0
        total = 0
        for batch_index,(inputs,teacher_outputs) in tqdm(enumerate(data_loader),total=len(data_loader)):
            inputs = inputs.to(device)
            teacher_outputs = teacher_outputs.to(device)
            # student_inputs = torch.randn_like(inputs)
            student_outputs = kd_gan.student_model(inputs,time_step) 
            for param in kd_gan.discriminator.parameters():
                param.requires_grad = True
            kd_gan.discriminator.zero_grad()
            # create real label 
            # real_labels = torch.ones(teacher_outputs.size(0)).to(device)
            real_outputs = kd_gan.discriminator(teacher_outputs,inputs)    
            # real_loss  = criterion(real_outputs,real_labels)
            # create fake label
            # fake_labels = torch.zeros(student_outputs.size(0)).to(device)
            fake_outputs = kd_gan.discriminator(student_outputs,inputs)
            # fake_loss = criterion(fake_outputs,fake_labels)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(kd_gan.discriminator,inputs, teacher_outputs.data, student_outputs.data,device)
            d_real_loss = F.softplus(-real_outputs).mean()
            d_fake_loss = F.softplus(fake_outputs).mean()
            # Adversarial loss
            disc_loss = d_real_loss + d_fake_loss+lambda_gp * gradient_penalty
            disc_loss.backward()
            optimizerD.step()
            for p in kd_gan.discriminator.parameters():
                p.data.clamp_(- weight_cliping_limit, weight_cliping_limit)
            for param in kd_gan.discriminator.parameters():
                param.requires_grad = False
            # optimize the student model
            kd_gan.student_model.zero_grad()
            # student_inputs = torch.randn_like(inputs)
            student_outputs = kd_gan.student_model(inputs,time_step)  
            # fake_labels = torch.ones(student_outputs.size(0)).to(device)
            fake_outputs = kd_gan.discriminator(student_outputs,inputs)
            gen_loss = F.softplus(-fake_outputs).mean()
            gen_loss.backward()
            optimizerG.step()
            discriminator_loss +=disc_loss.item()*inputs.size(0)
            student_loss += gen_loss.item()*inputs.size(0)
            total+=inputs.size(0)
        
        student_images= kd_gan.student_model(noise,time_step).squeeze(0)
        transform = transforms.ToPILImage()
        student_images = [wandb.Image(transform((image+1)/2)) for image in student_images]
        teacher_images = [wandb.Image(transform((image+1)/2)) for image in teacher_outputs[:5]]
        print(f" time_step:{time_step} epoch:{epoch} disc_loss:{discriminator_loss/total:.4f},stud_loss:{student_loss/total:.4f}" )
        log_metric(log_path,time_step,epoch,discriminator_loss/total,student_loss/total)
        metrics = {"discriminator_loss":discriminator_loss/total,"student_loss":student_loss/total,"student_images":student_images,"teacher_images":teacher_images}
        wandb.log(metrics)

def main():
    transform = get_transform()
    dataset_path = "../dataset"
    student_model = StudentModel()
    # student_model = torch.load("../logs_gan/student_model-8.pt")
    if not os.path.exists("../logs_gan"):
        os.mkdir("../logs_gan")
    log_path = "../logs_gan/log.txt"
    with open(log_path,mode="w")  as output_file:
        output_file.write("time_step,epoch,discriminator_loss,student_loss\n")
    for time_step in range(0,11):
        wandb.init(
        # Set the project where this run will be logged
        project="ProDDM", 
        # pass a run name
        name=f"KD-CWGAN-exp1-{time_step}", 
        # Track hyperparameters and run metadata
        config={
        "dlearning_rate": 0.0001, 
        "glearning_rate":0.0001,
        "beta1": 0.5,
        "epochs": 10*(time_step+1),
        })
        config = wandb.config
        dataset = DisttlationDataset(dataset_path,time_step,transform)
        kd_gan = KDGAN(student_model,time_step)
        if time_step<8:
            batch_size=64
        elif time_step<9:
            batch_size=32
        else:
            batch_size=16
        data_loader = DataLoader(dataset,batch_size = batch_size,shuffle=True,drop_last=True)
        # Initialize BCELoss function
        criterion = nn.BCELoss()
        for param in student_model.parameters():
            param.requires_grad=False
        trainable_params = []
        for i in range(time_step+1):
            trainable_params.extend(student_model.blocks[i].parameters())
        for param in trainable_params:
            param.requires_grad=True
        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(kd_gan.discriminator.parameters(), lr=config["dlearning_rate"], betas=(config["beta1"], 0.999))
        optimizerG = optim.Adam(trainable_params, lr=config["glearning_rate"],betas=(config["beta1"], 0.999))
        train(kd_gan,data_loader,optimizerD,optimizerG,criterion,time_step,wandb,log_path=log_path)    
        torch.save(student_model,f"../logs_gan/student_model-{time_step}.pt")
        wandb.finish()
        
if __name__=="__main__":
    main()
            