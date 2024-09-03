from diffusers import DDPMPipeline 
import torch
from tqdm.auto import tqdm

class MyDDPMPipeline(DDPMPipeline):

    @torch.no_grad()
    def _post_process_image(self, image, output_type):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        return image

    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, torch_device=None, output_type="pil", image_at_every = 100):
        """
        Modication of original ddpm pipeline to get mulitple intermediate images. image_at_every parameters is used
        to control the number of output images
        """
        if torch_device is None:
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.unet.to(torch_device)

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator)

        image = image.to(torch_device)

        # set step values
        self.scheduler.set_timesteps(1000)
        outputs = [self._post_process_image(image, output_type)]

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t)["sample"]

            # 2. compute previous image: x_t -> t_t-1
            image = self.scheduler.step(model_output, t, image)["prev_sample"]
            if (t + 1) % image_at_every == 0:
                outputs.append(self._post_process_image(image, output_type))

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        outputs.append(image)
        return {"sample": image, "intermediate": outputs} 