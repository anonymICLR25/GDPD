import torch
from torch import nn
import torch.nn.functional as F
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler
import math

class DiffKD_conditional(nn.Module): #pass both output loits so that both refined feature and unrefined feature 
    def __init__(
            self,
            student,
            mode,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.student=student
        # self.task_loss = task_loss
        self.mode=mode
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as the teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.diffmodel = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.diffmodel, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, input, teacher_feat=None):
        #get student features
        output_logits_init, student_feat, _, _, _= self.student(input,get_ha=True)

        #make feature maps BCHW format:
        student_feat=self._reshape_BCHW(student_feat)

        
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
 
        refined_feat = self.proj(refined_feat)#check this line , what proj exactly done, no use inside pipeline
        refined_feat_temp = refined_feat.squeeze(-1).squeeze(-1) #from BCHW to BC back
        output_logits = self.student(input.clone(), context=refined_feat_temp)
        
        
        if self.training:
            teacher_feat=self._reshape_BCHW(teacher_feat)
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None
            # train diffusion model
            ddim_loss = self.ddim_loss(teacher_feat)
            feature_loss = F.mse_loss(refined_feat, teacher_feat)
            # return output_logits, output_logits_init, refined_feat, teacher_feat, ddim_loss+feature_loss, rec_loss
            return output_logits, output_logits_init, refined_feat, teacher_feat, ddim_loss, rec_loss
            
        else:
            return output_logits


    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.diffmodel(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
        
        
class DiffKD_Network_variation1(nn.Module):
    def __init__(
            self,
            student,
            mode,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.student=student
        # self.task_loss = task_loss
        self.mode=mode
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.diffmodel = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.diffmodel, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, input, teacher_feat=None):
        #get student features
        output_logits_init, student_feat, _, _, _= self.student(input,get_ha=True)

        #make feature maps BCHW format:
        student_feat=self._reshape_BCHW(student_feat)

        
        # project student feature to the same dimension as teacher feature
        if self.use_ae:
            student_feat = self.ae.encoder(student_feat)
        else:
            student_feat = self.trans(student_feat)

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
 
        refined_feat = self.proj(refined_feat)#check this line , what proj exactly done, no use inside pipeline
        if self.use_ae:
            refined_feat = self.ae.decoder(refined_feat) #decode student feature back from AE, comment this if not using AE
        refined_feat = refined_feat.squeeze(-1).squeeze(-1) #from BCHW to BC back
        output_logits = self.student(input, context=refined_feat)
        
        
        if self.training:
            teacher_feat=self._reshape_BCHW(teacher_feat)
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None
            # train diffusion model
            ddim_loss = self.ddim_loss(teacher_feat)
            return output_logits, output_logits_init, refined_feat, teacher_feat, ddim_loss, rec_loss
            
        else:
            return output_logits

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.diffmodel(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x


class DiffKD_Network(nn.Module):
    def __init__(
            self,
            student,
            mode,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.student=student
        # self.task_loss = task_loss
        self.mode=mode
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as the teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.diffmodel = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.diffmodel, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, input, teacher_feat=None):
        #get student features
        _output_logits, student_feat, _, _, _= self.student(input,get_ha=True)

        #make feature maps BCHW format:
        student_feat=self._reshape_BCHW(student_feat)

        
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
 
        refined_feat = self.proj(refined_feat)#check this line , what proj exactly done, no use inside pipeline
        refined_feat_temp = refined_feat.squeeze(-1).squeeze(-1) #from BCHW to BC back
        output_logits = self.student(input, context=refined_feat_temp)
        
        
        if self.training:
            teacher_feat=self._reshape_BCHW(teacher_feat)
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None
            # train diffusion model
            ddim_loss = self.ddim_loss(teacher_feat)
            feature_loss = F.mse_loss(refined_feat, teacher_feat)
            return output_logits, refined_feat, teacher_feat, ddim_loss+feature_loss, rec_loss
            # return output_logits, refined_feat, teacher_feat, ddim_loss, rec_loss
            
        else:
            return output_logits


    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.diffmodel(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x
        
class DiffKD_Network_v2(nn.Module):
    def __init__(
            self,
            student,
            mode,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.student=student
        # self.task_loss = task_loss
        self.mode=mode
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        self.inv_trans = nn.Conv2d(teacher_channels, student_channels, 1)
        # diffusion model - predict noise
        self.diffmodel = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.diffmodel, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, input, teacher_feat=None):
        #get student features
        _output_logits, student_feat, _, _, _= self.student(input,get_ha=True)

        #make feature maps BCHW format:
        student_feat=self._reshape_BCHW(student_feat)

        
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
 
        refined_feat = self.proj(refined_feat)#check this line , what proj exactly done, no use inside pipeline
        refined_feat = self.inv_trans(refined_feat) #decode student feature back from AE, comment this if not using AE
        refined_feat = refined_feat.squeeze(-1).squeeze(-1) #from BCHW to BC back
        output_logits = self.student(input, context=refined_feat)
        
        
        if self.training:
            teacher_feat=self._reshape_BCHW(teacher_feat)
            # use autoencoder on teacher feature
            if self.use_ae:
                hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
                rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
                teacher_feat = hidden_t_feat.detach()
            else:
                rec_loss = None
            # train diffusion model
            ddim_loss = self.ddim_loss(teacher_feat)
            return output_logits, refined_feat, teacher_feat, ddim_loss, rec_loss
            
        else:
            return output_logits


    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.diffmodel(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss

    def _reshape_BCHW(self, x):
        """
        Reshape a 2d (B, C) or 3d (B, N, C) tensor to 4d BCHW format.
        """
        if x.dim() == 2:
            x = x.view(x.shape[0], x.shape[1], 1, 1)
        elif x.dim() == 3:
            # swin [B, N, C]
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
            x = x.transpose(-2, -1).reshape(B, C, H, W)
        return x


class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)
        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, student_feat, teacher_feat):
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        # use autoencoder on teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)
        
        # train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss


