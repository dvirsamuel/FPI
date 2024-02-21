from abc import abstractmethod
from typing import List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_pix2pix_zero import Pix2PixInversionPipelineOutput
from easydict import EasyDict

from StableDiffusionPipelineWithDDIMInversion import StableDiffusionPipelineWithDDIMInversion


class MyInvert(StableDiffusionPipelineWithDDIMInversion):

    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                 requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

    @torch.no_grad()
    def invert(
            self,
            prompt=None,
            image=None,
            num_inference_steps: int = 50,
            guidance_scale: float = 1,
            num_images_per_prompt=1,
            prompt_embeds=None,
            return_dict=True,
    ):
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Preprocess image
        image = preprocess(image)

        # 4. Prepare latent variables
        latents = self.prepare_image_latents(image, batch_size, self.vae.dtype, device)

        # 5. Encode input prompt
        # num_images_per_prompt = num_images_per_prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t+1
                # latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample
                latents = self.scheduler_next_step(noise_pred, t, latents).prev_sample

                progress_bar.update()

        inverted_latents = latents.detach().clone()

        if not return_dict:
            return (inverted_latents, image)

        return Pix2PixInversionPipelineOutput(latents=inverted_latents, images=image)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            None,
            latents,
        )

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)[0]

                progress_bar.update()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor)[0]
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return image

        if output_type == 'pil':
            return EasyDict({'images': image, 'latents': latents})
        else:
            EasyDict({'images': image})
        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)


class MyInvertOptimized(MyInvert):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                 requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

    def invert(
            self,
            prompt=None,
            image=None,
            num_inference_steps: int = 50,
            guidance_scale: float = 1,
            num_images_per_prompt=1,
            prompt_embeds=None,
            num_iter=30,
            return_dict=True,
    ):
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # do_classifier_free_guidance = True

        # 3. Preprocess image
        image = preprocess(image)

        # 4. Prepare latent variables
        with torch.no_grad():
            latents = self.prepare_image_latents(image, batch_size, self.vae.dtype, device)

            # 5. Encode input prompt
            # num_images_per_prompt = num_images_per_prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
            )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 7. Denoising loop where we obtain the cross-attention maps.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latents = self.optimize(latents, t, prompt_embeds, num_iter, i, do_classifier_free_guidance, guidance_scale)

                progress_bar.update()

        latents = latents.detach().clone()

        return EasyDict({'latents': latents})

    @abstractmethod
    def optimize(self, latent, t, prompt_embeds, num_iter, step_idx, do_classifier_free_guidance=False, guidance_scale=1.0):
        pass


class MyInvertFixedPoint(MyInvertOptimized):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                 requires_safety_checker: bool = True):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

    def invert(
            self,
            prompt=None,
            image=None,
            latents = None,
            targets = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 1,
            num_images_per_prompt=1,
            prompt_embeds=None,
            num_iter=30,
            return_dict=True,
    ):
        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # do_classifier_free_guidance = True

        # 4. Prepare latent variables
        with torch.no_grad():
            if latents is None:
                # 3. Preprocess image
                image = preprocess(image)
                latents = self.prepare_image_latents(image, batch_size, self.vae.dtype, device)
            init_latents = latents
            # 5. Encode input prompt
            # num_images_per_prompt = num_images_per_prompt
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
            )

        # 4. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        scores, grad_norm = [], []
        # 7. Denoising loop where we obtain the cross-attention maps.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latents = self.optimize(latents, t, prompt_embeds, num_iter, i, do_classifier_free_guidance, guidance_scale)
                progress_bar.update()

        latents = latents.detach().clone()

        return EasyDict({'latents': latents, 'scores': scores, 'grad_norm': grad_norm, 'init_latents': init_latents})

    @torch.no_grad()
    def optimize(self, latent, t, prompt_embeds, num_iter, step_idx, do_classifier_free_guidance=False, guidance_scale=1.0):
        last = latent
        latent_0 = latent.clone()
        for i in range(num_iter):
            # predict the noise residual
            latent_model_input = torch.cat([latent] * 2) if do_classifier_free_guidance else latent
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latent = self.scheduler_next_step(noise_pred, t, latent_0).prev_sample

            score = torch.norm(last - latent)
            last = latent
            if i == num_iter - 1:
                print('\n**** Reached Max Iterations: ', num_iter)

        print('score: ', score.item())
        return latent