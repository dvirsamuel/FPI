import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from matplotlib import pyplot as plt
from p2p import ptp_utils
from inversion import MyInvertFixedPoint
from p2p.p2p_functions import latent2image, make_controller


def fake_image(pipe, prompt, num_ddim_steps = 50, guidance_scale=6):
    init_seed = torch.randn((1, 4, 64, 64), device=pipe.device).to("cuda") * pipe.scheduler.init_noise_sigma
    out = pipe(prompt=[prompt], latents=init_seed, guidance_scale=guidance_scale,
               num_inference_steps=num_ddim_steps, output_type='pil')
    images, z_0_ = out.images, out.latents
    return images[0]


def run():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = MyInvertFixedPoint.from_pretrained(
        model_id,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
        safety_checker=None,
    ).to('cuda')

    tokenizer = pipe.tokenizer

    pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                           scheduler=DDIMScheduler.from_pretrained(model_id,
                                                                                                   subfolder="scheduler"),
                                                           cache_dir="/inputs/huggingface_cache/", ).to("cuda")

    prompt = "A painting of a squirrel eating a burger"
    torch.manual_seed(8888)

    # load model
    with torch.no_grad():
        GUIDANCE_SCALE = 6
        NUM_DDIM_STEPS = 50

        real_image = fake_image(pipe, prompt, NUM_DDIM_STEPS, GUIDANCE_SCALE)
        plt.imshow(real_image)
        plt.show()
        new_latents = pipe2(prompt=prompt, image=real_image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                            output_type="latent").images
        noised_image = latent2image(pipe.vae, new_latents)

        # RUN FP inversion on vae_latent
        latent = pipe.invert(prompt, latents=new_latents, num_inference_steps=NUM_DDIM_STEPS,
                             guidance_scale=GUIDANCE_SCALE, num_iter=20).latents
        images = pipe(prompt=[prompt], latents=latent, guidance_scale=GUIDANCE_SCALE,
                      num_inference_steps=NUM_DDIM_STEPS, output_type='pil').images
        new_vae_image = images[0]

        images_to_plot = [real_image, noised_image, new_vae_image]
        ptp_utils.plot_images(images_to_plot, num_rows=1, num_cols=len(images_to_plot),
                              titles=["Real", "Real Denoised", "FP inv"])

        prompts = [prompt,
                   prompt.replace("squirrel", "giraffe")]
        cross_replace_steps = {'default_': .2, }
        self_replace_steps = .6
        blend_word = ((('squirrel',), ("giraffe",)))  # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
        eq_params = {"words": ("giraffe",), "values": (6,)}  # amplify attention to the word "tiger" by *2

        controller = make_controller(tokenizer, prompts, True, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        images, x_t = ptp_utils.text2image_ldm_stable(pipe, prompts, controller, latent=latent,
                                                      num_inference_steps=NUM_DDIM_STEPS,
                                                      guidance_scale=GUIDANCE_SCALE)
        ptp_utils.view_images(images)

if __name__ == "__main__":
    run()


