import os
import torch
from PIL import Image
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from matplotlib import pyplot as plt
import sys
from p2p import ptp_utils
from inversion import MyInvertFixedPoint
from p2p.p2p_functions import load_im_into_format_from_path, make_controller


def run():
    image_path = "image.jpg"
    prompt = "A cat is sleeping in a window sill."
    src_blend_words = "cat"
    dst_blend_words = "dog"
    save_dir = "/tmp"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    torch.manual_seed(8888)


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
                                                           ).to("cuda")

    # load model
    with torch.no_grad():
        GUIDANCE_SCALE = 2
        NUM_DDIM_STEPS = 50

        real_image = load_im_into_format_from_path(image_path)
        real_image.save(os.path.join(save_dir, os.path.basename(image_path)))
        plt.imshow(real_image)
        plt.show()
        new_latents = pipe2(prompt=prompt, image=real_image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                            output_type="latent").images

        # RUN FP inversion on vae_latent
        latent = pipe.invert(prompt, latents=new_latents, num_inference_steps=NUM_DDIM_STEPS,
                             guidance_scale=GUIDANCE_SCALE, num_iter=20).latents
        images = pipe(prompt=[prompt], latents=latent, guidance_scale=GUIDANCE_SCALE,
                      num_inference_steps=NUM_DDIM_STEPS, output_type='pil').images
        new_vae_image = images[0]

        images_to_plot = [real_image, new_vae_image]
        ptp_utils.plot_images(images_to_plot, num_rows=1, num_cols=len(images_to_plot),
                              titles=["Real", "FP inv"])

        prompts = [prompt,
                   prompt.replace(src_blend_words, dst_blend_words)]
        cross_replace_steps = {'default_': .1, }
        self_replace_steps = 0.9
        blend_word = (((src_blend_words,), (
        dst_blend_words,)))  # for local edit. If it is not local yet - use only the source object: blend_word = ((('cat',), ("cat",))).
        eq_params = {"words": (dst_blend_words,), "values": (100,)}  # amplify attention to the word "tiger" by *2

        controller = make_controller(tokenizer, prompts, True, cross_replace_steps, self_replace_steps, blend_word, eq_params)
        images, x_t = ptp_utils.text2image_ldm_stable(pipe, prompts, controller, latent=latent,
                                                      num_inference_steps=NUM_DDIM_STEPS,
                                                      guidance_scale=GUIDANCE_SCALE)
        ptp_utils.view_images(images)
        edited_image = Image.fromarray(images[1])
        edited_image.save(save_path)


if __name__ == '__main__':
    run()
