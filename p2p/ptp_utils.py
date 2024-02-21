# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import sys
from typing import NamedTuple, Tuple


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, title=None):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    plt.imshow(pil_img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_images(images, num_rows, num_cols, titles=None):
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*2, num_rows*2))
    axes = axes.flatten()

    for i, (image, title) in enumerate(zip(images, titles)):
        axes[i].imshow(image)
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(title)

    plt.tight_layout()
    plt.show()


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
):
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)

    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)

    image = latent2image(model.vqvae, latents)

    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        low_resource: bool = False,
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)

    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, ):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


def plot_interpolation(images, images_titles, suptitle=None):
    fig, axes = plt.subplots(1, len(images), figsize=(2*len(images), 5))


    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.set_title(images_titles[i])
        ax.axis('off')

    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2
     of shape [B C H W]"""

    dot = torch.sum(v0 * v1 / (v0.norm(p=2, dim=[1, 2, 3], keepdim=True) * v1.norm(p=2, dim=[1, 2, 3], keepdim=True)), dim=[1, 2, 3], keepdim=True)
    # above_th = torch.where(torch.abs(dot) > DOT_THRESHOLD)
    # for i in above_th
    # if torch.abs(dot) > DOT_THRESHOLD:
    #     v2 = (1 - t) * v0 + t * v1
    # else:
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0 + s1 * v1

    return v2


def norm_like(src, ref):
    # works on batches
    # convert each sample in the src batch to have the norm of the corresponding sample in the ref batch
    src = src * ref.norm(p=2, dim=[1, 2, 3], keepdim=True) / src.norm(p=2, dim=[1, 2, 3], keepdim=True)
    return src


def match_patch_statistics(src, ref, patch_size):
    # Convert images to NumPy arrays
    src_array = np.array(src).astype(np.float32)
    ref_array = np.array(ref).astype(np.float32)

    # Calculate patch-wise means and standard deviations for the reference image
    ref_mean = uniform_filter(ref_array, size=patch_size, mode='constant', origin=-patch_size//2)
    ref_sq_mean = uniform_filter(ref_array**2, size=patch_size, mode='constant', origin=-patch_size//2)
    ref_std = np.sqrt(np.maximum(ref_sq_mean - ref_mean**2, 0))

    # Calculate patch-wise means and standard deviations for the source image
    src_mean = uniform_filter(src_array, size=patch_size, mode='constant', origin=-patch_size//2)
    src_sq_mean = uniform_filter(src_array**2, size=patch_size, mode='constant', origin=-patch_size//2)
    src_std = np.sqrt(np.maximum(src_sq_mean - src_mean**2, 0))

    # Adjust the source image patches to match the statistics of the reference image
    for c in range(src_array.shape[2]):
        src_array[:, :, c] = (src_array[:, :, c] - src_mean[:, :, c]) * (ref_std[:, :, c] / src_std[:, :, c]) + ref_mean[:, :, c]

    # Clip the resulting image to [0, 255]
    src_array = np.clip(src_array, 0, 255)

    # Convert the resulting array back to PIL Image format
    src_matched = Image.fromarray(src_array.astype(np.uint8))

    return src_matched


class Load_512:

    def __init__(self):
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5]*3, [0.5]*3)])

    def __call__(self, image_path):
        im = self.load_512(image_path)
        im = self.trans(im)
        return im

    @staticmethod
    def load_512(image_path, left=0, right=0, top=0, bottom=0):
        if type(image_path) is str:
            image = Image.open(image_path)
        else:
            image = image_path

        if image.mode == 'L':
            # for gray image
            image = image.convert('RGB')
        image = np.array(image)
        h, w, c = image.shape
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((512, 512)))
        return image


class HistogramMatching(object):
    """Histogram Matching operation class"""

    def __init__(self, channels=3, match_prop_input: float = 1.0):
        super().__init__()
        self.match_prop = float(match_prop_input)
        self.channels = range(channels)

        self.match_full = 1.0
        self.match_zero = 0.0

    def apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        result = np.copy(source)
        for channel in self.channels:
            result[:, :, channel] = \
                self._match_channel(source[:, :, channel],
                                    reference[:, :, channel])
        return result.astype(np.float32)

    def _match_channel(self, source: np.ndarray,
                       reference: np.ndarray) -> np.ndarray:
        if self.match_prop == self.match_zero:
            return source

        source_shape = source.shape
        source = source.ravel()
        reference = reference.ravel()

        # get unique pixel values (sorted),
        # indices of the unique array and counts
        _, s_indices, s_counts = np.unique(source,
                                           return_counts=True,
                                           return_inverse=True)
        r_values, r_counts = np.unique(reference, return_counts=True)

        # compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (
                source.size + sys.float_info.epsilon)
        r_quantiles = np.cumsum(r_counts).astype(float) / (
                reference.size + sys.float_info.epsilon)

        # interpolate linearly to find the pixel values in the reference
        # that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, r_quantiles, r_values)

        # pick the interpolated pixel values using the inverted source indices
        result = interp_values[s_indices]

        # apply matching proportion
        if self.match_prop < self.match_full:
            diff = source.astype(float) - result
            result = source.astype(float) - (diff * self.match_prop)

        return result.reshape(source_shape)


ChannelRange = NamedTuple('ChannelRange', [('min', float), ('max', float)])


class FeatureDistributionMatching(object):
    """Feature Distribution Matching operation class"""

    def __init__(self, channel_ranges: Tuple[ChannelRange, ...] = tuple([ChannelRange(0., 1.)] * 3), channels=3):
        super().__init__()
        self.channel_ranges = channel_ranges
        self.channels = range(channels)

    def apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:

        matching_result = self._matching(source[:, :, self.channels],
                                         reference[:, :, self.channels])

        result = np.copy(source)
        # Replace selected channels with matching result
        result[:, :, self.channels] = matching_result

        # Replace selected channels
        for channel in self.channels:
            result[:, :, channel] = np.clip(result[:, :, channel],
                                            self.channel_ranges[
                                                channel].min,
                                            self.channel_ranges[
                                                channel].max)

        return result.astype(np.float32)

    @staticmethod
    def _matching(source: np.ndarray,
                  reference: np.ndarray) -> np.ndarray:
        """ Run all transformation steps """
        # 1.) reshape to feature matrix (H*W,C)
        feature_mat_src = FeatureDistributionMatching._get_feature_matrix(
            source)
        feature_mat_ref = FeatureDistributionMatching._get_feature_matrix(
            reference)

        # 2.) center (subtract mean)
        feature_mat_src, _ = FeatureDistributionMatching._center_image(
            feature_mat_src)
        feature_mat_ref, reference_mean = \
            FeatureDistributionMatching._center_image(feature_mat_ref)

        # 3.) whitening: cov(feature_mat_src) = I
        feature_mat_src_white = FeatureDistributionMatching._whitening(
            feature_mat_src)

        # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
        feature_mat_src_transformed = \
            FeatureDistributionMatching._covariance_transformation(
                feature_mat_src_white, feature_mat_ref)

        # 5.) Add reference mean
        feature_mat_src_transformed += reference_mean

        # 6.) Reshape
        result = feature_mat_src_transformed.reshape(source.shape)

        return result

    @staticmethod
    def _get_feature_matrix(image: np.ndarray) -> np.ndarray:
        """ Reshapes an image (H, W, C) to
        a feature vector (H * W, C)
        :param image: H x W x C image
        :return feature_matrix: N x C matrix with N samples and C features
        """
        feature_matrix = np.reshape(image, (-1, image.shape[-1]))
        return feature_matrix

    @staticmethod
    def _center_image(image: np.ndarray):
        """ Centers the image by removing mean
        :returns centered image and original mean
        """
        image = np.copy(image)
        image_mean = np.mean(image, axis=0)
        image -= image_mean
        return image, image_mean

    @staticmethod
    def _whitening(feature_mat: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix so that cov(feature_map) = Identity or
        if the feature matrix is one dimensional so that var(feature_map) = 1.
        :param feature_mat: N x C matrix with N samples and C features
        :return feature_mat_white: A corresponding feature vector with an
        identity covariance matrix or variance of 1.
        """
        if feature_mat.shape[1] == 1:
            variance = np.var(feature_mat)
            feature_mat_white = feature_mat / np.sqrt(variance)
        else:
            data_cov = np.cov(feature_mat, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(data_cov)
            sqrt_s = np.diag(np.sqrt(s_vec))
            feature_mat_white = (feature_mat @ u_mat) @ np.linalg.inv(sqrt_s)
        return feature_mat_white

    @staticmethod
    def _covariance_transformation(feature_mat_white: np.ndarray,
                                   feature_mat_ref: np.ndarray) -> np.ndarray:
        """
        Transform the white (cov=Identity) feature matrix so that
        cov(feature_mat_transformed) = cov(feature_mat_ref). In the 2d case
        this becomes:
        var(feature_mat_transformed) = var(feature_mat_ref)
        :param feature_mat_white: input with identity covariance matrix
        :param feature_mat_ref: reference feature matrix
        :return: feature_mat_transformed with cov == cov(feature_mat_ref)
        """
        if feature_mat_white.shape[1] == 1:
            variance_ref = np.var(feature_mat_ref)
            feature_mat_transformed = feature_mat_white * np.sqrt(variance_ref)
        else:
            covariance_ref = np.cov(feature_mat_ref, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(covariance_ref)
            sqrt_s = np.diag(np.sqrt(s_vec))

            feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
        return feature_mat_transformed

