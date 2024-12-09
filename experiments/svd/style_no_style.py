import os, sys
sys.path.append('/home/jovyan/neudachina/style-aligned/')

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import sa_handler
from IPython.display import clear_output
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_concat_h(images):
    height, width = 0, 0
    for image in images:
        width += image.width
        height = max(height, image.height)
    dst = Image.new('RGB', (width, height))
    current_width = 0
    for image in images:
        dst.paste(image, (current_width, 0))
        current_width += image.width
    return dst

def get_concat_v(images):
    height, width = 0, 0
    for image in images:
        height += image.height
        width = max(width, image.width)
    dst = Image.new('RGB', (width, height))
    current_height = 0
    for image in images:
        dst.paste(image, (0, current_height))
        current_height += image.height
    return dst

def fix_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def style_exp(style_prompts, no_style_prompts, objects, style_name, image_name='output', seed=0):
    # просто хочется посмотреть, в чем вообще разница 
    # между обычными картинками объектов и картинок со стилем
    
    base_path = './images/style-no-style/'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda")
    
    images = []
    
    # сначала просто картинки без шеринга. сос тилем и без
    fix_seed(seed)   
    images.append(get_concat_h(pipeline(style_prompts, num_inference_steps=20).images))
    fix_seed(seed)   
    images.append(get_concat_h(pipeline(no_style_prompts, num_inference_steps=20).images))
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
    )
    
    handler.register(sa_args, )
    
    fix_seed(seed)   
    images.append(get_concat_h(pipeline(style_prompts, num_inference_steps=20).images))
    
    fix_seed(seed)   
    images.append(get_concat_h(pipeline(no_style_prompts, num_inference_steps=20).images))
    
    images = get_concat_v(images)
    images.save(os.path.join(base_path, image_name + '.png'))
    
    
    ylabels = ['style no sharing', 'no style no sharing', 'style sharing', 'no style sharing']
    xlabels = objects
    
    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    axes.set_title(style_name)
    axes.grid(False)
    axes.tick_params(axis=u'both', which=u'both', length=0)

    
    width, height = images.width, images.height

    axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    axes.set_xticklabels(xlabels)
    axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    axes.set_yticklabels(ylabels, rotation=90)

    fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    plt.close()
    clear_output()
    
    
# style_prompts = [
#     "a pine tree in watercolor sketch",
#     "an oak tree in watercolor sketch",
#     "a palm tree in watercolor sketch",
#     "a bonsai in watercolor sketch",
#     "a cherry blossom in watercolor sketch"
# ]
# no_style_prompts = [
#     "a pine tree",
#     "an oak tree",
#     "a palm tree",
#     "a bonsai",
#     "a cherry blossom"
# ]
# objects = ["pine tree", "oak tree", "palm tree", "bonsai", "cherry blossom"]
# style_name = "a [...] in watercolor sketch"
# style_exp(style_prompts, no_style_prompts, objects, style_name, 'watercolor')




# style_prompts = [
#     "a fox in stained glass design",
#     "a deer in stained glass design",
#     "an owl in stained glass design",
#     "a wolf in stained glass design",
#     "a bear in stained glass design"
# ]
# no_style_prompts = [
#     "a fox",
#     "a deer",
#     "an owl",
#     "a wolf",
#     "a bear"
# ]
# objects = ["fox", "deer", "owl", "wolf", "bear"]
# style_name = "a [...] in stained glass design"
# style_exp(style_prompts, no_style_prompts, objects, style_name, 'stained_glass')




# style_prompts = [
#     "a cat in geometric abstract art",
#     "a lion in geometric abstract art",
#     "an elephant in geometric abstract art",
#     "a bird in geometric abstract art",
#     "a fish in geometric abstract art"
# ]
# no_style_prompts = [
#     "a cat",
#     "a lion",
#     "an elephant",
#     "a bird",
#     "a fish"
# ]
# objects = ["cat", "lion", "elephant", "bird", "fish"]
# style_name = "a [...] in geometric abstract art"
# style_exp(style_prompts, no_style_prompts, objects, style_name, 'geometric_abstract')


style_prompts = [
    "a beach umbrella in summer pop art",
    "a surfboard in summer pop art",
    "a beach ball in summer pop art",
    "a sandcastle in summer pop art",
    "a sun lounger in summer pop art"
]
no_style_prompts = [
    "a beach umbrella",
    "a surfboard",
    "a beach ball",
    "a sandcastle",
    "a sun lounger"
]
objects = ["beach umbrella", "surfboard", "beach ball", "sandcastle", "sun lounger"]
style_name = "a [...] in summer pop art"
style_exp(style_prompts, no_style_prompts, objects, style_name, 'pop_art')



# у женщины нога куда-то ушла, что с мужчиной случилось???
style_prompts = [
    "a dancer in art deco style",
    "a waiter in art deco style",
    "an architect in art deco style",
    "a singer in art deco style",
    "a jazz musician in art deco style"
]
style_prompts = [
    "a dancer",
    "a waiter",
    "an architect",
    "a singer",
    "a jazz musician"
]
objects = ["dancer", "waiter", "architect", "singer", "jazz musician"]
style_name = "a [...] in art deco style"
style_exp(style_prompts, no_style_prompts, objects, style_name, 'art_deco')