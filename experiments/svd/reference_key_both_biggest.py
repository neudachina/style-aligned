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
    
    
    
def label(image, xlabels, ylabels, base_path, image_name, title=None):
    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 30

    fig, axes = plt.subplots()

    axes.imshow(image)
    if title is not None:
        axes.set_title(title)
    axes.grid(False)
    axes.tick_params(axis=u'both', which=u'both', length=0)

    
    width, height = image.width, image.height

    axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    axes.set_xticklabels(xlabels)
    axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    axes.set_yticklabels(ylabels, rotation=90)

    fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    plt.close()
    clear_output()    
    
    
    
def generate(pipeline, prompts, handler=None, args=None, seed=0, num_steps=20):
    if args and handler:
        handler.register(args)
    
    fix_seed(seed)   
    images = pipeline(prompts, num_inference_steps=num_steps, handler=handler).images
    
    if args and handler:
        handler.remove()
    return images



def svd_exp(prompt, style_name, objects, image_name='output', seed=0):
    # обнуляю первые компоненты у референса для key на [5-10] шагах
    # генерирую сначала с обычным шерингом, потом с обнулением, в конце без шеринга для сравнения
    
    base_path = './images/more-reference/5-10-key-biggest'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda")
    
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        adain_queries=False,
        adain_keys=False,
        adain_values=False
    )
    
    images = []
    # сначала обычный шеринг
    images.append(
            get_concat_v(
                generate(pipeline, prompt, handler=handler, args=sa_args, seed=seed)
            ).resize((512, 512 * len(prompt))))
    
    sa_args.svd = True
    sa_args.svd_reference_key = True
    sa_args.svd_reference_key_positive = True
    sa_args.svd_reference_key_negative = True
    sa_args.start_svd = 5
    sa_args.end_svd = 10
    sa_args.svd_reference_key_start = 0
    
    # хочу по горизонтали менять количество обнулённых сингулярных значений,
    # а по вертикали просто разные промпты
    xs = range(1, 10)
        
    for i in xs:
        sa_args.svd_reference_key_end = i
        
        images.append(
            get_concat_v(
                generate(pipeline, prompt, handler=handler, args=sa_args, seed=seed)
            ).resize((512, 512 * len(prompt))))
        
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
        
    images.append(
            get_concat_v(
                generate(pipeline, prompt, seed=seed)
            ).resize((512, 512 * len(prompt))))
    images = get_concat_h(images)
    images.save(os.path.join(base_path, image_name + '.png'))
    
    xlabels = [*map(str, xs)]
    for i in range(len(xlabels)):
        xlabels[i] = '[0-' + xlabels[i] + ')'
    xlabels = ['base sharing'] + xlabels + ['no sharing']
    
    label(images, xlabels, objects, base_path, image_name, title=style_name)



prompts = [
    "a cat in geometric abstract art",
    "a lion in geometric abstract art",
    "an elephant in geometric abstract art",
    "a bird in geometric abstract art",
    "a fish in geometric abstract art"
]
objects = ["cat", "lion", "elephant", "bird", "fish"]
style = "a [...] in geometric abstract art"
svd_exp(prompts, style, objects, 'geometric')



prompts = [
    "a sci-fi robot warrior. comic book illustration. cyberpunk theme",
    "a sci-fi spaceship. comic book illustration. cyberpunk theme",
    "a sci-fi cityscape. comic book illustration. cyberpunk theme",
    "a sci-fi alien creature. comic book illustration. cyberpunk theme",
    "a sci-fi futuristic car. comic book illustration. cyberpunk theme"
]
style = "a sci-fi [...]. comic book illustration. cyberpunk theme"
objects = ["robot warrior", "spaceship", "cityscape", "alien creature", "futuristic car"]
svd_exp(prompts, style, objects, 'sci_fi')


prompts = [
    "a firewoman in minimal flat design illustration",
    "a farmer in minimal flat design illustration",
    "a unicorn in minimal flat design illustration",
    "a dino in minimal flat design illustration",
    "a dog in minimal flat design illustration"
]
style = "a [...] in minimal flat design illustration"
objects = ["firewoman", "farmer", "unicorn", "dino", "dog"]
svd_exp(prompts, style, objects, 'flat')


prompts = [
    "a beach umbrella in summer pop art",
    "a surfboard in summer pop art",
    "a beach ball in summer pop art",
    "a sandcastle in summer pop art",
    "a sun lounger in summer pop art"
]
objects = ["beach umbrella", "surfboard", "beach ball", "sandcastle", "sun lounger"]
style = "a [...] in summer pop art"
svd_exp(prompts, style, objects, 'pop_art')


prompts = [
    "a peacock in psychedelic illustration",
    "a hummingbird in psychedelic illustration",
    "a butterfly in psychedelic illustration",
    "a chameleon in psychedelic illustration",
    "a parrot in psychedelic illustration"
]
objects = ["peacock", "hummingbird", "butterfly", "chameleon", "parrot"]
style = "a [...] in psychedelic illustration"
svd_exp(prompts, style, objects, 'psychedelic')


prompts = [
    "a spaceship in 80s retro wave",
    "a robot in 80s retro wave",
    "a laser gun in 80s retro wave",
    "a flying saucer in 80s retro wave",
    "a time machine in 80s retro wave"
]
objects = ["spaceship", "robot", "laser gun", "flying saucer", "time machine"]
style = "a [...] in 80s retro wave"
svd_exp(prompts, style, objects, 'retro_wave')
