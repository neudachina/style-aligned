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
    
    

def svd_exp(prompt, image_name='output', seed=0):
    # хочу попробовать удалять разные ПЕРВЫЕ собственные числа на разных шагах генерации 
    # у РЕФЕРЕНСНОЙ картинки при attention sharing'е
    
    # base_path = './images/reference/variance/base/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda:2")
    
    # то есть я сначала хочу сгенерировать две картинки: 
    # референс и таргет просто при обычном sharing'е
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
    )
    
    sa_args.svd = True
    sa_args.svd_reference_value = True
    sa_args.svd_variance_value = True
    sa_args.svd_variance_value_biggest = True
    sa_args.svd_variance_value_smallest = False
    

    # а потом уже все менять и сохранять картинку только таргета, 
    # потому что референс не должен меняться
    base_path = './images/reference/variance/value-no-style/'
    
    def work(start, end, name):
        images, ylabels = [], []
        ylabels = []
        
        path = os.path.join(base_path, name)
        Path(path).mkdir(parents=True, exist_ok=True)
        
        xs = np.linspace(start, end, 10, endpoint=True)
        
        # хочу по горизонтали менять долю обнулённых сингулярных значений, 
        # а по вертикали -- таймстемпы, для которых применяю svd
        timestamps = [*range(0, 30, 5)]
        for i in range(len(timestamps) - 1):
            sa_args.start_svd = timestamps[i]
            sa_args.end_svd = timestamps[i+1]
            
            
            ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
            
            current_images = []
            for j in xs:
                sa_args.svd_variance_value_threshhold = j
                
                current_images.append(
                    generate(pipeline, prompt, handler=handler, 
                             args=sa_args, seed=seed)[1].resize((512, 512))
                )
            
            images.append(get_concat_h(current_images))
            images = [get_concat_v(images)]
            images[0].save(os.path.join(path, image_name + '.png'))
    
    
        sa_args.start_svd = -1
        sa_args.end_svd = 10000000
        
        current_images = []
        for j in xs:
            sa_args.svd_variance_value_threshhold = j
            
            current_images.append(
                generate(pipeline, prompt, handler=handler, 
                            args=sa_args, seed=seed)[1].resize((512, 512))
            )
            
        images.append(get_concat_h(current_images))
        images = get_concat_v(images)
        images.save(os.path.join(path, image_name + '.png'))
    
        ylabels.append('[0-25]')
        xlabels = [*map(str, np.round(xs, 2))]
        for i in range(len(xlabels)):
            xlabels[i] = '[0-' + xlabels[i] + ']'
            
        label(images, xlabels, ylabels, path, image_name, title=prompt[-1])
    
    
    work(0.1, 1, 'multiplier-011')
    # work(0.01, 0.1, 'multiplier-00101')
    

prompts = [
    "a peacock in psychedelic illustration",
    "a chameleon"
]
svd_exp(prompts, 'psychedelic')


prompts = [
    "a pine tree in watercolor sketch",
    "an oak tree"
]
svd_exp(prompts, 'watercolor')

prompts = [
    "a cat in geometric abstract art",
    "a fish"
]
svd_exp(prompts, 'abstract_0')

prompts = [
    "a cat in geometric abstract art",
    "a bird",
]
svd_exp(prompts, 'abstract_1')

prompts = [
    "a spaceship in 80s retro wave",
    "a robot"
]
svd_exp(prompts, 'retro')


prompts = [
    "a firewoman in minimal flat design illustartion",
    "a unicorn"
]
svd_exp(prompts, 'flat')


prompts = [
    "a fox in stained glass design",
    "a deer"
]
svd_exp(prompts, 'glass')


prompts = [
   "a bridge in futuristic sci-fi rendering",
   "a skyscraper"
]
svd_exp(prompts, 'sci-fi')


prompts = [
   "a beach umbrella in summer pop art",
   "a surfboard"
]
svd_exp(prompts, 'pop_art')

    
    
# prompts = [
#     "a peacock in psychedelic illustration",
#     "a chameleon in psychedelic illustration"
# ]
# svd_exp(prompts, 'psychedelic')


# prompts = [
#     "a pine tree in watercolor sketch",
#     "an oak tree in watercolor sketch"
# ]
# svd_exp(prompts, 'watercolor_0')

# prompts = [
#     "a cat in geometric abstract art",
#     "a fish in geometric abstract art"
# ]
# svd_exp(prompts, 'abstract_0')

# prompts = [
#     "a cat in geometric abstract art",
#     "a bird in geometric abstract art",
# ]
# svd_exp(prompts, 'abstract_1')

# prompts = [
#     "a spaceship in 80s retro wave",
#     "a robot in 80s retro wave"
# ]
# svd_exp(prompts, 'retro')


# prompts = [
#     "a firewoman in minimal flat design illustartion",
#     "a unicorn in minimal flat design illustartion"
# ]
# svd_exp(prompts, 'flat')


# prompts = [
#     "a fox in stained glass design",
#     "a deer in stained glass design"
# ]
# svd_exp(prompts, 'glass')


# prompts = [
#    "a bridge in futuristic sci-fi rendering",
#    "a skyscraper in futuristic sci-fi rendering"
# ]
# svd_exp(prompts, 'sci-fi')


# prompts = [
#    "a beach umbrella in summer pop art",
#    "a surfboard in summer pop art"
# ]
# svd_exp(prompts, 'pop_art')