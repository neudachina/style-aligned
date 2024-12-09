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



def generate_basic(prompt, image_name='output', seed=0):
    base_path = './images/more-reference/'
    Path(os.path.join(base_path, 'no-sharing')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(base_path, 'no-svd')).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda")
    
    # сначала генерирую референс и таргет без шеринга,
    # а потом с шерингом, но без свд
    
    get_concat_v(generate(pipeline, prompt, seed=seed)).save(
        os.path.join(base_path, 'no-sharing', image_name + '.png')
    )
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
    )
    
    get_concat_v(generate(pipeline, prompt, handler, sa_args, seed=seed)).save(
        os.path.join(base_path, 'no-svd', image_name + '.png')
    )


def svd_exp(prompt, image_name='output', seed=0):
    # хочу попробовать уменьшать значения разных ПОСЛЕДНИХ собственных чисел 
    # на первых шагах генерации у РЕФЕРЕНСНОЙ картинки при attention sharing'е 
    
    # по горизонтали -- количество компонент, 
    # по вертикали -- коэффициент
    
    # картинки с 0-5 отдельно, с 5-10 отдельно
    
    base_path = './images/more-reference/'
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
        adain_values=False,
        svd=True,
        svd_reference_value=True,
        svd_reference_value_positive=True,
        svd_reference_value_negative=False
    )
    
    # хочу по горизонтали менять количество обнулённых сингулярных значений, 
    # а по вертикали -- коэффициент, на который умножаю
    def work(start, end, path):
        images = []        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        sa_args.start_svd = start
        sa_args.end_svd = end
        
        xs, ys = range(6), np.linspace(0, 0.5, 6)
        
        for i in xs:
            sa_args.svd_reference_value_start = i
            
            current_images = []
            for j in ys:
                sa_args.svd_reference_value_coef = j
                current_images.append(generate(pipeline, prompt, handler=handler, args=sa_args, seed=seed)[1])
                # сохраняю картинку только таргета, потому что референс не должен меняться
            
            images.append(get_concat_v(current_images))   
            images = [get_concat_h(images)]
            images[0].save(os.path.join(path, image_name + '.png'))
            
        images = images[0]
        xlabels = [*map(str, xs)]
        for i in range(len(xlabels)):
            xlabels[i] = '[' + xlabels[i] + '-N]'
        ylabels = [*map(str, np.round(ys, 2))]
        
        label(images, xlabels, ylabels, path, image_name, title=prompt[-1])
        
    work(0, 5, os.path.join(base_path, '0-5-smallest-coef'))
    work(5, 10, os.path.join(base_path, '5-10-smallest-coef'))

    


# слева направо и сверху вниз 
# у нас должно увеличиваться количество контент лика

prompt = [
    "a peacock in psychedelic illustration",
    "a hummingbird in psychedelic illustration"
]
# generate_basic(prompt, 'psychedelic')
svd_exp(prompt, 'psychedelic')

prompt = [
    "a pine tree in watercolor sketch",
    "an oak tree in watercolor sketch"
]
generate_basic(prompt, 'watercolor')
svd_exp(prompt, 'watercolor')

prompt = [
    "a spaceship in 80s retro wave",
    "a robot in 80s retro wave"
]
generate_basic(prompt, 'retro_0')
svd_exp(prompt, 'retro_0')

prompt = [
    "a spaceship in 80s retro wave",
    "a time machine in 80s retro wave"
]
generate_basic(prompt, 'retro_1')
svd_exp(prompt, 'retro_1')

prompt = [
    "a firewoman in minimal flat design illustration",
    "a unicorn in minimal flat design illustration"
]
generate_basic(prompt, 'flat')
svd_exp(prompt, 'flat')


prompt = [
    "a cat in geometric abstract art",
    "a fish in geometric abstract art"
]
generate_basic(prompt, 'abstract_0')
svd_exp(prompt, 'abstract_0')

prompt = [
    "a cat in geometric abstract art",
    "a fish in geometric abstract art"
]
generate_basic(prompt, 'abstract_1')
svd_exp(prompt, 'abstract_1')


prompt = [
    "a beach umbrella in summer pop art",
    "a surfboard in summer pop art"
]
generate_basic(prompt, 'pop_art_0')
svd_exp(prompt, 'pop_art_0')

prompt = [
    "a beach umbrella in summer pop art",
    "a sandcastle in summer pop art"
]
generate_basic(prompt, 'pop_art_1')
svd_exp(prompt, 'pop_art_1')

