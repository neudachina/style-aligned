import os, sys
sys.path.append('/home/jovyan/neudachina/style-aligned/')

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import mediapy
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


def svd_exp(prompts, image_name='output', seed=0):
    # просто обнуляю собственные значения у весов
    # без аттеншн шеринга, начинаю обнулять с самых больших, увеличивая окно
    
    # наверное, имеет смысл по две картинки в трех варинтах по вертикали сделать, 
    # по горизонтали -- количество обнуленных сингулярных чисел 
    
    
    base_path = './images/'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        scheduler=scheduler
    ).to("cuda")
    
    images = []
    
    fix_seed(seed)  
    orig_images = get_concat_v(pipeline(prompts, num_inference_steps=20).images)
    images.append(get_concat_v([orig_images, orig_images.copy(), orig_images.copy()]))
    
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        enable_attention_sharing=False,
        adain_queries=False,
        adain_keys=False,
        adain_values=False
    )
    
    # так что у нас по оси y просто обычные промпты
    # по оси x можно занулять первые компоненты, все больше и больше
    
    for i in range(1, 6):
        print('WEIGHTS EXP')
        current_images = []
        
        # query
        sa_args.weights_query_svd = True
        sa_args.weights_key_svd = False
        sa_args.weights_value_svd = False
        
        sa_args.weights_query_svd_start = 0
        sa_args.weights_query_svd_end = i
        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        # key
        sa_args.weights_query_svd = False
        sa_args.weights_key_svd = True
        sa_args.weights_value_svd = False
        
        sa_args.weights_key_svd_start = 0
        sa_args.weights_key_svd_end = i
        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        # value
        sa_args.weights_query_svd = False
        sa_args.weights_key_svd = False
        sa_args.weights_value_svd = True
        
        sa_args.weights_value_svd_start = 0
        sa_args.weights_value_svd_end = i
        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        images.append(get_concat_v(current_images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
    
    images = images[0]
   
    ylabels = ['query', 'query', 'key', 'key', 'value', 'value']
    xlabels = [*map(str, range(6))]
    for i in range(len(xlabels)):
        xlabels[i] = '[0-' + xlabels[i] + ')'
    xlabels[0] = 'base'

    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    # axes.set_title(prompts)
    axes.grid(False)
    axes.tick_params(axis=u'both', which=u'both', length=0)

    
    width, height = images.width, images.height

    axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    axes.set_xticklabels(xlabels)
    axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    axes.set_yticklabels(ylabels, rotation=90)

    fig.savefig(os.path.join(base_path, image_name + '.png'), bbox_inches='tight')
    plt.close()
    clear_output()


prompts = [
    "a spaceship in 80s retro wave",
    "a fox in stained glass design"
]
svd_exp(prompts, 'weights_biggest_0')


prompts = [
    "a dragon in fantasy medieval painting",
    "a peacock in psychedelic illustration",
]
svd_exp(prompts, 'weights_biggest_1')






def svd_exp(prompts, image_name='output', seed=0):
    # просто обнуляю собственные значения у весов
    # без аттеншн шеринга, обнуляю конец, двигая начало
    
    # наверное, имеет смысл по две картинки в трех варинтах по вертикали сделать, 
    # по горизонтали -- количество обнуленных сингулярных чисел 
    
    
    base_path = './images/'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        scheduler=scheduler
    ).to("cuda")
    
    images = []
    
    fix_seed(seed)  
    orig_images = get_concat_v(pipeline(prompts, num_inference_steps=20).images)
    images.append(get_concat_v([orig_images, orig_images.copy(), orig_images.copy()]))
    
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        enable_attention_sharing=False,
        adain_queries=False,
        adain_keys=False,
        adain_values=False
    )
    
    # так что у нас по оси y просто обычные промпты
    # по оси x можно занулять первые компоненты, все больше и больше
    
    for i in range(6):
        print('WEIGHTS EXP')
        current_images = []
        
        # query
        sa_args.weights_value_svd_start = None
        sa_args.weights_query_svd = True
        sa_args.weights_key_svd = False
        sa_args.weights_value_svd = False
        
        sa_args.weights_query_svd_start = i

        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        # key
        sa_args.weights_query_svd_start = None
        sa_args.weights_query_svd = False
        sa_args.weights_key_svd = True
        sa_args.weights_value_svd = False
        
        sa_args.weights_key_svd_start = i
        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        # value
        sa_args.weights_key_svd_start = None
        sa_args.weights_query_svd = False
        sa_args.weights_key_svd = False
        sa_args.weights_value_svd = True
        
        sa_args.weights_value_svd_start = i
        
        handler.register(sa_args, )
        fix_seed(seed)   
        current_images += pipeline(prompts, num_inference_steps=20, handler=handler).images
        handler.remove()
        
        
        images.append(get_concat_v(current_images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
    
    images = images[0]
   
    ylabels = ['query', 'query', 'key', 'key', 'value', 'value']
    xlabels = [*map(str, range(7))]
    for i in range(len(xlabels)):
        xlabels[i] = '[' + xlabels[i] + '-N]'
    xlabels[0] = 'base'

    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    # axes.set_title(prompts)
    axes.grid(False)
    axes.tick_params(axis=u'both', which=u'both', length=0)

    
    width, height = images.width, images.height

    axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    axes.set_xticklabels(xlabels)
    axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    axes.set_yticklabels(ylabels, rotation=90)

    fig.savefig(os.path.join(base_path, image_name + '.png'), bbox_inches='tight')
    plt.close()
    clear_output()


prompts = [
    "a spaceship in 80s retro wave",
    "a fox in stained glass design"
]
svd_exp(prompts, 'weights_smallest_0')


prompts = [
    "a dragon in fantasy medieval painting",
    "a peacock in psychedelic illustration",
]
svd_exp(prompts, 'weights_smallest_1')