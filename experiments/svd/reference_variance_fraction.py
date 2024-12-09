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
    
    
    
    
    

def svd_exp(prompt, image_name='output', seed=0):
    # хочу попробовать удалять разные ПЕРВЫЕ собственные числа на разных шагах генерации 
    # у РЕФЕРЕНСНОЙ картинки при attention sharing'е
    
    # base_path = './images/reference/variance/base/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)
     
    # scheduler = DDIMScheduler(
    #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
    #     clip_sample=False, set_alpha_to_one=False)
    
    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-base-1.0", 
    #     scheduler=scheduler
    # ).to("cuda:1")
    
    # # то есть я сначала хочу сгенерировать две картинки: 
    # # референс и таргет просто при обычном sharing'е
    # handler = sa_handler.Handler(pipeline)
    # sa_args = sa_handler.StyleAlignedArgs(
    #     share_group_norm=False,
    #     share_layer_norm=False,
    #     share_attention=True,
    #     adain_queries=False,
    #     adain_keys=False,
    #     adain_values=False,
    # )
    
    
    # handler.register(sa_args, )
    # fix_seed(seed)   
    # get_concat_v(pipeline(prompt, num_inference_steps=20).images).save(
    #     os.path.join(base_path, image_name + '_no_svd.png')
    # )
    # handler.remove()
    
    # # а потом уже все менять и сохранять картинку только таргета, 
    # # потому что референс не должен меняться
    
    # base_path = './images/reference/variance/ts-key-biggest/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # images = []
    # ylabels = []
    
    # sa_args.svd = True
    # sa_args.svd_reference_key = True
    # sa_args.svd_variance_key = True
    # sa_args.svd_variance_key_biggest = True
    # sa_args.svd_variance_key_smallest = False
    
    # # хочу по горизонтали менять долю обнулённых сингулярных значений, 
    # # а по вертикали -- таймстемпы, для которых применяю svd
    # timestamps = [*range(0, 30, 5)]
    # for i in range(len(timestamps) - 1):
    #     sa_args.start_svd = timestamps[i]
    #     sa_args.end_svd = timestamps[i+1]
        
        
    #     ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
    #     current_images = []
    #     for j in np.linspace(0.1, 1, 10, endpoint=True):
    #         sa_args.svd_variance_key_threshhold = j
            
    #         handler.register(sa_args, )
            
    #         fix_seed(seed)   
    #         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #         handler.remove()
        
    #     images.append(get_concat_h(current_images))   
    #     images = [get_concat_v(images)]
    #     images[0].save(os.path.join(base_path, image_name + '.png'))
        

    # current_images = []
    
    # sa_args.start_svd = -1
    # sa_args.end_svd = 10000000
    # for j in np.linspace(0.1, 1, 10, endpoint=True):
    #     sa_args.svd_variance_key_threshhold = j
            
    #     handler.register(sa_args, )
        
    #     fix_seed(seed)   
    #     current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #     handler.remove()
    
    # images.append(get_concat_h(current_images))   
    # images = [get_concat_v(images)]
    # images[0].save(os.path.join(base_path, image_name + '.png'))
    
    # ylabels.append('[0-25]')
    
    # images = images[0]
    
    # xlabels = [*map(str, np.round(np.linspace(0.1, 1, 10, endpoint=True), 2))]

    # plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    # plt.rcParams["font.size"] = 20

    # fig, axes = plt.subplots()

    # axes.imshow(images)
    # axes.set_title(prompt[-1])
    # axes.grid(False)
    # axes.tick_params(axis=u'both', which=u'both', length=0)

    
    # width, height = images.width, images.height

    # axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    # axes.set_xticklabels(xlabels)
    # axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    # axes.set_yticklabels(ylabels, rotation=90)

    # fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    # plt.close()
    # clear_output()
    
    
    
    
    
    # base_path = './images/reference/variance/ts-key-biggest-01/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # images = []
    # ylabels = []
    
    # timestamps = [*range(0, 30, 5)]
    # for i in range(len(timestamps) - 1):
    #     sa_args.start_svd = timestamps[i]
    #     sa_args.end_svd = timestamps[i+1]
        
        
    #     ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
    #     current_images = []
    #     for j in np.linspace(0.01, 0.1, 10, endpoint=True):
    #         sa_args.svd_variance_key_threshhold = j
            
    #         handler.register(sa_args, )
            
    #         fix_seed(seed)   
    #         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #         handler.remove()
        
    #     images.append(get_concat_h(current_images))   
    #     images = [get_concat_v(images)]
    #     images[0].save(os.path.join(base_path, image_name + '.png'))
        

    # current_images = []
    
    # sa_args.start_svd = -1
    # sa_args.end_svd = 10000000
    # for j in np.linspace(0.01, 0.1, 10, endpoint=True):
    #     sa_args.svd_variance_key_threshhold = j
            
    #     handler.register(sa_args, )
        
    #     fix_seed(seed)   
    #     current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #     handler.remove()
    
    # images.append(get_concat_h(current_images))   
    # images = [get_concat_v(images)]
    # images[0].save(os.path.join(base_path, image_name + '.png'))
    
    # ylabels.append('[0-25]')
    
    # images = images[0]
    
    # xlabels = [*map(str, np.round(np.linspace(0.01, 0.1, 10, endpoint=True), 3))]

    # plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    # plt.rcParams["font.size"] = 20

    # fig, axes = plt.subplots()

    # axes.imshow(images)
    # axes.set_title(prompt[-1])
    # axes.grid(False)
    # axes.tick_params(axis=u'both', which=u'both', length=0)

    
    # width, height = images.width, images.height

    # axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    # axes.set_xticklabels(xlabels)
    # axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    # axes.set_yticklabels(ylabels, rotation=90)

    # fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    # plt.close()
    # clear_output()
    
    
    
    
    
    
    # base_path = './images/reference/variance/ts-key-smallest/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # images = []
    # ylabels = []
    
    # sa_args.svd_variance_key_biggest = False
    # sa_args.svd_variance_key_smallest = True
    
    # # хочу по горизонтали менять долю обнулённых сингулярных значений, 
    # # а по вертикали -- таймстемпы, для которых применяю svd
    # timestamps = [*range(0, 30, 5)]
    # for i in range(len(timestamps) - 1):
    #     sa_args.start_svd = timestamps[i]
    #     sa_args.end_svd = timestamps[i+1]
        
        
    #     ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
    #     current_images = []
    #     for j in np.linspace(1, 0.1, 10, endpoint=True):
    #         sa_args.svd_variance_key_threshhold = j
            
    #         handler.register(sa_args, )
            
    #         fix_seed(seed)   
    #         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #         handler.remove()
        
    #     images.append(get_concat_h(current_images))   
    #     images = [get_concat_v(images)]
    #     images[0].save(os.path.join(base_path, image_name + '.png'))
        

    # current_images = []
    
    # sa_args.start_svd = -1
    # sa_args.end_svd = 10000000
    # for j in np.linspace(1, 0.1, 10, endpoint=True):
    #     sa_args.svd_variance_key_threshhold = j
            
    #     handler.register(sa_args, )
        
    #     fix_seed(seed)   
    #     current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #     handler.remove()
    
    # images.append(get_concat_h(current_images))   
    # images = [get_concat_v(images)]
    # images[0].save(os.path.join(base_path, image_name + '.png'))
    
    # ylabels.append('[0-25]')
    
    # images = images[0]
    
    # xlabels = [*map(str, np.round(np.linspace(1, 0.1, 10, endpoint=True), 2))]

    # plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    # plt.rcParams["font.size"] = 20

    # fig, axes = plt.subplots()

    # axes.imshow(images)
    # axes.set_title(prompt[-1])
    # axes.grid(False)
    # axes.tick_params(axis=u'both', which=u'both', length=0)

    
    # width, height = images.width, images.height

    # axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    # axes.set_xticklabels(xlabels)
    # axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    # axes.set_yticklabels(ylabels, rotation=90)

    # fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    # plt.close()
    # clear_output()
    
    
    
    
    
    
    # base_path = './images/reference/variance/ts-key-smallest-01/'
    # Path(base_path).mkdir(parents=True, exist_ok=True)

    # images = []
    # ylabels = []
    
    # # хочу по горизонтали менять долю обнулённых сингулярных значений, 
    # # а по вертикали -- таймстемпы, для которых применяю svd
    # timestamps = [*range(0, 30, 5)]
    # for i in range(len(timestamps) - 1):
    #     sa_args.start_svd = timestamps[i]
    #     sa_args.end_svd = timestamps[i+1]
        
        
    #     ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
    #     current_images = []
    #     for j in np.linspace(0.1, 0.01, 10, endpoint=True):
    #         sa_args.svd_variance_key_threshhold = j
            
    #         handler.register(sa_args, )
            
    #         fix_seed(seed)   
    #         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #         handler.remove()
        
    #     images.append(get_concat_h(current_images))   
    #     images = [get_concat_v(images)]
    #     images[0].save(os.path.join(base_path, image_name + '.png'))
        

    # current_images = []
    
    # sa_args.start_svd = -1
    # sa_args.end_svd = 10000000
    # for j in np.linspace(0.1, 0.01, 10, endpoint=True):
    #     sa_args.svd_variance_key_threshhold = j
            
    #     handler.register(sa_args, )
        
    #     fix_seed(seed)   
    #     current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #     handler.remove()
    
    # images.append(get_concat_h(current_images))   
    # images = [get_concat_v(images)]
    # images[0].save(os.path.join(base_path, image_name + '.png'))
    
    # ylabels.append('[0-25]')
    
    # images = images[0]
    
    # xlabels = [*map(str, np.round(np.linspace(0.1, 0.01, 10, endpoint=True), 3))]

    # plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    # plt.rcParams["font.size"] = 20

    # fig, axes = plt.subplots()

    # axes.imshow(images)
    # axes.set_title(prompt[-1])
    # axes.grid(False)
    # axes.tick_params(axis=u'both', which=u'both', length=0)

    
    # width, height = images.width, images.height

    # axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    # axes.set_xticklabels(xlabels)
    # axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    # axes.set_yticklabels(ylabels, rotation=90)

    # fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    # plt.close()
    # clear_output()
    
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda")
    
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
    
    
    base_path = './images/reference/variance/expexp/'
    Path(base_path).mkdir(parents=True, exist_ok=True)

    images = []
    ylabels = []
    
    # хочу по горизонтали менять долю обнулённых сингулярных значений, 
    # а по вертикали -- таймстемпы, для которых применяю svd
    # timestamps = [*range(0, 30, 5)]
    # for i in range(len(timestamps) - 1):
    #     sa_args.start_svd = timestamps[i]
    #     sa_args.end_svd = timestamps[i+1]
        
        
    #     ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
    #     current_images = []
    #     for j in np.linspace(0.1, 0.01, 10, endpoint=True):
    #         sa_args.svd_variance_key_threshhold = j
            
    #         handler.register(sa_args, )
            
    #         fix_seed(seed)   
    #         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #         handler.remove()
        
    #     images.append(get_concat_h(current_images))   
    #     images = [get_concat_v(images)]
    #     images[0].save(os.path.join(base_path, image_name + '.png'))
    
    current_images = []
    
    sa_args.svd = True
    sa_args.svd_reference_value = True
    sa_args.svd_variance_value = True
    sa_args.svd_variance_value_biggest = True
    sa_args.svd_variance_value_smallest = False
    
    for j in np.linspace(0.1, 0.01, 10, endpoint=True):
        sa_args.svd_variance_value_threshhold = j
        
        handler.register(sa_args, )
        
        fix_seed(seed)   
        current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
        handler.remove()
        
        current_images = [get_concat_h(current_images)]
        current_images[0].save(os.path.join(base_path, image_name + '.png'))
        

    # current_images = []
    
    # sa_args.start_svd = -1
    # sa_args.end_svd = 10000000
    # for j in np.linspace(0.1, 0.01, 10, endpoint=True):
    #     sa_args.svd_variance_key_threshhold = j
            
    #     handler.register(sa_args, )
        
    #     fix_seed(seed)   
    #     current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[1])
    #     handler.remove()
    
    # images.append(get_concat_h(current_images))   
    # images = [get_concat_v(images)]
    # images[0].save(os.path.join(base_path, image_name + '.png'))
    
    # ylabels.append('[0-25]')
    
    # images = images[0]
    
    # xlabels = [*map(str, np.round(np.linspace(0.1, 0.01, 10, endpoint=True), 3))]

    # plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    # plt.rcParams["font.size"] = 20

    # fig, axes = plt.subplots()

    # axes.imshow(images)
    # axes.set_title(prompt[-1])
    # axes.grid(False)
    # axes.tick_params(axis=u'both', which=u'both', length=0)

    
    # width, height = images.width, images.height

    # axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
    # axes.set_xticklabels(xlabels)
    # axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
    # axes.set_yticklabels(ylabels, rotation=90)

    # fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    # plt.close()
    # clear_output()
    
    
    
    
    
prompts = [
    "a peacock in psychedelic illustration",
    "a chameleon in psychedelic illustration"
]
svd_exp(prompts, 'psychedelic')


prompts = [
    "a pine tree in watercolor sketch",
    "an oak tree in watercolor sketch"
]
svd_exp(prompts, 'watercolor_0')


prompts = [
    "a spaceship in 80s retro wave",
    "a robot in 80s retro wave"
]
svd_exp(prompts, 'retro')


prompts = [
    "a firewoman in minimal flat design illustartion",
    "a unicorn in minimal flat design illustartion"
]
svd_exp(prompts, 'flat')


prompts = [
    "a fox in stained glass design",
    "a deer in stained glass design"
]
svd_exp(prompts, 'glass')


prompts = [
   "a bridge in futuristic sci-fi rendering",
   "a skyscraper in futuristic sci-fi rendering"
]
svd_exp(prompts, 'sci-fi')


prompts = [
   "a beach umbrella in summer pop art",
   "a surfboard in summer pop art"
]
svd_exp(prompts, 'pop_art')