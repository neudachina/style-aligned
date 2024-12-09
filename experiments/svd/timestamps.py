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
    
    base_path = './images/ts-biggest/'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler
    ).to("cuda")
    
    fix_seed(seed)   
    pipeline(prompt, num_inference_steps=20).images[0].save(
        os.path.join(base_path, image_name + '_no_svd.png')
    )
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        enable_attention_sharing=False,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
        svd=True,
        svd_null_start = 0
    )
    
    
    images = []
    ylabels = []
    
    # хочу по горизонтали менять количество обнулённых сингулярных значений, 
    # а по вертикали -- таймстемпы, для которых применяю svd
    timestamps = [*range(0, 30, 5)]
    for i in range(len(timestamps) - 1):
        sa_args.start_svd = timestamps[i]
        sa_args.end_svd = timestamps[i+1]
        
        print('TIMESTAMPS EXP')
        ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
        current_images = []
        for j in range(1, 6):
            sa_args.svd_null_end = j
            
            handler.register(sa_args, )
            
            fix_seed(seed)   
            current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
            handler.remove()
        
        images.append(get_concat_h(current_images))   
        images = [get_concat_v(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        

    current_images = []
    
    sa_args.start_svd = -1
    sa_args.end_svd = 10000000
    for j in range(1, 6):
        sa_args.svd_null_end = j
            
        handler.register(sa_args, )
        
        fix_seed(seed)   
        current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
        handler.remove()
    
    images.append(get_concat_h(current_images))   
    images = [get_concat_v(images)]
    images[0].save(os.path.join(base_path, image_name + '.png'))
    
    ylabels.append('[0-25]')
    
    images = images[0]
    
    xlabels = [*map(str, range(1, 6))]
    for i in range(len(xlabels)):
        xlabels[i] = '[0-' + xlabels[i] + ')'

    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    axes.set_title(prompt)
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


# по идее у меня слева направо должно становится все меньше информации
# при этом как будто сверху вниз количество информации как будто должно увеличиваться, 
# потому что меняем что-то на более поздних таймстемпах

prompt = "a peacock in psychedelic illustration"
svd_exp(prompt, 'psychedelic')

prompt = "a pine tree in watercolor sketch"
svd_exp(prompt, 'watercolor_0')

prompt = "a pine tree in watercolor sketch"
svd_exp(prompt, 'watercolor_1', seed=69)

prompt = "a spaceship in 80s retro wave"
svd_exp(prompt, 'retro')

prompt = "a firewoman in minimal flat design illustration"
svd_exp(prompt, 'flat')






# def svd_exp(prompt, image_name='output', seed=0):
#     # хочу попробовать удалять разные ПОСЛЕДНИЕ собственные числа на разных шагах генерации
    
#     base_path = './images/ts-smallest/'
#     Path(base_path).mkdir(parents=True, exist_ok=True)
     
#     scheduler = DDIMScheduler(
#         beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
#         clip_sample=False, set_alpha_to_one=False)
    
#     pipeline = StableDiffusionXLPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0", 
#         scheduler=scheduler
#     ).to("cuda")
    
#     fix_seed(seed)   
#     pipeline(prompt, num_inference_steps=20).images[0].save(
#         os.path.join(base_path, image_name + '_no_svd.png')
#     )
    
#     handler = sa_handler.Handler(pipeline)
#     sa_args = sa_handler.StyleAlignedArgs(
#         share_group_norm=False,
#         share_layer_norm=False,
#         share_attention=True,
#         enable_attention_sharing=False,
#         adain_queries=False,
#         adain_keys=False,
#         adain_values=False,
#         svd=True
#     )
    
    
#     images = []
    
#     ylabels = []
    
#     timestamps = [*range(0, 30, 5)]
#     for i in range(len(timestamps) - 1):
#         print('TIMESTAMPS EXP')
#         sa_args.start_svd = timestamps[i]
#         sa_args.end_svd = timestamps[i+1]
        
#         ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
#         current_images = []
#         for j in range(6):
#             sa_args.svd_null_start = j
            
#             handler.register(sa_args, )
            
#             fix_seed(seed)   
#             current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
#             handler.remove()
        
#         images.append(get_concat_h(current_images))
#         images = [get_concat_v(images)]
#         images[0].save(os.path.join(base_path, image_name + '.png'))
        
    
#     current_images = []
    
#     sa_args.start_svd = -1
#     sa_args.end_svd = 10000000
#     for j in range(6):
#         sa_args.svd_null_start = j
        
#         handler.register(sa_args, )
        
#         fix_seed(seed)   
#         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
#         handler.remove()
    
#     images.append(get_concat_h(current_images))
#     images = [get_concat_v(images)]
#     images[0].save(os.path.join(base_path, image_name + '.png'))
    
#     ylabels.append('[0-25]')
    
    
    
#     images = images[0]
    
    
#     xlabels = [*map(str, range(6))]
#     for i in range(len(xlabels)):
#         xlabels[i] = '[' + xlabels[i] + '-N]'

#     plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
#     plt.rcParams["font.size"] = 20

#     fig, axes = plt.subplots()

#     axes.imshow(images)
#     axes.set_title(prompt)
#     axes.grid(False)
#     axes.tick_params(axis=u'both', which=u'both', length=0)

    
#     width, height = images.width, images.height

#     axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
#     axes.set_xticklabels(xlabels)
#     axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
#     axes.set_yticklabels(ylabels, rotation=90)

#     fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
#     plt.close()
#     clear_output()



# # по идее у меня слева направо должно становится все больше информации, 
# # потому что мы оставляем все больше самых главных компонент

# # при этом как будто сверху вниз количество информации как будто должно увеличиваться, 
# # потому что меняем что-то на более поздних таймстемпах

# prompt = "a peacock in psychedelic illustration"
# svd_exp(prompt, 'psychedelic')

# prompt = "a pine tree in watercolor sketch"
# svd_exp(prompt, 'watercolor_0')

# prompt = "a pine tree in watercolor sketch"
# svd_exp(prompt, 'watercolor_1', seed=69)

# prompt = "a spaceship in 80s retro wave"
# svd_exp(prompt, 'retro')

# prompt = "a firewoman in minimal flat design illustration"
# svd_exp(prompt, 'flat')





# def svd_exp(prompt, image_name='output', seed=0):
#     # хочу попробовать только одно собственное число на разных шагах генерации
    
#     base_path = './images/ts-remain-one'
#     Path(base_path).mkdir(parents=True, exist_ok=True)
     
#     scheduler = DDIMScheduler(
#         beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
#         clip_sample=False, set_alpha_to_one=False)
    
#     pipeline = StableDiffusionXLPipeline.from_pretrained(
#         "stabilityai/stable-diffusion-xl-base-1.0",
#         scheduler=scheduler
#     ).to("cuda")
    
#     fix_seed(seed)   
#     pipeline(prompt, num_inference_steps=20).images[0].save(
#         os.path.join(base_path, image_name + '_no_svd.png')
#     )
    
#     handler = sa_handler.Handler(pipeline)
#     sa_args = sa_handler.StyleAlignedArgs(
#         share_group_norm=False,
#         share_layer_norm=False,
#         share_attention=True,
#         enable_attention_sharing=False,
#         adain_queries=False,
#         adain_keys=False,
#         adain_values=False,
#         svd=True
#     )
    
    
#     images = []
    
#     ylabels = []
    
#     timestamps = [*range(0, 30, 5)]
#     for i in range(len(timestamps) - 1):
#         print('TIMESTAMPS EXP')
#         sa_args.start_svd = timestamps[i]
#         sa_args.end_svd = timestamps[i+1]
        
#         ylabels.append('[' + str(timestamps[i]) + '-' + str(timestamps[i+1]) + ']')
        
#         current_images = []
#         for j in range(7):
#             sa_args.svd_remain_one = j
            
#             handler.register(sa_args, )
            
#             fix_seed(seed)   
#             current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
#             handler.remove()
            
#         images.append(get_concat_h(current_images))
            
#         images = [get_concat_v(images)]
#         images[0].save(os.path.join(base_path, image_name + '.png'))
        
        
#     current_images = []
    
#     sa_args.start_svd = -1
#     sa_args.end_svd = 10000000
#     for j in range(7):
#         sa_args.svd_remain_one = j
            
#         handler.register(sa_args, )
            
#         fix_seed(seed)   
#         current_images.append(pipeline(prompt, num_inference_steps=20, handler=handler).images[0])
#         handler.remove()
            
#     images.append(get_concat_h(current_images))
#     images = [get_concat_v(images)]
#     images[0].save(os.path.join(base_path, image_name + '.png'))
    
#     ylabels.append('[0-25]')
    
    
    
#     images = images[0]
    
#     xlabels = [*map(str, range(7))]

#     plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
#     plt.rcParams["font.size"] = 20

#     fig, axes = plt.subplots()

#     axes.imshow(images)
#     axes.set_title(prompt)
#     axes.grid(False)
#     axes.tick_params(axis=u'both', which=u'both', length=0)

    
#     width, height = images.width, images.height

#     axes.set_xticks(np.arange(0, len(xlabels)) * width / len(xlabels) + width / len(xlabels) / 2)
#     axes.set_xticklabels(xlabels)
#     axes.set_yticks(np.arange(0, len(ylabels)) * height / len(ylabels) + height / len(ylabels) / 2)
#     axes.set_yticklabels(ylabels, rotation=90)

#     fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
#     plt.close()
#     clear_output()


# # слева направо должно быть все меньше и меньше информации, 
# # потому что оставляю все более маленькие собственные значения

# # dсерху вниз непонятно, но как будто должно появляться все больше и больше инфы

# prompt = "a peacock in psychedelic illustration"
# svd_exp(prompt, 'psychedelic')

# prompt = "a pine tree in watercolor sketch"
# svd_exp(prompt, 'watercolor_0')

# prompt = "a pine tree in watercolor sketch"
# svd_exp(prompt, 'watercolor_1', seed=69)

# prompt = "a spaceship in 80s retro wave"
# svd_exp(prompt, 'retro')

# prompt = "a firewoman in minimal flat design illustration"
# svd_exp(prompt, 'flat')