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
    # это эксп, где я просто обнуляю собственные значения у обычных скоров, без аттеншн шеринга
    
    base_path = './images/scores-no-sharing/'
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
    images.append(get_concat_v(pipeline(prompts, num_inference_steps=20).images))
    
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        enable_attention_sharing=False,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
        score_svd=True,
        score_svd_start=0
    )
    
    # так что у нас по оси y просто обычные промпты
    # по оси x можно занулять первые компоненты, все больше и больше
    
    for i in range(1, 6):
        print('SCORES EXP')
        sa_args.score_svd_end = i
        handler.register(sa_args, )
        
        fix_seed(seed)   
        images.append(get_concat_v(pipeline(prompts, num_inference_steps=20, handler=handler).images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
        handler.remove()
    
    images = images[0]
   
    ylabels = prompts
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

    fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    plt.close()
    clear_output()


# prompt = "a peacock in psychedelic illustration"
# svd_exp(prompt, 'ts_first_to_null_psychedelic')

# prompt = "a pine tree in watercolor sketch"
# svd_exp(prompt, 'ts_first_to_null_watercolor')

# prompt = "a spaceship in 80s retro wave"
# svd_exp(prompt, 'ts_first_to_null_retro')



prompts = [
    "a spaceship in 80s retro wave",
    "a fox in stained glass design",
    "a dragon in fantasy medieval painting",
    "a peacock in psychedelic illustration",
    "a pine tree in watercolor sketch"
]
svd_exp(prompts, 'biggest')






def svd_exp(prompts, image_name='output', seed=0):
    # это эксп, где я просто обнуляю ВСЕ собственные значения у обычных скоров, 
    # без аттеншн шеринга, оставляя только самые большие
    
    base_path = './images/scores-no-sharing/'
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
    images.append(get_concat_v(pipeline(prompts, num_inference_steps=20).images))
    
    
    handler = sa_handler.Handler(pipeline)
    sa_args = sa_handler.StyleAlignedArgs(
        share_group_norm=False,
        share_layer_norm=False,
        share_attention=True,
        enable_attention_sharing=False,
        adain_queries=False,
        adain_keys=False,
        adain_values=False,
        score_svd=True
    )
    
    # так что у нас по оси y просто обычные промпты
    # по оси x оставляем первые компоненты, то есть я двигаю начало
    
    for i in range(6):
        print('SCORES EXP')
        sa_args.score_svd_start = i
        handler.register(sa_args, )
        
        fix_seed(seed)   
        images.append(get_concat_v(pipeline(prompts, num_inference_steps=20, handler=handler).images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
        handler.remove()
    
    
    images = images[0]
   
    ylabels = prompts
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

    fig.savefig(os.path.join(base_path, image_name + '_labeled.png'), bbox_inches='tight')
    plt.close()
    clear_output()


prompts = [
    "a spaceship in 80s retro wave",
    "a fox in stained glass design",
    "a dragon in fantasy medieval painting",
    "a peacock in psychedelic illustration",
    "a pine tree in watercolor sketch"
]
svd_exp(prompts, 'smallest')











def svd_exp(prompts, style, objects, image_name='output', seed=0):
    # теперь зануляю именно у референсной картинки ПЕРВЫЕ собственные значения при attention sharing'е
    
    base_path = './images/scores-reference-start/'
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
    
    handler.register(sa_args, )
    fix_seed(seed)   
    images.append(get_concat_v(pipeline(prompts, num_inference_steps=20).images))
    handler.remove()
    
    sa_args.score_svd = True
    sa_args.scores_reference_svd = True
    
    # так что у нас по оси y объекты
    # по оси x оставляем зануляю компоненты, то есть я двигаю конец
    
    for i in range(1, 6):
        sa_args.score_svd_end = i
        handler.register(sa_args, )
        
        fix_seed(seed)   
        images.append(get_concat_v(pipeline(prompts, num_inference_steps=20, handler=handler).images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
        handler.remove()
    
   
    images = images[0]
    
    ylabels = objects
    xlabels = [*map(str, range(6))]
    for i in range(len(xlabels)):
        xlabels[i] = '[0-' + xlabels[i] + ')'
    xlabels[0] = 'base'

    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    axes.set_title(style)
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
    "a cat in geometric abstract art",
    "a lion",
    "an elephant",
    "a bird",
    "a fish"
]
objects = ["cat", "lion", "elephant", "bird", "fish"]
style = "a [...] in geometric abstract art"
svd_exp(prompts, style, objects, 'geometric_no_style')
    


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
    "a sci-fi robot warrior. comic book illustration. cyberpunk theme",
    "a sci-fi spaceship",
    "a sci-fi cityscape",
    "a sci-fi alien creature",
    "a sci-fi futuristic car"
]
style = "a sci-fi [...]. comic book illustration. cyberpunk theme"
objects = ["robot warrior", "spaceship", "cityscape", "alien creature", "futuristic car"]
svd_exp(prompts, style, objects, 'sci_fi_no_style')


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
    "a firewoman in minimal flat design illustration",
    "a farmer",
    "a unicorn",
    "a dino",
    "a dog"
]
style = "a [...] in minimal flat design illustration"
objects = ["firewoman", "farmer", "unicorn", "dino", "dog"]
svd_exp(prompts, style, objects, 'flat_no_style')



















def svd_exp(prompts, style, objects, image_name='output', seed=0):
    # теперь зануляю именно у референсной картинки ПОСЛЕДНИЕ собственные значения при attention sharing'е
    
    base_path = './images/scores-reference-end/'
    Path(base_path).mkdir(parents=True, exist_ok=True)
     
    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
        clip_sample=False, set_alpha_to_one=False)
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "/home/ekneudachina/main/research/pretrained/stable-diffusion-xl/fp32", use_safetensors=False,
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
    
    handler.register(sa_args, )
    fix_seed(seed)   
    images.append(get_concat_v(pipeline(prompts, num_inference_steps=20).images))
    handler.remove()
    
    sa_args.score_svd = True
    sa_args.scores_reference_svd = True
    
    # так что у нас по оси y объекты
    # по оси x оставляем зануляю компоненты, то есть я двигаю начало
    
    for i in range(6):
        sa_args.score_svd_start = i
        handler.register(sa_args, )
        
        fix_seed(seed)   
        images.append(get_concat_v(pipeline(prompts, num_inference_steps=20, handler=handler).images))
        images = [get_concat_h(images)]
        images[0].save(os.path.join(base_path, image_name + '.png'))
        
        handler.remove()
    
    images = images[0]
   
    ylabels = objects
    xlabels = [*map(str, range(7))]
    for i in range(len(xlabels)):
        xlabels[i] = '[' + xlabels[i] + '-N]'
    xlabels[0] = 'base'

    plt.rcParams["figure.figsize"] = (len(xlabels) * 7.5, len(ylabels) * 7.5)
    plt.rcParams["font.size"] = 20

    fig, axes = plt.subplots()

    axes.imshow(images)
    axes.set_title(style)
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
    "a cat in geometric abstract art",
    "a lion",
    "an elephant",
    "a bird",
    "a fish"
]
objects = ["cat", "lion", "elephant", "bird", "fish"]
style = "a [...] in geometric abstract art"
svd_exp(prompts, style, objects, 'geometric_no_style')
    


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
    "a sci-fi robot warrior. comic book illustration. cyberpunk theme",
    "a sci-fi spaceship",
    "a sci-fi cityscape",
    "a sci-fi alien creature",
    "a sci-fi futuristic car"
]
style = "a sci-fi [...]. comic book illustration. cyberpunk theme"
objects = ["robot warrior", "spaceship", "cityscape", "alien creature", "futuristic car"]
svd_exp(prompts, style, objects, 'sci_fi_no_style')


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
    "a firewoman in minimal flat design illustration",
    "a farmer",
    "a unicorn",
    "a dino",
    "a dog"
]
style = "a [...] in minimal flat design illustration"
objects = ["firewoman", "farmer", "unicorn", "dino", "dog"]
svd_exp(prompts, style, objects, 'flat_no_style')