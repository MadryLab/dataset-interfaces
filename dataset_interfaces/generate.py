import torch
import numpy as np
import os
import h5py
import dataset_interfaces.inference_utils as infer_utils
from dataset_interfaces.imagenet_utils import IMAGENET_COMMON_CLASS_NAMES

import json
from tqdm import tqdm

def get_pipe(encoder_root, c, use_token, pretrained_model_name_or_path): #WORKING HERE! To integrate tokens dict!
    
    tokens_dict = torch.load(f"{encoder_root}/tokens.pt")

    if use_token:

        text_encoder = infer_utils.get_text_encoder(
            model_name=encoder_root
        )
        tokenizer = infer_utils.get_tokenizer(
            model_name=encoder_root
        )
    
        pipe = infer_utils.get_pipe(text_encoder=text_encoder, tokenizer=tokenizer, model_name=pretrained_model_name_or_path)
        return pipe, tokens_dict['tokens'][c]
    
    else:
        tokenizer = infer_utils.get_tokenizer(model_name=pretrained_model_name_or_path)
        text_encoder = infer_utils.get_text_encoder(model_name=pretrained_model_name_or_path)
        pipe = infer_utils.get_pipe(text_encoder=text_encoder, tokenizer=tokenizer, model_name=pretrained_model_name_or_path)
        return pipe, tokens_dict['initializer_words'][c]


def evaluate_pipe(pipe, prompt, num_samples, random_seed=-1):
        
    cmd_args = {
        'num_inference_steps': 50,
        'guidance_scale': 7.5,
    }

    if random_seed != -1:
        cmd_args['generator'] = torch.Generator("cuda").manual_seed(random_seed)
        
    images = []
    
    for i in tqdm(range(num_samples)):

        results = pipe(prompt, **cmd_args)
        image = results.images[0]
        image = image.resize((512, 512))
        images.append(image)

    return images
        

def generate(
    encoder_root, 
    c, 
    prompts, 
    num_samples=50, 
    use_token=True,
    model_name="stabilityai/stable-diffusion-2",
    random_seed=-1):
    
    os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'
    
    # Get Stable Diffusion Pipeline pre-loaded with token
    pipe, placeholder_token = get_pipe(encoder_root, c, use_token, model_name)

    # Place token into prompt         
    if type(prompts) == list:
        
        if type(random_seed) == int:
            random_seed = [random_seed] * len(prompts)
        
        prompts = [prompt.replace("<TOKEN>", placeholder_token) for prompt in prompts]
        return [evaluate_pipe(pipe, prompts[i], num_samples, random_seed=random_seed[i]) for i in range(len(prompts))]
                   
    
    else:
        
        prompt = prompts.replace("<TOKEN>", placeholder_token)
        return evaluate_pipe(pipe, prompt, num_samples, random_seed=random_seed)
