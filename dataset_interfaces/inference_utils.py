import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import open_clip
import time
import os
import tqdm

from dataset_interfaces.imagenet_utils import IMAGENET_COMMON_CLASS_NAMES


def get_pipe(text_encoder=None, tokenizer=None, model_name='stabilityai/stable-diffusion-2', v2=True):
    #@title Load the Stable Diffusion pipeline
    
    if v2:
        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_name, 
            subfolder="scheduler"
        )
        return StableDiffusionPipeline.from_pretrained(
            model_name,
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            #safety_checker=None
        ).to("cuda")
    
    else:
        return StableDiffusionPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            #safety_checker=None
        ).to("cuda")


def get_tokenizer(model_name='stabilityai/stable-diffusion-2'):
    tokenizer = CLIPTokenizer.from_pretrained(
        model_name,
        subfolder="tokenizer",
    )
    return tokenizer

def get_text_encoder(model_name='stabilityai/stable-diffusion-2'):
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=torch.float16,
    )
    return text_encoder

def create_token_dictionary_root(tokens_root, encoder_root, class_names, text_encoder, tokenizer):

    placeholder_tokens = []
    
    for i in range(len(class_names)):
        
        embed_path = os.path.join(tokens_root, f"{i}/3000_learned_embeds.bin") # put embedding dim
        loaded_learned_embeds = torch.load(embed_path, map_location="cpu")
        
        placeholder_token, placeholder_emb = loaded_learned_embeds['placeholder']
        
        load_emb_in_token(text_encoder, tokenizer, placeholder_token, placeholder_emb)
        placeholder_tokens.append(placeholder_token)
        
    tokenizer.save_pretrained(f"{encoder_root}/tokenizer")
    text_encoder.save_pretrained(f"{encoder_root}/text_encoder")
    torch.save({'initializer_words': class_names, 'tokens': placeholder_tokens}, f'{encoder_root}/tokens.pt')

    
def create_encoder(embeds, tokens, class_names, encoder_root, model_name='stabilityai/stable-diffusion-2'):

    tokenizer = get_tokenizer(model_name=model_name)
    text_encoder = get_text_encoder(model_name=model_name)
    
    placeholder_tokens = []
    
    for i in tqdm.trange(len(class_names)):
        
        load_emb_in_token(text_encoder, tokenizer, tokens[i], embeds[i])
        placeholder_tokens.append(tokens[i])
        
    tokenizer.save_pretrained(f"{encoder_root}/tokenizer")
    text_encoder.save_pretrained(f"{encoder_root}/text_encoder")
    torch.save({'initializer_words': class_names, 'tokens': placeholder_tokens}, f'{encoder_root}/tokens.pt')

    
def create_imagenet_star_encoder(path, encoder_root):
    
    embeds = [torch.load(os.path.join(path, f"{i}.bin")) for i in range(1000)]
    class_names = IMAGENET_COMMON_CLASS_NAMES
    tokens = [f"<{class_names[i]}-{i}>" for i in range(len(class_names))]
    
    create_encoder(embeds=embeds, tokens=tokens, class_names=class_names, encoder_root="./encoder_root_imagenet")

    
def load_clip_model(
    arch="ViT-H-14",
    version="laion2b_s32b_b79k",
    text_encoder=None,
    tokenizer=None,
    v2=True
                   ):
    
    if v2:
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=version)
        if text_encoder:
            model.token_embedding = text_encoder.text_model.embeddings.token_embedding
            model.positional_embedding = text_encoder.text_model.embeddings.position_embedding.weight
        if tokenizer:
            eos_token = tokenizer.eos_token
            eos_token_id = tokenizer.encoder[eos_token]
            preprocess.eos_token_id = eos_token_id
            preprocess.tokenizer = tokenizer
        else:
            preprocess.tokenizer = get_tokenizer()
            preprocess.eos_token_id = None
            
        return model, preprocess
        
    else:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if text_encoder:
            clip_model.text_model = text_encoder.text_model
        if tokenizer:
            clip_processor.tokenizer = tokenizer
        return clip_model, clip_processor

def load_emb_in_token(text_encoder, tokenizer, token, embeds):
    # add a token embedding pair
    
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds    

def alt_text_forward(clip_model, clip_processor, prompts=None, inp=None, v2=True):    
    if v2:
        inputs = clip_processor.tokenizer(prompts, padding='max_length', return_tensors="pt")
        text = inputs['input_ids'].cuda()
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x = clip_model.token_embedding(text)

                x = x + clip_model.positional_embedding
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = clip_model.transformer(x, attn_mask=clip_model.attn_mask)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = clip_model.ln_final(x)
                
                if clip_processor.eos_token_id is not None:
                    text_embeds = x[torch.arange(x.shape[0]),
                            (text==clip_processor.eos_token_id).int().argmax(dim=-1)] @ clip_model.text_projection
                    
                else:
                    text_embeds = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection
                
                text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
                return text_embeds
    else:
        
        if inp is None:
            inp = clip_processor(text=prompts, images=None, return_tensors='pt', padding=True)
        else:
            assert prompts is None and imgs is None

        eos_token = clip_processor.tokenizer.eos_token
        eos_token_id = clip_processor.tokenizer.encoder[eos_token]
        inp = {k: v.cuda() for k, v in inp.items() if v is not None}
        input_ids = inp['input_ids']
        output_attentions = clip_model.config.output_attentions
        output_hidden_states = clip_model.config.output_hidden_states
        return_dict = clip_model.config.use_return_dict
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                # get text_outputs
                text_outputs = clip_model.text_model(
                    input_ids=input_ids,
                    attention_mask=inp['attention_mask'],
                    position_ids=None,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict)
                last_hidden_state = text_outputs.last_hidden_state
                text_embeds = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=inp['input_ids'].device),
                    (inp['input_ids']==eos_token_id).int().argmax(dim=-1)
                ]
                text_embeds = clip_model.text_projection(text_embeds)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                return text_embeds
    
def alt_img_forward(clip_model, clip_processor, imgs, v2=True):
    # Use the line below to preprocess PIL images: 
    #imgs = torch.stack([clip_processor(img) for img in imgs]).cuda()
    imgs = imgs.cuda()
    if v2:
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                image_embeds = clip_model.encode_image(imgs)
                image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
                return image_embeds    
    else:
        output_attentions = clip_model.config.output_attentions
        output_hidden_states = clip_model.config.output_hidden_states
        return_dict = clip_model.config.use_return_dict
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                vision_outputs = clip_model.vision_model(
                    pixel_values=imgs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

                image_embeds = vision_outputs[1]
                image_embeds = clip_model.visual_projection(image_embeds)

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                return image_embeds
            
            
            
def clip_similarity(imgs, prompt):

    clip_model, clip_processor = load_clip_model()
    clip_model = clip_model.eval().cuda()
    
    text_embeds = alt_text_forward(clip_model, clip_processor, prompts=[prompt])
    
    imgs_processed = torch.stack([clip_processor(img) for img in imgs]).cuda()
    image_embeds = alt_img_forward(clip_model, clip_processor, imgs=imgs_processed)
    
    return torch.matmul(text_embeds, image_embeds.t())[0].cpu()
    