import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder

from PIL import Image
import matplotlib.pyplot as plt

def load_initializer_text(text_encoder, tokenizer, init_str, placeholder_name):
    tokenization = tokenizer.tokenize(init_str)
    tok_ids = tokenizer.convert_tokens_to_ids(tokenization)
    embs = text_encoder.get_input_embeddings().weight.data[tok_ids].mean(0) # take average of the tokens
    num_added_tokens = tokenizer.add_tokens(placeholder_name)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {placeholder_name}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(placeholder_name)
    text_encoder.get_input_embeddings().weight.data[token_id] = embs
    return token_id

class SplitEmbedding(nn.Module):
    def __init__(self, num_main, num_aux, dim, magnitude_targets=None):
        super().__init__()
        self.main_embedding = nn.Embedding(num_main, dim)
        self.aux_embedding = nn.Embedding(num_aux, dim)
        self.num_main = num_main
        self.num_aux = num_aux
        self.dim = dim
        if magnitude_targets is not None:
            self.register_buffer('magnitude_targets', magnitude_targets)
        else:
            self.magnitude_targets = None
        
    def normalize_aux(self):
        if self.magnitude_targets is None:
            return self.aux_embedding.weight
        else:
            target = torch.linalg.norm(self.main_embedding(self.magnitude_targets), dim=1, keepdims=True)
            current = torch.linalg.norm(self.aux_embedding.weight, dim=1, keepdims=True)
            return self.aux_embedding.weight *target / current
        
    @property
    def weight(self):
        aux = self.normalize_aux()
        return torch.cat([self.main_embedding.weight, aux])
    
    def initialize_from_embedding(self, emb):
        self.main_embedding.weight.data = emb.weight.data[:self.num_main]
        self.aux_embedding.weight.data = emb.weight.data[self.num_main:]
        
        
    def forward(self, x):
        W = self.weight
        return W[x]
    
    def get_collapsed_embedding(self):
        new_emb = nn.Embedding(self.num_main + self.num_aux, self.dim)
        new_emb.weight.data = self.weight.data
        return new_emb

    
class ImageNet_Star_Dataset(ImageFolder):
    
    def __init__(self, path, shift="base", mask_path=None, transform=None):
            
        super().__init__(os.path.join(path, shift), transform=transform)
        
        if mask_path is not None:
            self.mask = np.load(mask_path, allow_pickle=True).item()[shift]
            self.mask_indices = np.arange(len(self.mask))[self.mask==1]
        else:
            self.mask = None
        
    def __getitem__(self, index):
    
        if self.mask is not None:
            index = self.mask_indices[index]

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.mask is not None:
            return self.mask.sum()
        else:
            return len(self.samples)
        
        
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def visualize_samples(images, names, desc, title=None, figsize=None, fontsize=14, dpi=100):
    
    N = len(names)
    M = len(desc)

    if figsize == None:
        figsize=(2*M-1.4, 2*N)
        
    fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=dpi)
    subfigs = fig.subfigures(nrows=N, ncols=1)


    for i in range(N):

        subfig = subfigs[i]

        ax = subfig.subplots(nrows=1, ncols=M)

        for j in range(M):

            ax[j].imshow(images[i][j])
            ax[j].set_xticks([])
            plt.setp(ax[j].spines.values(), visible=False)
            ax[j].tick_params(left=False, labelleft=False)
            
            if j > 0:
                ax[0].set_ylabel(' ', fontsize=fontsize)

        ax[0].set_ylabel(names[i], fontsize=fontsize)


        if i == 0:

            for j in range(M):

                ax[j].set_title(desc[j], fontsize=fontsize)   
        
        else:
            for j in range(M):
                ax[j].set_title(' ', fontsize=fontsize)

    fig.suptitle(title, fontsize=fontsize+5)

    plt.gca().set_xticks([])
        
    plt.show()