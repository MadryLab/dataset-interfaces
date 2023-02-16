# Dataset Interfaces

This repository contains the code for our recent work:

**Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation** <br>
*Joshua Vendrow\*, Saachi Jain\*, Logan Engstrom, Aleksander Madry* <br>
Paper: [https://arxiv.org/abs/2302.07865](https://arxiv.org/abs/2302.07865) <br>
Blog post: TBD

FIGURE HERE TODO

## Getting started
Install using pip, or clone our repository.
```
pip install dataset-interfaces
```

**Example:** For a walkthrough of codebase, check out our [example notebook](notebooks/Example.ipynb). This notebook shows how to
construct a dataset interface for a subset of ImageNet and generate counterfactual examples. 

## Constructing a Dataset Interface
Constructing a dataset interface consists or learning a *class token* for each class in a datset, which can then be included in textual prompts. We include a tokenizer and text encoder pre-loaded with our learned token for the ImageNet dataset at `/encoder_root_imagenet` (see the next section for how to generate images using the learned tokens).

To learn a single token, we use the following function:
```
from dataset_interfaces import run_textual_inversion

embed = run_textual_inversion (
    train_path=train_path,  # path to directory with training set for a single class
    token=token,            # text to use for new token, e.g "<plate>"
    class_name=class_name,  # natrual language class description, e.g., "plate"
)
```

Once all the class tokens are learned, we can create a custom tokenizer and text encoder pre-loaded with these tokens:

```
from dataset_interfaces import create_token_dictionary

infer_utils.create_token_dictionary (
    embeds=embeds,             # list of learned embeddings (from the code block above)
    tokens=tokens,             # list of token strings
    class_names=class_names,   # list of natural language class descriptions
    encoder_root=encoder_root  # path where to store the tokenizer and encoder
)
```

## Generating Counterfactual Examples

We can now generate counterfactual examples by incorporating our learned tokens in textual prompts. The ``generate`` function generates images for a specific class in the dataset (indexed in the order that classes are passed when constructing the pre-loaded encoder). When specifying the text prompt, "<TOKEN>" acts as a placeholder for the class token.
```
from dataset_interfaces import generate

generate (
    encoder_root=encoder_root,
    c=c,                                          # index of a specific class
    prompts="a photo of a <TOKEN> in the grass",  # can be a single prompt or a list of prompts
    num_samples=10, 
    random_seed=0                                 # no seed by default
)
```

## CLIP Metric

To directly evaluate the quality of the generated image, we use *CLIP similarity* to quantify the presence of the object of interest and desired distribution shift in the image.

We can measure CLIP similarity between a set of generated images and a given caption as follows:

```
import dataset_interfaces.inference_utils as infer_utils

sim_class = infer_utils.clip_similarity(imgs, "a photo of a dog")
sim_shift = infer_utils.clip_similarity(imgs, "a photo in the grass")
```

## Citation
To cite this paper, please use the following BibTex entry:
```
@inproceedings{vendrow2023dataset,
   title = {Dataset Interfaces: Diagnosing Model Failures Using Controllable Counterfactual Generation},
   author = {Joshua Vendrow and Saachi Jain and Logan Engstrom and Aleksander Madry}, 
   booktitle = {ArXiv preprint arXiv:2302.07865},
   year = {2023}
}
```

## Maintainers:
[Josh Vendrow](https://twitter.com/josh_vendrow)<br>
[Saachi Jain](https://twitter.com/saachi_jain_)
