# Structured Noise Injection (Official TensorFlow Implementation)
A TensorFlow implementation of structured noise injection as described in the paper. We adapt the original StyleGAN architecture code from https://github.com/NVlabs/stylegan. 

The code allows:
-  Disentangled editing of generated images (local features, mid-scale features, pose, and overall style)
-  Training a model with structured noise injection on any dataset
-  Modifying the paper's choices of grid dimensions, local code length, shared code length, and global code length 


# Examining a pretrained network
We follow the same approach as the original StyleGAN code.

First, download the pretrained network from:
https://drive.google.com/file/d/1jxzRnLX2OhPos4E1pqz-7ed4mqVyLwoQ/view?usp=sharing
and place it in the same folder as ```pretrained_SNI.py```

In order to randomly generate a few images, and preview the changes possible by our method:
```
python3 pretrained_SNI.py
```
This will generate two unique faces, and multiple figure showing specific modifications while maintaining the face identity.
Any cell of the noise grid can be changed individually by providing an 8x8 binary mask to the function ``` randomize_specific_local_codes ``` as demonstrated in the example file.

Changing the globally-shared code entry (affects pose)
![GlobalCodeExamples](/example_fakes_global.png)

Changing the codes that are shared by region (affects mid-level features such as age and accessories)
![SharedCodeExamples](/example_fakes_shared.png)

Changing all local codes (affects the fine details of the face)
![localCodeExamples](/example_fakes_alllocal.png)

Changing specific local codes (4x4 cells around the mouth)
![mouthCodeExamples](/example_fakes_mouth.png)

Changing specific local codes (3x7 cells covering the top of the head)
![hairCodeExamples](/example_fakes_hair.png)

# Training a network from scratch
To run training on the FFHQ datasets with the default settings:
``` python3 train.py ```

The network can be trained similarly to training the original StyleGAN but with a different generator. The code for our generator is included under ``` training/networks_structurednoiseinjection.py ```.

Please refer to https://github.com/NVlabs/stylegan for the datasets and code requirements.



# Testing new settings of structured noise injection
## Changing cell resolution / changing when to beging style modulation
This can be done in the synthesis part. 
In order to change where to begin style modulation the ```layer_epilogue``` function can be edited. Please note that each resolution contains two style modulation layers.
It is difficult to change cell resolution above 8x8 currently since it loses the benefits of progressive growing and lower resolution information. By default, the 8x8 resolution is used as in the paper.

## Changing global/shared/local code length
This can be done in the mapping function. The user can feed random codes that are arranged in a certain way, but the user must specify how to assemble the final code for each grid cell given the random codes. If the code lengths or arrangement are changed, the ```my_randoms``` in ```training/misc.py``` should be updated to chech the changes during training.



