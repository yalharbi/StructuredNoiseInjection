# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating exemplary images using structured noise injection. """

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import training.misc as misc

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    #url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    #with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    _G, _D, Gs = misc.load_pkl('SNI.pkl')#pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        
    
    # Print network details.
    Gs.print_layers()

    
    number_unique_faces = 2 #They appear on rows of the generated figure
    variations_per_face = 5 #They appear on columns of the generated figure

    # Pick latent vector.
    #rnd = np.random.RandomState(2)
    #latents = rnd.randn(number_unique_faces, Gs.input_shape[1])

    latents = np.random.randn(number_unique_faces, Gs.input_shape[1])

    grid_latents = misc.randomize_global_codes(latents, variations_per_face)
    grid_fakes = Gs.run(grid_latents, None,  truncation_psi=0.7)
    misc.save_image_grid(grid_fakes, 'example_fakes_global.png', drange=[-1,1], grid_size=(variations_per_face,number_unique_faces))
    
    
    grid_latents = misc.randomize_global_shared_codes(latents, variations_per_face)
    grid_fakes = Gs.run(grid_latents, None, truncation_psi=0.7)
    misc.save_image_grid(grid_fakes, 'example_fakes_shared.png', drange=[-1,1], grid_size=(variations_per_face,number_unique_faces))

    grid_latents = misc.randomize_all_local_codes(latents, variations_per_face)
    grid_fakes = Gs.run(grid_latents, None,  truncation_psi=0.7)
    misc.save_image_grid(grid_fakes, 'example_fakes_alllocal.png', drange=[-1,1], grid_size=(variations_per_face,number_unique_faces))    
    
    mask = np.zeros( shape=(8,8,1))
    mask[4:8, 2:6] = 1 
    grid_latents = misc.randomize_specific_local_codes(latents, mask, variations_per_face)
    grid_fakes = Gs.run(grid_latents, None, truncation_psi=0.7)
    misc.save_image_grid(grid_fakes, 'example_fakes_mouth.png', drange=[-1,1], grid_size=(variations_per_face,number_unique_faces)) 
    
    mask = np.zeros( shape=(8,8,1))
    mask[0:3, 1:7] = 1 
    grid_latents = misc.randomize_specific_local_codes(latents, mask, variations_per_face)
    grid_fakes = Gs.run(grid_latents, None,  truncation_psi=0.7)
    misc.save_image_grid(grid_fakes, 'example_fakes_hair.png', drange=[-1,1], grid_size=(variations_per_face,number_unique_faces)) 


if __name__ == "__main__":
    main()
