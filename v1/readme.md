# Sketch Colorization V1  

## Things I've learned: 

- *tf.keras.layers* and *keras.layers* **are not** the same things. (This caused issues with *tf.train.Checkpoint*.)  
- ~~The nets train a bit faster with tanh and sigmoid as the last activations for the generator and discriminator respectively, but results in the generator seemingly disregarding the style hint (every image becomes sepia colored).~~
- Not using any activations for the last layers results in some color being added to the output but overall not looking as good (too much detail lost in the outlines).
- Read documentation more carefully. (Couldn't find L1 loss in tf so decided to make my own, pretty bad version, where I forgot to mean the sum. Forgot to look for MAE.)  
- Reiterating above, tf's vgg_preprocess function rescales and recenters pixel values, which I had done in the dataloader, resulting in the sepia tinge. 
        
        image /= 127.5
        image -= 1
- Use *tf.train.CheckpointManager* for training. Also prevents the checkpoints folder from growing too huge.  
- Bigger batch size != faster training. Batch size 24 took ~26 minutes to train on a Tesla K80 vs ~11 mins for batch size 16 (the saved version was trained on a P100 which is a lot more powerful).  
- A VGG model trained on image data from a similar distribution (anime images) probably would've worked better. 
- Also, using a custom VGG would allow the right input size(256, 256, 3) to be used. What I've done, getting the center_crop(224, 224), is a bandaid fix:

        image = image[ : , 16:240, 16:240, : ]
- ~~Should've added a variable to track loss. Some of the later epochs seems to produce worse outputs~~
- Added wandb for tracking.