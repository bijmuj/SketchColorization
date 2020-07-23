# Sketch Colorization V1  

## Things I've learned: 

- *tf.keras.layers* and *keras.layers* **are not** the same things. (This caused issues with *tf.train.Checkpoint*.)  
- The nets train a bit faster with tanh and sigmoid as the last activations for the generator and discriminator respectively, but results in the generator seemingly disregarding the style hint (every image becomes sepia colored).  
- Not using any activations for the last layers results in some color being added to the output but overall not looking as good (too much detail lost in the outlines).
- Read documentation more carefully. (Couldn't find L1 loss in tf so decided to make my own, pretty bad version, where I forgot to mean the sum. Forgot to look for MAE.)  
- Saving the weights in an hdf5 file and loading does not seem to work for continued training:  
        
        discriminator.save_weights('disc.h5')  
        ...  
        discriminator.load_weights('disc.h5')  
        # Doing this does not result in a pretrained discriminator being loaded.
        # Haven't figured out why.
- Use *tf.train.CheckpointManager* for training. Also prevents the checkpoints folder from growing too huge.  
- Bigger batch size != faster training. Batch size 24 took ~26 minutes to train on a Tesla K80 vs ~11 mins for batch size 16 (the saved version was trained on a P100 which is a lot more powerful).  
- A VGG model trained on image data from a similar distribution (anime images) probably would've worked better. 
- Also, using a custom VGG would allow the right input size(256, 256, 3) to be used. What I've done, getting the top-left crop(224, 224), is a bandaid fix:

        image = image[ : , :224, :224, : ]
- Should've added a variable to track loss. Some of the later epochs seems to produce worse outputs. 