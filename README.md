# CommonFate

## Background
Common Fate describes a perceptual phenomenon where many objects moving in sync
seem to form a shape greater than their individual components. A good illustration of this 
is how this murmuration of starlings seems to morph into a larger shape:

https://www.youtube.com/watch?v=V4f_1_r80RY

Common Fate is an example of a larger family of perceptual quirks known as "Gestalt Perception".
Gestalt Perception (generally - how we see the "Whole" from the "pieces") is a top-down perceptual 
mechanism we use all the time.

It's something that comes naturally to humans, but imbuing machines with similar perceptual biases
isn't as straightforward. Here we seek to build a model of Common Fate perception using a dataset 
consisting of 3 dimensional white shapes on a white background, where the shapes are textured
with black dots. The shapes come from a class of shapes called "Superquadrics", and can be described 
with a few parameters.

We argue that an agent with a generative model of these underlying shapes should be able to solve
the problem of identifying the shape / rotation of these shapes, and that is a similar mechanism
to how humans may be solving the problem.

## How to run the code
To run the generative pipeline, you can first generate the meshes of your superquadrics by running

```python generate_meshes.py```

To render the scenes you will need to install Blender. Once you install Blender, you'll need to pip install
the relevant libraries to Blender's built in python installation. It's a pretty inconvenient process, so 
we should probably provide a Singularity / Docker container for this work. 

Once you have the Blender environment set up (and presumably a GPU or two to speed things along), you can run
```Blender -b --python blender_qrotation.py```
to render out the superquadric rotation scenes.
