## AO bakery for Assetto Corsa

This is a CUDA-accelerated ambient occlusion bakery for racing simulator Assetto Corsa with Custom Shaders Patch, supporting tracks and cars. Made possible with NVIDIA OptiX Prime, forked from [this project](https://github.com/nvpro-samples/optix_prime_baking).

[![Model by Kunos](https://i.imgur.com/pzCpqUV.jpg)](https://i.imgur.com/pzCpqUV.jpg)

## Features

- Tracks:
  - Both KN5 and INI files are supported as inputs;
  - Special trees baking (without self-occlusion, with normals facing up);
  - Special grass baking (without any shadows from grass, with normals facing up);
  - Optional trees transparency factor;
  - Skips dynamic objects to shadow them dynamically later on;
  - Baking of extra samples along the track for occlusion for dynamic objects;

- Cars:

  - Full LODs support, as well as COCKPIT_HR/COCKPIT_LR;
  - Special processing for seatbelts;
  - Special processing for rotating objects to keep shadows uniform;
  - AO splitting for doors, steering wheel and other animations, allowing to transition between AO values realtime ([demo](https://gfycat.com/felinepassionateangwantibo));
  - Baking shadows for driver;
  - Baking shadows from driver in an alternative AO set;
  - Raising ambient brightness to compensate for new AO;
  - Extra pass for adding a bit of light “bounced” from the ground;

- Alter brightness, opacity and gamma for resulting AO;
- Adjust AO per-object if necessary;
- Adjust sampling offsets per-object to get rid of arifacts in complicated cases.

## Tips

- Various options could be changed in `baked_shadow_params.ini`;
- To run it on CPU, set `CPU_ONLY = 1`, although at least on my PC with GTX 1060 it runs about 
100 times faster with hardware acceleration (2.5 s vs 4 min for main pass).

## Credits

- Project based on [Optix Prime baking](https://github.com/nvpro-samples/optix_prime_baking);
- Powered by NVIDIA Optix Prime.
