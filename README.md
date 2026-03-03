# Vertex AO bakery for Assetto Corsa

A tool to bake vertex ambient occlusion for racing simulator Assetto Corsa with Custom Shaders Patch. Supports tracks and cars. Initially made possible with NVIDIA OptiX, forked from [this project](https://github.com/nvpro-samples/optix_prime_baking), now instead uses Embree for CPU-only baking and Vulcan for GPU-accelerated baking avaiable, functional on any hardware.

[![Model by Kunos](https://i.imgur.com/pzCpqUV.jpg)](https://i.imgur.com/pzCpqUV.jpg)

*Nowadays, CSP expects VAO to be present, especially for tracks, as it uses this information to estimate all sorts of things.*

## Features

- Tracks:
  - Both “.kn5” and “.ini” files are supported as inputs;
  - Special trees baking (without self-occlusion, with normals facing up);
  - Special grass baking (without any shadows from grass, with normals facing up);
  - Optional trees transparency factor;
  - Skips dynamic objects to shadow them dynamically later on;
  - Baking of extra samples along the track for occlusion for dynamic objects;
  - Baking of shadows cast by CSP procedural trees, as well as shadows cast upon CSP procedural trees (also, this tool will compile those trees into a binary list, greatly improving loading performance, so I would highly recommend to use it even if you don’t need VAO, just disable VAO stuff in configs);
  - Rebaking of overly darken areas with increased number of rays per triangle.
- Cars:
  - Full LODs support, as well as COCKPIT_HR/COCKPIT_LR;
  - Special processing for seatbelts;
  - Special processing for rotating objects to keep shadows uniform;
  - AO splitting for doors, steering wheel and other animations, allowing to transition between AO values realtime ([demo](https://gfycat.com/felinepassionateangwantibo));
  - Baking shadows for driver;
  - Baking shadows from driver in an alternative AO set;
  - Raising ambient brightness to compensate for new AO;
  - Extra pass for adding a bit of light “bounced” from the ground.
- Takes into account light bounces to keep AO brighter;
- Alter brightness, opacity and gamma for resulting AO;
- Adjust AO per-object if necessary;
- Adjust sampling offsets per-object to get rid of arifacts in complicated cases;
- Add extra KN5s with casters or emissive entities to help with tunnels and such;
- Many more potential tweaks — check out the config!

## Tips

- Various options could be changed in `baked_shadow_params.ini`;
- GPU baking requires NVIDIA RTX 20…, AMD RX 6000, Intel Arc A-Series or newer, but if it’s not available, Embree should be pretty fast as well.

## Credits

- Project based on [Optix Prime baking](https://github.com/nvpro-samples/optix_prime_baking);
- Made possible thanks to [Intel Embree](https://www.embree.org/).
