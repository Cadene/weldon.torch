# Weldon Pooling for Torch7

***Pull requests are more than welcome!***

Weldon Pooling is a critical module developped in the frame of a conference paper published at CVPR 2016 ["WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks"](http://webia.lip6.fr/~durandt/pdfs/2016_CVPR/Durand_WELDON_CVPR_2016.pdf).

The implementation included in this repositery was made by the first author, namely [Thibaut Durand](http://webia.lip6.fr/~durandt). [Remi Cadene](http://remicadene.com) helped a bit by providing support and, above all, butter croissants.

### Installation

```
$ git clone https://github.com/Cadene/weldon.torch
$ cd weldon.torch
$ luarocks make rocks/weldon-scm-1.rockspec
```

### How to

```lua 
require 'nn'
require 'WeldonPooling'
m = nn.WeldonPooling(5, 2)
m:forward(torch.ones(10,5,10,10))
```

### References

If this helps you, please cite the paper:

```
@inproceedings{Durand_WELDON_CVPR_2016,
author = {Durand, Thibaut and Thome, Nicolas and Cord, Matthieu},
title = {{WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2016}
}
```

### Licence

MIT License