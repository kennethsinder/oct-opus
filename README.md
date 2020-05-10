# oct-opus

## Summary

Image processing for [OCT](https://en.wikipedia.org/wiki/Optical_coherence_tomography) retinal cross-sections to infer blood flow from single acquisitions of each spot. The dominant implementation we're proceeding with the pix2pix cGAN (conditional generative adversarial network). 2020 University of Waterloo Software Engineering capstone design project.

The entry point into this software is via `cgan.py` which calls into the code in the `/cgan` directory.
We are also working on implementing a CNN-based baseline under the `/cnn` directory.

**Official project webpage**: [kennethsinder.github.io/oct-opus](https://kennethsinder.github.io/oct-opus/) (brought to you by the HTML in the `/docs` directory)

Information on how our software came about and how to use it, including to generate inferred OMAG-like images and to update the model with additional training data (more images), can be found in our [**official manual**](https://docs.google.com/document/d/1kIQ93V5Y-wmiLAy-IjhyLocZaWY7mGQE2tBXcIRTTXM/edit?usp=sharing).

## Other Resources

- Simple GUI tool for one-off histogram equalization for normalizing images: [image-normalization-gui](https://github.com/kennethsinder/image-normalization-gui)
