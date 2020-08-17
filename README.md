# oct-opus

## Summary

Image processing for [OCT](https://en.wikipedia.org/wiki/Optical_coherence_tomography) retinal cross-sections to infer blood flow from single acquisitions of each spot. The dominant implementation we're proceeding with the pix2pix cGAN (conditional generative adversarial network). 2020 University of Waterloo Software Engineering capstone design project.

The entry point into this software is via `cgan.py` which calls into the code in the `/cgan` directory.
We also implemented a CNN (convolutional neural network) baseline under the `/cnn` directory, accessible via `cnn.py`.

**Official project webpage**: [kennethsinder.github.io/oct-opus](https://kennethsinder.github.io/oct-opus/) (brought to you by the HTML in the `/docs` directory)

Information on how our software came about and how to use it, including to generate inferred OMAG-like images and to update the model with additional training data (more images), can be found in our [**official manual**](https://docs.google.com/document/d/1kIQ93V5Y-wmiLAy-IjhyLocZaWY7mGQE2tBXcIRTTXM/edit?usp=sharing). There is additional documentation available in each of the Markdown (`.md`) files (including code snippets) in the `docs/` folder. To best view Markdown files with all of the appropriate formatting, a code editor like VS Code or PyCharm is recommended. Clicking into the file on the GitHub website works too.

Many files have header comments at the top referencing original source papers and links from where we have drawn code snippets and ideas. For a full list of references, please refer to our manuscript in the proceedings of OP502 Applications of Machine Learning of the SPIE Optics + Photonics conference.

## Other Resources

- Simple GUI tool for one-off histogram equalization for normalizing images: [image-normalization-gui](https://github.com/kennethsinder/image-normalization-gui)
