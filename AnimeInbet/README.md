
This directory has code for feeding your own data to [AnimeInbet](https://github.com/lisiyao21/AnimeInbet),
which requires vectorizing it. **The input resolution is 720x720.**

* `vectorize.py`: "vectorizes" a raster line drawing by skeletonizing the black pixels and then flood-filling
  them recursively (with a limit of 1000 on recursion depth...) and connecting pixels while doing so.
  very, very hacky, you might want to change it if you decide to use it.
* `custom_data.py`: more reasonable code loading .png and .json files into Python objects that can be fed
  to the inbetweening inference code.
* `visualize_custom.py`: take the model output and produce a black-and-white png image showing the resulting
  inbetween
* `PATCHES.py`: a couple of patches to apply to AnimeInbet source code in order to feed & visualize custom data.
* `sample.py`: code for creating a "data sample" = a graph of points connected by lines, and saving it as a
  pair of .png and .json files. `vectorize.py` uses this to write its output, which `custom_data.py` can then
  load into Python objects you can feed to inference.


