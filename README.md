# lidisa

`lidisa` is a **li**ghtweight implementation of **di**rect **s**ampling in Python. This is a simplified version of the algorithm described in *The Direct Sampling method to perform multiple‐point geostatistical simulations* published by Mariethoz, Renard and Straubhaar in *Water Resources Research* (2010). It requires only numpy and numba as dependencies and makes use of the numba just-in-time compiler to significantly speed up sampling. `lidisa` is able to conduct both conditional and unconditional sampling of new images based on training data.



Currently, `lidisa` supports only categorical-valued images. Images with continuous values will be supported in a future release. See the attached Jupyter notebook for a detailed example workflow using `lidisa` to conduct stochastic simulation.



