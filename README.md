# VP-NgbnNET

## MainTraffic.ipynb 
This notebook serves as the primary tool for training on real traffic estimation data.

## Simulation.ipynb 
This notebook is designed for testing simulation data. It aims to evaluate the model's ability to effectively recognize the inverted U-shape and subsequently incorporate noise.

## Two Versions of the Network
Distance-Based Version
This version is straightforward to use when the time steps range from 1 to 24 (e.g., [1, 2, 3, ..., 24]). Since the distances remain consistent, it simplifies training with real traffic data. 
eg. 24 is time steps of daily in the real traffic data.

## Time-Based Version
In this version, time is continuous, with distances likely set to 0.5 or 0.7, and time steps can extend to 40 or 50. This model is particularly effective at recognizing patterns in the data.


