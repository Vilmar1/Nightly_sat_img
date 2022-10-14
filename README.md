# Nightly Satellite Image

This repository aims to predict the energy consumption in a city using satellite imagery with Deep Learning. The main contributions is to test when Convolutional Neural Networks (CNNs) can be used to achieve the proposed goal.

## Data
The Suomi National Polar-orbiting Partnership (or Suomi NPP) satellite collects the radiation in the earth daily. Next, it sends this information to the US National Oceanic and Atmospheric Administration (NOAA), generating the VIIRS database. Surprisingly, satellites could also capture night lights from human settlements, allowing various types of socioeconomic research, including estimating income, wealth, and GDP. Recent research efforts have replaced the DMSP images with the new generation of NTL, namely the Visible Infrared Imaging Radiometer Suite (VIIRS) Day/Night Band (DNB). See more details in https://eogdata.mines.edu/products/vnl/.

The last dataset is the Brazilian daily energy, where no open sources are available. Nevertheless, the Electric Energy Trading Chamber (CCEE in Portuguese) — a Brazilian public agency related to the Ministry of Energy — promotes auctions in the Brazilian open energy market hourly. These auctions occur in a secondary market, where prices and demand can vary to match the supply in each location. Therefore, the trade records are freely available on its website, including all classes (generation, commercialization, and distribution). We aggregated these two  daily datasets for brazilian cities of Porto Alegre between May 2021 and October 2021 as a proxy for the energy consumption in the period.


<p align="center">
  <img src="https://user-images.githubusercontent.com/38505459/185278031-9df3c9f9-d81c-4acd-a4cd-c49b2fa27a40.png" width="400">
</p>


## Methods

The preprocessing of the images is out of the scope of this repository. But, in a nutshell, the previous parts includes: dowloading the images, specifying the Coordinate Reference Systems, applying a symbology to enchance contrast and, cutting the images to the specific area. 

Then, after opening the satellite and energy databases for analysis, we normalize them to input in a deep learning framework.

For this specific objective, diferent types of CNNs architectures were tested. The one which led to the best results contains two convolutional layers followed by a Max Pooling layers in each one. After, the output is flattened an then joined to a dense layer with 50 neurons. A last neuron is added to get the regression output.

The hiperparameters used were: `test_size=.25`, `batch_size=50`, `epochs=60`. In both convolutional layers, we used `filters=32`, `kernel_size=2`, `activation='relu'`. The network is optimized with `adam` and corrected by the mean squared loss. To avoid overfitting, an Early Stopping phase is applied in the validation loss with `patience=5`. 
 
## Results

The results were achieved by using the Mean Absolute Percentage Error to avoid misconceptions when comparing cities with diferent sizes. This metric achieved 5% of error in both cities, with results varying arround this value due to stochasticity.
 
 Obs.: there are commented instructions in the first lines of the code if you want to use a GPU.

