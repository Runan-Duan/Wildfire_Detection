

### Data Source

Our **California Fire Perimeter** dataset is provided by the California Department of Forestry and Fire Protection and covers fire events from 1950 to 2025. 

Each fire event is recorded in detail, including 
* Fire's name, 
* Year and state in which it occurred, 
* Date the alarm was raised,
* Date the fire was contained,
* Number of acres burned calculated by GIS,
* Geometry of the burned area,
* Cause of the fire and other attributes.


**Copernicus Sentinel-2** (COPERNICUS/S2_SR_HARMONIZED) data involves using the surface reflectance information in the data to calculate spectral indices such as the normalized Burn Ratio (NBR), or employs machine learning algorithms with spectral and textural features to distinguish burned areas from other land cover changes. 

Compared to Landsat 8, Sentinel-2 is characterized by a higher repetition rate (5 days), higher resolution (10 meters) and a higher change of acquiring cloud-free images. This makes Sentinel-2 ideal for caputuring pre- and post-fire scenes, especially for short-duration fire events.

In our project, we work on [Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#description), a snap of bands information as show in the below table:

| Band name | Resolution (m) | Central wavelength (nm)| Purpose |
|-----------|----------------|------------------------|---------|
|TCI_R|10|None|Red|
|TCI_G|10|None|Green|
|TCI_B|10|None|Blue|
|B08|10|842|Near infrared|
|B12|20|2190|Snow / ice / cloud discrimination|



### Data Preprocessing

In order to generate an image sized 224 * 224 pixels from satellite imagery with a resolution of 10 metres, we need to capture at least 100 acres of events. 
Thus we adopt fire events covering between 100 and 10,000 acres to acquire the corresponding satellite imagery.

After collecting the area of interest, we will take the expected output size into consideration here, by computing the image size and adding padding if the export image size is smaller than 224 * 224. We will then give the area of interest new geometry bounds and clip the satellite with the new geometry.

##### 2.2 Parameters

* `CLOUDY_PIXEL_PERCENTAGE`: with this metadata property , it's easier for us to filter images less than a certain cloud pixel percentage, here we choose images with less than 20% cloud pixels.

*  `ee.Date` objects allow us to convert a string to a date in Google Earth Engie and handle temporal data. For each fire event, we would like to acquire two images: one showing the area without fire and another representing the area's status after the fire. Therefore, we sort the timeline for each fire event as follows:

<p style="text-align:center;">|-----60 days before------|alarm date|-----on fire----|containment date|----30 days after-----|</p>

And filter images in time periods `60 days before` and `30 days after`.



### Dataset Setup


