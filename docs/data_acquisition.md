### Data Source

#### California Fire Perimeter Dataset

Our[**California Fire Perimeter** dataset](https://gis.data.ca.gov/datasets/CALFIRE-Forestry::california-fire-perimeters-all/explore?location=37.187645%2C-120.227660%2C5.95) is provided by the California Department of Forestry and Fire Protection and covers fire events from 1950 to 2025. 

Each fire event is recorded in detail, including 
* Fire's name, 
* Year and state in which it occurred, 
* Date the alarm was raised,
* Date the fire was contained,
* Number of acres burned calculated by GIS,
* Geometry of the burned area,
* Cause of the fire and other attributes.

#### Copernicus Sentinel-2 Satellite Imagery

We utilize [Harmonized Sentinel-2 MSI: MultiSpectral Instrument, Level-2A (SR)](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED#description) dataset from the Google Earth Engine.

This data is used to calculate spectral indices like the Normalized Burn Ratio (NBR) or to feed machine learning algorithms with spectral and textural features to distinguish burned areas from other land cover changes.

Compared to Landsat 8, Sentinel-2 offers a higher revisit rate (5 days), higher spatial resolution (10 meters), and a better chance of acquiring cloud-free images. This makes it ideal for capturing pre- and post-fire scenes, especially for short-duration fire events.

A snapshot of the key bands used is shown in the table below:

| Band name | Resolution (m) | Central wavelength (nm)| Purpose |
|-----------|----------------|------------------------|---------|
|TCI_R|10|None|Red|
|TCI_G|10|None|Green|
|TCI_B|10|None|Blue|
|B08|10|842|Near infrared|
|B12|20|2190|Snow / ice / cloud discrimination|



### Data Preprocessing

To generate an image of 224×224 pixels from satellite imagery with a resolution of 10 meters per pixel, we need to capture an area of at least 100 acres. Therefore, we filter for fire events that burned between 100 and 10,000 acres to acquire manageable and consistent satellite imagery.

After defining the area of interest (AOI) from the fire perimeter, we ensure the output meets our size requirement. If the calculated image size is smaller than 224×224, we add padding. We then redefine the geometry bounds of the AOI and use this new geometry to clip the satellite image collection.

##### Parameters

* `CLOUDY_PIXEL_PERCENTAGE`: We use this metadata property to filter out images with excessive cloud cover, selecting only those with less than 20% cloud pixels.

*  `ee.Date` objects: These are used to handle temporal data in Google Earth Engine. For each fire event, we acquire two images: one pre-fire and one post-fire. The timeline for filtering images is structured as follows:

<p style="text-align:center;">|----- 60 days before -----| Alarm Date |----- Fire Duration -----| Containment Date |---- 30 days after -----|</p>

We filter the image collection for scenes captured during the 60 days before the alarm date (pre-fire) and the 30 days after the containment date (post-fire).



