# Dstl-Satellite-Imagery-Features-Detection
1. Dstl provides you with 1km x 1km satellite images in both 3-band and 16-band formats. The goal is to detect and classify the types of objects found in these regions.
2. Every object class is described in the form of Polygons and MultiPolygons, which are simply a list of polygons. We provide two different formats for these shapes: GeoJson and WKT. These are both open source formats for geo-spatial shapes.
3. Apply U-Net architecture to solve the image segmentation problem.
![Model](model.png)
