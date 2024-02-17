import os
from osgeo import gdal
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument( "--feature_path", default='origin_data/feature/', type=str)
parser.add_argument( "--label_path", default='origin_data/label/label1.tif', type=str)
args = parser.parse_args()

def get_tiff_attributes(tiff_path):
    tiff = gdal.Open(tiff_path)
    width, height = tiff.RasterXSize, tiff.RasterYSize
    transform = tiff.GetGeoTransform()
    return width, height, transform

consistency = {"w": [], "h":[], "transform":[]}
#label 
w, h, transform = get_tiff_attributes(args.label_path)
consistency["w"].append(w)
consistency["h"].append(h)
consistency["transform"].append([int(value) for value in transform])
#feature
for tif_data in os.listdir(args.feature_path,):
    w, h, transform = get_tiff_attributes(os.path.join(args.feature_path, tif_data))
    print('Reading '+tif_data+' data.')
    consistency["w"].append(w)
    consistency["h"].append(h)
    consistency["transform"].append([int(value) for value in transform])
consistency["w"] = set(consistency["w"])
consistency["h"] = set(consistency["h"])
if(len(consistency["w"])!=1):
    print("The width of the data is different")
    print(consistency["w"])
    sys.exit()
if(len(consistency["h"])!=1):
    print("The height of the data is different")
    print(consistency["h"])
    sys.exit()
flag = consistency["transform"][0]  # Get the first tuple in the list
if(not(all(t == flag for t in consistency["transform"]))):
    print("The transform of the data is different")
    print(consistency["transform"])
    sys.exit()

print("Congratulations! There are no problems with the data.")
