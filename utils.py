# import json
# from imgaug.augmentables.polys import Polygon, PolygonsOnImage

# # create a PolygonsOnImage object
# polygon = Polygon([(0,0), (100,0), (100,100), (0,100)])
# polys = PolygonsOnImage([polygon], shape=(256, 256, 3))

# # convert the PolygonsOnImage object to list of coordinates
# coords = polys.to_xy_array()

# # create the labelme-compatible dictionary
# labelme_dict = {
#     "version": "4.5.7",
#     "flags": {},
#     "shapes": [
#         {
#             "label": "polygon",
#             "points": coords.tolist(),
#             "group_id": None,
#             "shape_type": "polygon",
#             "flags": {}
#         }
#     ],
#     "imagePath": "path/to/image/file.jpg",
#     "imageData": None,
#     "imageHeight": 256,
#     "imageWidth": 256
# }

# # save the labelme-compatible dictionary to a file
# with open("labelme_data.json", "w") as f:
#     json.dump(labelme_dict, f)


import json
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
'''
# create PolygonsOnImage object with multiple Polygon objects
polygons = [Polygon([(0,0), (100,0), (100,100), (0,100)]), Polygon([(50,50), (75,50), (75,75), (50,75)])]
polys = PolygonsOnImage(polygons, shape=(256, 256, 3))

# convert the list of polygons to a list of lists of coordinates
coords = [[[(x, y) for (x, y) in polygon.exterior]] for polygon in polygons]

# create the labelme-compatible dictionary
labelme_dict = {
    "version": "4.5.7",
    "flags": {},
    "shapes": [
        {
            "label": "polygon",
            "points": coord_list,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        } for coord_list in coords
    ],
    "imagePath": "path/to/image/file.jpg",
    "imageData": None,
    "imageHeight": 256,
    "imageWidth": 256
}

# save the labelme-compatible dictionary to a file
with open("labelme_data2.json", "w") as f:
    json.dump(labelme_dict, f)
'''

import json
import numpy as np

# define custom JSON encoder
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16,np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                               np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# create PolygonsOnImage object with multiple Polygon objects
polygons = [Polygon([(0,0), (100,0), (100,100), (0,100)]), Polygon([(50,50), (75,50), (75,75), (50,75)])]
polys = PolygonsOnImage(polygons, shape=(256, 256, 3))

# convert the list of polygons to a list of lists of coordinates
coords = [[[(x, y) for (x, y) in polygon.exterior]] for polygon in polygons]

# create the labelme-compatible dictionary
labelme_dict = {
    "version": "4.5.7",
    "flags": {},
    "shapes": [
        {
            "label": "polygon",
            "points": coord_list,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        } for coord_list in coords
    ],
    "imagePath": "path/to/image/file.jpg",
    "imageData": None,
    "imageHeight": 256,
    "imageWidth": 256
}

# save the labelme-compatible dictionary to a file
# with open("labelme_data.json", "w") as f:
#     json.dump(labelme_dict, f, cls=NumpyEncoder)
