"""
Abstract class for detection object parsers

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""
import itertools
from shapely.ops import cascaded_union
from parsers.detection.metadata import Metadata


flatten = itertools.chain.from_iterable


def get_poly_cords(geometry):
    """
    Returns the coordinates ('x|y') of edges/vertices of a Polygon/others
    https://stackoverflow.com/a/55662063
    """

    # Parse the geometries and grab the coordinate

    if geometry.type == 'Polygon':
        return list(zip(list(geometry.exterior.coords.xy[0]), list(geometry.exterior.coords.xy[1])))

    if geometry.type == 'MultiPolygon':
        all_x = []
        all_y = []
        for ea in geometry:
            all_x.append(list(ea.exterior.coords.xy[0]))
            all_y.append(list(ea.exterior.coords.xy[1]))

        x_cods = list(flatten(all_x))  # [item for item in _list for _list in all_x]
        y_cods = list(flatten(all_y))  # [item for item in _list for _list in all_y]
        return list(zip(x_cods, y_cods))

    else:
        # Finally, return empty list for unknown geometries
        raise Exception('Not supported!')


class DetectionObject(object):
    """

    :Description:

    The **DetectionObject** class holds properties that are needed for a object
    in each annotation file.

    - annotation type: detection.

    """

    def __init__(self, class_name="", box={'x_min': 0.0, 'y_min': 0.0,
                                           'x_max': 0.0, 'y_max': 0.0},
                 metadata=Metadata(), polygon_instance=None):
        """

        :param class_name: object class name.
        :param box: bounding box co-ordinates.
        :param metadata: meta data related to object annotation.

        :return: An **DetectionObject** object.

        :Description:

        This is the **DetectionObject** class constructor.

        """
        self.__class_name = class_name
        self.__box = box
        self.__metadata = metadata
        self.__polygon_instance = polygon_instance

        self.__pascalobject = {'class': self.__class_name,
                               'metadata': self.__metadata,
                               'Box': self.__box}

    def set_class_name(self, class_name):
        """

        :param class_name: annotation object class

        :Description:

        setter for class name

        """
        self.__class_name = class_name.lower()
        self.__pascalobject['class'] = class_name

    def set_metadata(self, metadata):
        """

        :param metadata: annotation object metadata

        :Description:

        setter for metadata

        """
        self.__metadata = metadata
        self.__pascalobject['metadata'] = metadata

    def set_box(self, x_min, y_min, x_max, y_max):
        """

        :param x_min: x_min for bounding box
        :param y_min: y_min for bounding box
        :param x_max: x_max for bounding box
        :param y_max: y_max for bounding box

        :Description:

        setter for bounding box

        """
        self.__box = {'x_min': x_min, 'y_min': y_min,
                      'x_max': x_max, 'y_max': y_max}
        self.__pascalobject['Box'] = {'x_min': x_min, 'y_min': y_min,
                                      'x_max': x_max, 'y_max': y_max}

    def get_class_name(self):
        """

        :Description:

        Getter for the class name.

        :return: The class name in lower case, as a string.

        """
        return self.__class_name

    def get_box(self):
        """

        :Description:

        Getter for the detection box.

        :return: The rectangle of the detection box, as tuple
        (x_min, y_min, x_max, y_max).

        """
        return self.__box

    def get_metadata(self):
        """

        :Description:

        Getter for metadata.

        :return: The metadata of the object, as a **Metadata** object.

        """
        return self.__metadata

    def get_pascal_style_info(self):

        return self.__pascalobject

    def set_polygon(self, polygon_instance):
        '''
        save the polygon representation
        '''
        self.__polygon_instance = polygon_instance

    def get_polygon(self):
        '''
        return polygon instancce
        '''
        return self.__polygon_instance

    def get_poly_cods(self):
        '''
        '''
        return get_poly_cords(self.__polygon_instance)

    def merge(self, object_to_merge, class_name=None):
        """

        :Description:

        Merge this object with object_to_merge.

        :param object_to_merge: The object to merge as **DetectionObject**.

        :param class_name: The class name value for the merged object.

        :return: None

        :raise:

        The method raises exceptions if the class names and metadata are
        different.

        """
        if (class_name is None
                and object_to_merge.get_class_name() != self.__class_name):
            raise ValueError("Objects must have same class name!")
        if class_name is not None:
            self.__class_name = class_name
        if self.__metadata.get_body_position() is None:
            self.__metadata.set_body_position(
                object_to_merge.get_metadata().get_body_position())

        # Merge Bounding boxes
        x_max_1 = self.__box['x_max']
        x_min_1 = self.__box['x_min']
        y_max_1 = self.__box['y_max']
        y_min_1 = self.__box['y_min']

        x_max_2 = object_to_merge.__box['x_max']
        x_min_2 = object_to_merge.__box['x_min']
        y_max_2 = object_to_merge.__box['y_max']
        y_min_2 = object_to_merge.__box['y_min']

        x_min = min(x_min_1, x_min_2)
        x_max = max(x_max_1, x_max_2)
        y_min = min(y_min_1, y_min_2)
        y_max = max(y_max_1, y_max_2)

        # Merge polygons
        frist_polygon = self.get_polygon()
        second_polygon = object_to_merge.get_polygon()
        union_polygon = cascaded_union([frist_polygon, second_polygon])

        self.set_class_name(self.__class_name)
        self.set_metadata(self.__metadata)
        self.set_box(x_min, y_min, x_max, y_max)
        self.set_polygon(union_polygon)
