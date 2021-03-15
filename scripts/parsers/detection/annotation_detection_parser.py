"""
Abstract class for detection object parsers

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""

import json
import copy

from builtins import super
from shapely.geometry import Polygon

from parsers.detection.class_names import ClassNames
from parsers.detection.occlusion_level import OcclusionLevel
from parsers.detection.detection_object import DetectionObject
from parsers.detection.abstract_detection_parser import AbstractDetectionParser
from parsers.detection.metadata import Metadata


class AnnotationDetectionParser(AbstractDetectionParser):
    """

    :Description:

    The **AnnotationDetectionParser** class holds properties that are read by the
    annotation readers from the MightyAI format annotation files.

    - annotation type: detection.

    """

    def __init__(self, *args, **kwargs):
        """

        :Description:

        The **MAIDetectionParser** implementation of
        AbstractDetectionParser for MightyAI detection parser.

        """
        super(AnnotationDetectionParser, self).__init__(*args, **kwargs)

    def _parse(self, ann_path):
        """

        :Description:

        This is the implementation of the *parse* method in
        **AbstractDetectionParser**.

        :param ann_path: Path to the JSON that contains the annotation

        :return: None

        """
        with open(ann_path) as json_file:
            data = json.load(json_file)

        self._image_width = list(data.values())[0]["image_width"]
        self._image_height = list(data.values())[0]["image_height"]
        self._image_channels = list(data.values())[0]["image_channels"]
        scale_x = self._resize[0] / float(self._image_width)
        scale_y = self._resize[1] / float(self._image_height)
        annotations = list(data.values())[0]["annotation"]
        linked_objects = {}
        for annotation in annotations:
            group_id = annotation.get("group_id")

            try:
                mai_object = self.__parse_mai_object(
                    annotation, scale_x, scale_y)
            except Exception as e:
                print('Could not parse annotation: {} {}'.format(ann_path, str(e)))
                mai_object = None

            if not mai_object:
                continue

            if group_id:
                grouped_objects = linked_objects.get(group_id, [])
                grouped_objects.append(mai_object)
                linked_objects[group_id] = grouped_objects
            else:
                self._objects.append(mai_object)

        # merge linked objects
        for _, object_list in linked_objects.items():
            if self.__is_same_object(object_list):
                # general case: any parts of any split object
                self.__merge_append_object(object_list)
            elif self.__is_rider(object_list, "bicycle"):
                # bike rider handling
                merged_object_list = self.__merge_bicycle_motorcycle_rider(
                    object_list, "bicycle")
                self._objects += merged_object_list
            elif self.__is_rider(object_list, "motorcycle"):
                # motorcycle rider handling
                merged_object_list = self.__merge_bicycle_motorcycle_rider(
                    object_list, "motorcycle")
                self._objects += merged_object_list
            elif self.__is_rider(object_list, "vehicles"):
                # vehicle rider handling
                merged_object = self.__merge_vehicle_rider(object_list)
                self._objects.append(merged_object)
            else:
                # unkown object groups -> add boxes as they are
                self._objects += object_list

        # self._objects = MAIDetectionParser.__rename_static_2_wheeled_vehicles(
        #     self._objects)

    def __merge_append_object(self, object_list):
        """

        :Description:

        Merge a list of objects into a new object and append it to the object
        list. The name will be taken from the first object in the list

        :param object_list: the list of grouped objects.

        :return: None

        """
        aux_object = object_list[0]
        obj_class_name = aux_object.get_class_name()
        for mai_object in object_list[1:]:
            aux_object.merge(mai_object, obj_class_name)
        self._objects.append(aux_object)

    @staticmethod
    def __merge_object(object_list, class_name=None, copy_object=False):
        """

        :Description:

        Merge a list of objects into a new object and return

        :param object_list: the list of grouped objects.

        :param class_name: the class name to be used.

        :param copy_object: Copy object to perform 2 different merges on it.

        :return: merged object

        """
        if copy_object:
            aux_object = copy.deepcopy(object_list[0])
        else:
            aux_object = object_list[0]

        for mai_object in object_list[1:]:
            aux_object.merge(mai_object, class_name)
        # make sure object class is renamed when only 1 object is passed
        if class_name is not None:
            aux_object.set_class_name(class_name)
        return aux_object

    @staticmethod
    def __is_same_object(object_list):
        """

        :Description:

        Check if the objects form the list belong to the same object class.

        :param object_list: the list of grouped objects.

        :return: True if the objects belong to the same object class, False
        otherwise.

        """
        aux_object = object_list[0]
        for mai_object in object_list[1:]:
            if mai_object.get_class_name() != aux_object.get_class_name():
                return False
        return True

    @staticmethod
    def __is_rider(object_list, rider_type):
        """

        :Description:

        Check if the objects from the list is a rider (vehicle,

        :param object_list: the list of grouped objects.
        :param rider_type: String from rider type: vehicle, bicycle, motorcycle

        :return: True if the objects are a rider, False
        otherwise.

        """
        class_list = [x.get_class_name() for x in object_list]
        rider = "rider" in class_list
        person = "person" in class_list
        vehicle = rider_type in class_list
        return (rider or person) and vehicle

    @staticmethod
    def __merge_bicycle_motorcycle_rider(object_list, rider_type):
        """

        :Description:

        Handle the merge of motorcylce/bicycle riders into 3 boxes:
        - person
        - bicyclerider/motorcyclerider
        - 2wheeledvehicle

        :param object_list: the list of grouped objects.
        :param rider_type: string of rider type (bicycle, motorcycle)

        :return: list of objects to be added to final group

        """
        # merge all parts of person and bike (if splitted)
        all_person_parts = []
        all_bike_parts = []
        for x in object_list:
            if x.get_class_name() in ["rider", "person"]:
                all_person_parts.append(x)
            if rider_type == x.get_class_name():
                all_bike_parts.append(x)
        person_merged = AnnotationDetectionParser.__merge_object(all_person_parts,
                                                                 "person")
        bike_merged = AnnotationDetectionParser.__merge_object(all_bike_parts,
                                                               "2wheeledvehicle")
        # copy object as this is an additional fusion of person and bike
        rider_merged = AnnotationDetectionParser.__merge_object(object_list,
                                                                "rider",
                                                                copy_object=True)
        return [person_merged, bike_merged, rider_merged]

    @staticmethod
    def __merge_vehicle_rider(object_list):
        """

        :Description:

        Handle the merge of vehicle riders into:
        - vehicle
        (remove person as this is usually non detectable part of humans)

        :param object_list: the list of grouped objects.

        :return: list of objects to be added to final group

        """
        # merge all parts of person and bike (if split)
        all_vehicle_parts = [x for x in object_list if
                             "vehicles" == x.get_class_name()]
        vehicle_merged = AnnotationDetectionParser.__merge_object(all_vehicle_parts,
                                                                  "vehicles")
        return vehicle_merged

    @staticmethod
    def __rename_static_2_wheeled_vehicles(object_list):
        """

        :Description:

        Handle renaming of static bikes and motorcycles that have not yet been
        merged with a rider to 2wheeledvehicles.

        :param object_list: the list of grouped objects.

        :return: list of objects

        """
        object_list = AnnotationDetectionParser.__rename_objects_by_name_filter(
            object_list, "bicycle", "vehicles")

        object_list = AnnotationDetectionParser.__rename_objects_by_name_filter(
            object_list, "motorcycle", "vehicles")
        return object_list

    @staticmethod
    def __rename_objects_by_name_filter(object_list, name_search,
                                        rename_value):
        """

        :Description:

        Rename objects queried by filter name

        :param object_list: the list of grouped objects.
        :param name_search: name string to be searched
        :param rename_value: name to be replaced for matching objects

        :return: list of objects

        """
        for x in object_list:
            if name_search == x.get_class_name():
                x.set_class_name(rename_value)
        return object_list

    def __scale_points(self, point_2d, scale_x, scale_y):
        """
        Scale points in a polygon
        """
        x_new = max(min(point_2d[0] * scale_x, self._resize[0]), 0)
        y_new = max(min(point_2d[1] * scale_y, self._resize[1]), 0)

        return x_new, y_new

    def __parse_mai_object(self, annotation, scale_x, scale_y):
        """

        :Description: parse MightyAI object

        :param annotation: JSON with the object data
        :param scale_x: scale x value
        :param scale_y: scale y value

        :return: A **DetectionObject** object.

        """
        class_name = ClassNames.get_class_name_from_mai(annotation["tags"][0])
        points = [self.__scale_points(
            (point[0], point[1]), scale_x, scale_y) for point in annotation["segmentation"]]

        polygon = Polygon(points)
        x_min, y_min, x_max, y_max = polygon.bounds

        box = {'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max}

        metadata = Metadata()
        if 'states' in annotation.keys():
            for key, value in annotation["states"].items():
                if "depiction" in key:
                    metadata.set_depiction(value["id"].split("-")[-1] == "yes")
                elif "glass" in key:
                    metadata.set_cover_by_glass(
                        value["id"].split("-")[-1] == "yes")
                elif "occlusion" in key:
                    metadata.set_occlusion(
                        OcclusionLevel.get_occlusion_level_from_mai(value["id"]))
                elif "position" in key:
                    metadata.set_body_position(value["id"].replace("-", "_"))

        return DetectionObject(class_name, box, metadata, polygon)
