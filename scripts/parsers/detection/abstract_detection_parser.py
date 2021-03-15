"""
Abstract class for detection object parsers

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""


from abc import abstractmethod
from parsers.detection.filter_params import FilterParams


class AbstractDetectionParser(object):
    """"

    :Description:

    The **AbstractDetectionParser** class holds the logic needed to parse
    annotations models.
    It requires implementations based on different scenarios:

    - PascalVOC annotation
    - MightyAI annotation

    """

    def __init__(self,
                 ann_path,
                 resize,
                 filter_params=FilterParams(),
                 image=None):
        """

        :Description:

        This is the **AbstractDetectionParser** class constructor.

        :param ann_path: the path to the annotation file
        :param resize: the resize values as a tuple
        :param filter_params: the filter params.

        :return:

        """
        self._image_height = 0
        self._image_width = 0
        self._image_channels = 0
        self._resize = resize
        self._objects = []

        self.__filter_params = filter_params
        self._image = image
        self._parse(ann_path)

        if filter_params:
            self._objects = [x for x in self._objects if self.__is_valid_object(x)]

    def get_image_height(self):
        """

        :return:
        """
        return self._image_height

    def get_image_width(self):
        """

        :return:

        """
        return self._image_width

    def get_image_channels(self):
        """
        :return:
        """
        return self._image_channels

    def get_objects(self):
        """
        :return:
        """
        return self._objects

    @abstractmethod
    def _parse(self, *args, **kwargs):
        """

        :Description:

        Method to be implemented by their children.

        :param args:
        :param kwargs:
        :return:

        :raise:

        """
        raise Exception('Method not implemented!')

    def __is_valid_object(self, detection_object):
        """

        :Description:

        Check if the object fulfill the filter requirements.

        :param detection_object: DetectionObject to check

        :return: true if the object fulfill the filter requirements, false
        otherwise

        """
        y_min = detection_object.get_box().get("y_min")
        y_max = detection_object.get_box().get("y_max")
        x_min = detection_object.get_box().get("x_min")
        x_max = detection_object.get_box().get("x_max")
        height = y_max - y_min
        width = x_max - x_min

        # filter annotation errors (more filtering is done after augmentation)
        if height <= 0 or width <= 0:
            return False

        if detection_object.get_class_name() \
            not in self.__filter_params.get_class_names():
            return False

        metadata = detection_object.get_metadata()
        if not self.__filter_params.get_depiction() and metadata.is_depiction():
            return False
        if not self.__filter_params.get_cover_by_glass() \
            and metadata.is_cover_by_glass():
            return False
        if not FilterParams.is_fulfill(self.__filter_params.get_occluded(),
                                       metadata.get_occlusion()):
            return False
        if not FilterParams.is_fulfill(self.__filter_params.get_position(),
                                       metadata.get_body_position()):
            return False
        if not FilterParams.is_fulfill(self.__filter_params.get_difficult(),
                                       metadata.get_difficult()):
            return False
        return True
