"""
Class for object filter properties

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""


class FilterParams(object):
    """

    :Description:

    The **FilterParams** class holds filtering properties used while parsing.

    """
    pass

    def __init__(self,
                 depiction=None,
                 cover_by_glass=None,
                 occluded=None,
                 position=None,
                 difficult=None,
                 class_names=[]):
        """

        :Description:

        This is the **Metadata** class constructor.

        :param depiction:
        :param cover_by_glass:
        :param occluded:
        :param position:
        :param difficult:
        :param class_names:list of class names to be included

        :return: A **Metadata** object

        """

        self.__depiction = depiction
        self.__cover_by_glass = cover_by_glass
        self.__occluded = occluded
        self.__position = position
        self.__difficult = difficult
        self.__class_names = class_names


    def get_class_names(self):
        """
        :Description:

        Getter for class_names

        :return list of class names to be included
        """
        return self.__class_names


    def get_depiction(self):
        """

        :Description:

        Getter for depiction.

        :return: the depiction value, as a boolean.

        """
        return self.__depiction


    def get_cover_by_glass(self):
        """

        :Description:

        Getter for cover_by_glass.

        :return: the cover_by_glass value, as a boolean.

        """
        return self.__cover_by_glass


    def get_occluded(self):
        """

        :Description:

        Getter for occluded.

        :return: the occluded value, as a boolean.

        """
        return self.__occluded


    def get_position(self):
        """

        :Description:

        Getter for position.

        :return: the position value, as a string.

        """
        return self.__position


    def get_difficult(self):
        """

        :Description:

        Getter for difficult.

        :return: the difficult value, as an integer.

        """
        return self.__difficult

    @staticmethod
    def is_fulfill(parser_value, metadata_value):
        """
        :Description:

        Check if the filter parameter is fulfill.

        :param parser_value: The condition that must be fulfill.
        :param metadata_value: The value to compare with.

        :return: True if the value is fulfill, False otherwise.

        """
        if parser_value is None:
            return True
        if parser_value == metadata_value:
            return True
        return False
