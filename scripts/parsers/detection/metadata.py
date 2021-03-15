"""
Class for detection objects meta data

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""


class Metadata(object):
    """

    :Description:

    The **Metadata** class holds properties that represent the state of an object.

    """

    def __init__(self):
        """

        :Description:

        This is the **Metadata** class constructor.

        :return: A **Metadata** object

        """
        self.__depiction = False
        self.__cover_by_glass = False
        self.__occlusion = 0
        self.__body_position = None
        self.__difficult = 0

    def is_depiction(self):
        """

        :Description:

        Getter for depiction.

        :return: True if the object is depicted, False otherwise.

        """
        return self.__depiction

    def set_depiction(self, depiction):
        """

        :Description:

        Setter for depiction.

        :param depiction: The depiction to set.

        :return: None

        """
        self.__depiction = depiction

    def is_cover_by_glass(self):
        """

        :Description:

        Getter for cover_by_glass.

        :return: True if the object is covered by glass, False otherwise.

        """
        return self.__cover_by_glass

    def set_cover_by_glass(self, cover_by_glass):
        """

        :Description:

        Setter for cover_by_glass.

        :param cover_by_glass: the cover by glass value to set.

        :return: None.

        """
        self.__cover_by_glass = cover_by_glass

    def get_occlusion(self):
        """

        :Description:

        Getter for occlusion.

        :return: The Occlusion of the object, as an integer.

        """
        return self.__occlusion

    def set_occlusion(self, occlusion):
        """

        :Description:

        Setter for occlusion.

        :param occlusion: The occlusion value to set.

        :return: None.

        """
        self.__occlusion = occlusion

    def get_body_position(self):
        """

        :Description:

        Getter for body position.

        :return: The body position of the object, as a string.

        """
        return self.__body_position

    def set_body_position(self, body_position):
        """

        :Description:

        Setter for body position.

        :param body_position: The body position to set.

        :return: None.

        """
        self.__body_position = body_position

    def get_difficult(self):
        """

        :Description:

        Getter for difficult.

        :return: The difficult of the object, as an integer.

        """
        return self.__difficult

    def set_difficult(self, difficult):
        """

        :Description:

        Setter for difficult.

        :param difficult: The difficult to set.

        :return: None.

        """
        self.__difficult = difficult
