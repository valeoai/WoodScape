"""
Abstract class for detection object parsers

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""


class OcclusionLevel(object):
    """

    :Description:

    The **OcclusionLevel** class is a container mapping annotated occlusion
    values to Valeo values.

    """

    __mai_occlusion_level = {
        "occluded-0-25": 0,
        "occluded-26-50": 1,
        "occluded-51-75": 2,
        "occluded-76-99": 3
    }
    __scale_ai_occlusion_level = {
        "0-25% (poor)": 3,
        "26-50% (medium)": 2,
        "51-75% (good)": 1,
        "Over 75% (very good)": 0,
        None: 0,
        "": 0
    }
    __next_limit_occlusion_level = {
        False: 0,
        True: 3,
        None: 0,
        "": 0
    }

    @classmethod
    def get_occlusion_level_from_mai(cls, occlusion):
        """

        :Description:

        This is a class method used to retrieve the occlusion level based on MAI
        occlusion value.

        :param cls: Default class method parameter, part of the signature.
        :param occlusion: the occlusion from MAI annotation.

        :return: the occlusion level as an integer

        :raise:

        An exception will be raised if *mai_occlusion* is not supported.

        """
        if occlusion not in cls.__mai_occlusion_level.keys():
            text = 'Invalid occlusion value: %s \n' % occlusion
            text += '\t- Available occlusion values:\n'
            text += '\n;'.join(
                ['\t\t-%s' % occ for occ in cls.__mai_occlusion_level.keys()])
            raise Exception(text)
        return cls.__mai_occlusion_level[occlusion]

    @classmethod
    def get_occlusion_level_from_scale_ai(cls, occlusion):
        """

        :Description:

        This is a class method used to retrieve the occlusion level based on 
        ScaleAI occlusion value.

        :param cls: Default class method parameter, part of the signature.
        :param occlusion: the occlusion from ScaleAI annotation.

        :return: the occlusion level as an integer

        :raise:

        An exception will be raised if *scale_ai_occlusion* is not supported.

        """
        if occlusion not in cls.__scale_ai_occlusion_level.keys():
            text = 'Invalid occlusion value: %s \n' % occlusion
            text += '\t- Available occlusion values:\n'
            text += '\n;'.join(['\t\t-%s' % occ for occ
                                in cls.__scale_ai_occlusion_level.keys()])
            raise Exception(text)
        return cls.__scale_ai_occlusion_level[occlusion]

    @classmethod
    def get_occlusion_level_from_nextlimit(cls, occlusion):
        """

        :Description:

        This is a class method used to retrieve the occlusion level based on 
        NextLimit occlusion value.

        :param cls: Default class method parameter, part of the signature.
        :param occlusion: the occlusion from NextLimit annotation.

        :return: the occlusion level as an integer

        :raise:

        An exception will be raised if *next_limit_occlusion* is not supported.

        """
        if occlusion not in cls.__next_limit_occlusion_level.keys():
            text = 'Invalid occlusion value: %s \n' % occlusion
            text += '\t- Available occlusion values:\n'
            text += '\n;'.join(['\t\t-%s' % occ for occ
                                in cls.__next_limit_occlusion_level.keys()])
            raise Exception(text)
        return cls.__next_limit_occlusion_level[occlusion]
