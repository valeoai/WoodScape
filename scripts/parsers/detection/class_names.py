"""
Instance level class mapping.
Modify this to access more classes for segmentation and detection

# author: carlos pol
# credits: DL team valeo vision systems

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; Authors provide no warranty with the software
and are not liable for anything.

"""


class ClassNames(object):
    """

    :Description:

    The **ClassNames** class is a container mapping class names from MAI with Valeo class names.

    """

    __mai_class_names = {
        "road_surface": "road",
        "curb": "curb",
        "car": "vehicles",
        "train/tram": "vehicles",
        "truck": "vehicles",
        "other_wheeled_transport": "vehicles",
        "trailer": "vehicles",
        "van": "vehicles",
        "caravan": "vehicles",
        "bus": "vehicles",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "person": "person",
        "rider": "rider",
        "grouped_botts_dots": "catseyebottsdots",
        "cats_eyes_and_botts_dots": "catseyebottsdots",
        "parking_marking": "lanemarks",
        "lane_marking": "lanemarks",
        "parking_line": "lanemarks",
        "other_ground_marking": "lanemarks",
        "zebra_crossing": "lanemarks",
        "trafficsign_indistingushable": "void",
        "sky": "void",
        "fence": "void",
        "traffic_light_yellow": "traffic_light",
        "ego_vehicle": "void",
        "pole": "void",
        "structure": "void",
        "traffic_sign": "traffic_sign",
        "animal": "void",
        "free_space": "void",
        "traffic_light_red": "traffic_light",
        "unknown_traffic_light": "traffic_light",
        "movable_object": "void",
        "traffic_light_green": "traffic_light",
        "void": "void",
        "grouped_vehicles": "void",
        "grouped_pedestrian_and_animals": "void",
        "grouped_animals": "void",
        "green_strip": "void",
        "nature": "void",
        "construction": "void",
        "Other_NoSight": "void"}

    __scale_ai_class_names = {
        "road surface": "road",
        "curb": "curb",
        "car": "vehicles",
        "train/tram": "vehicles",
        "truck": "vehicles",
        "other wheeled transport": "vehicles",
        "trailer": "vehicles",
        "van": "vehicles",
        "caravan": "vehicles",
        "bus": "vehicles",
        "tractor": "vehicles",
        "vehicle": "vehicles",
        "personalized wheeled transport": "vehicles",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "person": "person",
        "rider": "rider",
        "vehicle light": "vehicle_light",
        "vehicle plate": "vehicle_plate",
        "wheel_chair": "vehicles",
        "pram/buggy/stroller": "vehicles",
        "shopping cart/trolley": "vehicles",
        "person face": "person_face",
        "grouped cat’s eye and botts’ dots": "catseyebottsdots",
        "cat’s eye and botts’ dots": "catseyebottsdots",
        "parking marking": "lanemarks",
        "lane marking": "lanemarks",
        "parking line": "lanemarks",
        "ground marking": "lanemarks",
        "other ground marking": "lanemarks",
        "zebra crossing": "lanemarks",
        "traffic sign – indistinguishable": "void",
        "sky": "void",
        "fence": "void",
        "traffic light": "void",
        "traffic light – yellow": "void",
        "ego vehicle": "void",
        "pole": "void",
        "structure": "void",
        "traffic sign": "void",
        "animal": "void",
        "free space": "void",
        "traffic light – red": "void",
        "traffic light – unknown": "void",
        "movable objects": "void",
        "traffic light – green": "void",
        "void": "void",
        "grouped vehicles": "void",
        "grouped pedestrians or animals": "void",
        "grouped_animals": "void",
        "green_strip": "void",
        "nature": "void",
        "construction": "void",
        "other nosight": "void"}

    __next_limit_class_names = {
        "people": "person",
        "biker": "rider",
        "cyclist": "rider",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "car": "vehicles",
        "truck": "vehicles",
        "bus": "vehicles",
        "van": "vehicles",
        "baby_cart": "vehicles",
        "walker": "vehicles",
        "crosswalk": "lanemarks",
        "parking area": "lanemarks",
        "road lines": "lanemarks",
        "lane bike": "lanemarks",
        "road": "road",
        "garbage - road": "road",
        "sewer - road": "road",
        "vegetation - road": "road",
        "longitudinal crack": "road",
        "polished aggregated": "road",
        "transversal crack": "road",
        "kerb rising edge": "curb",
        "road lines": "lanemarks",
        "bots": "catseyebottsdots",
        "cat_eye": "catseyebottsdots",
        "air conditioning": "void",
        "alley": "void",
        "barrel": "void",
        "bench": "void",
        "billboard": "void",
        "bin": "void",
        "bird": "void",
        "box": "void",
        "building": "void",
        "concrete_benchs": "void",
        "construction_concrete": "void",
        "construction_container": "void",
        "construction_cord": "void",
        "construction_fence": "void",
        "construction_pallet": "void",
        "construction_post / cone": "void",
        "construction_scaffold": "void",
        "container": "void",
        "dog": "void",
        "electric power": "void",
        "electric_post": "void",
        "electric_post_conductor": "void",
        "electric_post_insulator": "void",
        "electric_post_insulator_break": "void",
        "fences": "void",
        "fire": "void",
        "fire hydrant": "void",
        "garbage": "void",
        "garbage bag": "void",
        "house": "void",
        "jersey_barrier": "void",
        "kerb stone": "void",
        "mailbox": "void",
        "marquees": "void",
        "parking_bicycles": "void",
        "phone booth": "void",
        "pivot": "void",
        "plumbing": "void",
        "portable_bathroom": "void",
        "poster": "void",
        "press box": "void",
        "railings": "void",
        "rock": "void",
        "sewer": "void",
        "sidewalk": "void",
        "sky": "void",
        "stairs": "void",
        "street lights": "void",
        "subway": "void",
        "terrace": "void",
        "terrain": "void",
        "traffic lights bulb rounded green": "void",
        "traffic lights bulb rounded red": "void",
        "traffic lights bulb rounded yellow": "void",
        "traffic lights head": "void",
        "traffic lights poles": "void",
        "traffic signs back": "void",
        "traffic signs poles or structure": "void",
        "traffic signs": "void",
        "traffic_cameras": "void",
        "tram tracks": "void",
        "trash can": "void",
        "umbrellas": "void",
        "vegetation": "void",
        "vending machines": "void",
        "wall": "void",
        "water": "void",
        "yellow_barrel": "void"
    }

    @classmethod
    def get_class_name_from_mai(cls, class_name):
        """

        :Description:

        This is a class method used to retrieve the class name on MAI class names convention.

        :param cls: Default class method parameter, part of the signature.
        :param class_name: the class name from MAI annotation.

        :return: the class name.

        :raise:

        An exception will be raised if *mai_occlusion* is not supported.

        """
        if class_name not in cls.__mai_class_names.keys():
            text = 'Invalid class name: %s \n' % class_name
            text += '\t- Available class names:\n'
            text += '\n;'.join(['\t\t-%s' % name for name in cls.__mai_class_names.keys()])
            raise Exception(text)
        return cls.__mai_class_names[class_name]

    @classmethod
    def get_class_name_from_scale_ai(cls, class_name):
        """

        :Description:

        This is a class method used to retrieve the class name on 
        ScaleAI class names convention.

        :param cls: Default class method parameter, part of the signature.
        :param class_name: the class name from ScaleAI annotation.

        :return: the class name.

        :raise:

        An exception will be raised if *scale_ai_occlusion* is not supported.

        """
        if class_name not in cls.__scale_ai_class_names.keys():
            text = 'Invalid class name: %s \n' % class_name
            text += '\t- Available class names:\n'
            text += '\n;'.join(['\t\t-%s' % name
                                for name in cls.__scale_ai_class_names.keys()])
            raise Exception(text)
        return cls.__scale_ai_class_names[class_name]

    @classmethod
    def get_class_name_from_next_limit(cls, class_name):
        """

        :Description:

        This is a class method used to retrieve the class name on 
        NextLimit class names convention.

        :param cls: Default class method parameter, part of the signature.
        :param class_name: the class name from NextLimit annotation.

        :return: the class name.

        :raise:

        An exception will be raised if *next_limit_occlusion* is not supported.

        """
        if class_name not in cls.__next_limit_class_names.keys():
            text = 'Invalid class name: %s \n' % class_name
            text += '\t- Available class names:\n'
            text += '\n;'.join(['\t\t-%s' % name
                                for name in cls.__next_limit_class_names.keys()])
            raise Exception(text)
        return cls.__next_limit_class_names[class_name]
