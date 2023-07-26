# Contains Obstacles class.
# Its attributes are the different sets of obstacles used in the experiment.
# Any number of sets of obstacles can be defined.
# The sets of obstacles are defined as dictionaries.
# Currently only circles, rectangles and bounding boxes are supported.
# Addtionally, functionality can be developed by adding to the refine_cbf/utils.py file.

import numpy as np

class Obstacles():

    # attributes
    # 1st (Initial) set of obstacles
    OBSTACLES_1 = {
        "circle": {
            "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        "rectangle": {
            "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
            "rectangle_2": {"center": np.array([1.5, 1.5]), "length": np.array([0.25, 0.25])},
        },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
        "iteration": 0,
    }
    # 2nd set of obstacles
    OBSTACLES_2  = {
        "circle": {
            "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        "rectangle": {
            "rectangle_1": {"center": np.array([1.5, 0.5]), "length": np.array([0.25, 0.25])},
        },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
        "iteration": 10,
    }

    # 3rd set of obstacles
    OBSTACLES_3 = {
        "circle": {
            "circle_1": {"center": np.array([1.1, 1.0]), "radius": 0.1},
        },
        "bounding_box": {
            "bounding_box_1": {"center": np.array([1.0, 1.0]), "length": np.array([2.0, 2.0])},
        },
        "iteration": 20,
    }

    # Add more obstacles here, if necessary.





    # Methods
    def get_obstacle_list(self):
        # Get all attributes defined in the class
        all_attributes = vars(type(self))

        # Filter the attributes that start with 'OBSTACLES_'
        obstacle_attributes = [value for key, value in all_attributes.items() if key.startswith('OBSTACLES_')]

        return obstacle_attributes
    
    def get_iteration_list(self):
        # Get all obstacle attributes
        obstacle_attributes = self.get_obstacle_list()

        # Extract the "iteration" values from the obstacle dictionaries
        iteration_list = [obstacle["iteration"] for obstacle in obstacle_attributes]

        return iteration_list
        