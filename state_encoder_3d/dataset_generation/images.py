import copy
from typing import Tuple, List

import numpy as np
from pydrake.all import (
    Diagram,
    SceneGraph,
    Context,
)


class ImageGenerator:
    def __init__(
        self, max_depth_range: float, diagram: Diagram, scene_graph: SceneGraph
    ) -> None:
        self._diagram = diagram
        self._max_depth_range = max_depth_range
        self._scene_graph = scene_graph

    def get_camera_data(
        self, camera_name: str, context: Context
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        # Need to make a copy as the original value changes with the simulation
        rgba_image = copy.deepcopy(
            self._diagram.GetOutputPort(f"{camera_name}_rgb_image").Eval(context).data
        )
        rgb_image = rgba_image[:, :, :3]

        depth_image = copy.deepcopy(
            self._diagram.GetOutputPort(f"{camera_name}_depth_image")
            .Eval(context)
            .data.squeeze()
        )
        depth_image[depth_image == np.inf] = self._max_depth_range

        label_image = copy.deepcopy(
            self._diagram.GetOutputPort(f"{camera_name}_label_image")
            .Eval(context)
            .data.squeeze()
        )
        object_labels = np.unique(label_image)
        masks = [
            np.uint8(np.where(label_image == label, 255, 0)) for label in object_labels
        ]

        return rgb_image, depth_image, object_labels, masks
