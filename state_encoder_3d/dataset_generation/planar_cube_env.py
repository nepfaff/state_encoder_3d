import os
from typing import List
import shutil

import numpy as np
from manipulation.utils import AddPackagePaths
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    MeshcatVisualizer,
    Simulator,
    MeshcatVisualizerParams,
    Role,
    MultibodyPlant,
    Parser,
)
import zarr

from .images import ImageGenerator, AddRgbdSensors


def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().AddPackageXml(os.path.abspath("package.xml"))
    return parser


class PlanarCubeEnvironment:
    def __init__(
        self,
        time_step: float,
        scene_directive_path: str,
        num_cameras: int,
        initial_box_position: List[float],
        initial_finger_position: List[float],
        min_pos: float = -1.5,
        max_pos: float = 1.5,
    ):
        self._time_step = time_step
        self._scene_directive_path = scene_directive_path
        self._num_cameras = num_cameras
        self._initial_box_position = initial_box_position
        self._initial_finger_position = initial_finger_position
        self._min_pos = min_pos
        self._max_pos = max_pos

        self._setup()

    def _set_env_state(self, finger_pos: List[float], box_pos: List[float]) -> None:
        # Set box position
        context = self._simulator.get_mutable_context()
        plant_context = self._plant.GetMyMutableContextFromRoot(context)
        box = self._plant.GetBodyByName("box")
        box_model_instance = box.model_instance()
        if box.is_floating():
            self._plant.SetFreeBodyPose(
                plant_context,
                box,
                RigidTransform([*box_pos, 0.0]),
            )
        else:
            self._plant.SetPositions(plant_context, box_model_instance, box_pos)

        # Set finger position
        sphere = self._plant.GetBodyByName("sphere")
        sphere_model_instance = sphere.model_instance()
        if sphere.is_floating():
            self._plant.SetFreeBodyPose(
                plant_context,
                sphere,
                RigidTransform([*finger_pos, 0.0]),
            )
        else:
            self._plant.SetPositions(plant_context, sphere_model_instance, finger_pos)

    def _setup(self) -> None:
        # self._meshcat = StartMeshcat()

        # Setup environment
        builder = DiagramBuilder()
        self._plant, self._scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._time_step
        )
        parser = get_parser(self._plant)
        parser.AddAllModelsFromFile(self._scene_directive_path)
        self._plant.Finalize()

        AddRgbdSensors(builder, self._plant, self._scene_graph)

        # visualizer_params = MeshcatVisualizerParams()
        # visualizer_params.role = Role.kIllustration
        # self._visualizer = MeshcatVisualizer.AddToBuilder(
        #     builder,
        #     self._scene_graph,
        #     self._meshcat,
        #     visualizer_params,
        # )

        diagram = builder.Build()
        self._simulator = Simulator(diagram)

        # Set up image generator
        self._image_generator = ImageGenerator(
            max_depth_range=10.0, diagram=diagram, scene_graph=self._scene_graph
        )

        self._set_env_state(
            finger_pos=self._initial_finger_position, box_pos=self._initial_box_position
        )
        if not self._is_env_state_feasible():
            raise RuntimeError("Initial env state is infeasible.")

    def _is_env_state_feasible(self) -> bool:
        """Returns false if the finger is inside the box and true otherwise."""
        context = self._simulator.get_mutable_context()
        scene_graph_context = self._scene_graph.GetMyMutableContextFromRoot(context)
        query_object = self._scene_graph.get_query_output_port().Eval(
            scene_graph_context
        )
        inspector = query_object.inspector()
        box_frame_id = self._plant.GetBodyFrameIdOrThrow(
            self._plant.GetBodyByName("box").index()
        )
        sphere_frame_id = self._plant.GetBodyFrameIdOrThrow(
            self._plant.GetBodyByName("sphere").index()
        )
        box_geometry_id = inspector.GetGeometries(box_frame_id, Role.kProximity)[0]
        sphere_geometry_id = inspector.GetGeometries(sphere_frame_id, Role.kProximity)[
            0
        ]
        distance = query_object.ComputeSignedDistancePairClosestPoints(
            box_geometry_id, sphere_geometry_id
        ).distance
        return distance >= 0.0

    def _save_dataset(
        self,
        path: str,
        images: np.ndarray,
        finger_positions: np.ndarray,
        box_positions: np.ndarray,
    ) -> None:
        if os.path.exists(path):
            print(
                f"Dataset storage path {path} already exists. Deleting the old dataset."
            )
            shutil.rmtree(path)

        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store)
        image_store = root.zeros_like("images", images)
        image_store[:] = images
        finger_pos_store = root.zeros_like("finger_positions", finger_positions)
        finger_pos_store[:] = finger_positions
        box_pos_store = root.zeros_like("box_positions", box_positions)
        box_pos_store[:] = box_positions

    def generate_sample_dataset(self, dataset_path: str, num_samples: int) -> None:
        finger_positions = []  # Shape (N, 2)
        box_positions = []  # Shape (N, 2)
        images = []  # Shape (N, num_views, W, H, C)
        for _ in range(num_samples):
            # Set random scene state
            finger_pos = (self._max_pos - self._min_pos) * np.random.random_sample(
                2
            ) + self._min_pos
            box_pos = (self._max_pos - self._min_pos) * np.random.random_sample(
                2
            ) + self._min_pos
            self._set_env_state(finger_pos=finger_pos, box_pos=box_pos)

            if not self._is_env_state_feasible():
                continue

            views = []
            for cam_idx in range(self._num_cameras):
                image, _, _, _ = self._image_generator.get_camera_data(
                    camera_name=f"camera{cam_idx}",
                    context=self._simulator.get_context(),
                )
                views.append(image)

            finger_positions.append(finger_pos)
            box_positions.append(box_pos)
            images.append(views)

        finger_positions = np.asarray(finger_positions)
        box_positions = np.asarray(box_positions)
        images = np.asarray(images)

        self._save_dataset(dataset_path, images, finger_positions, box_positions)

    def generate_grid_dataset(self) -> None:
        # TODO: Implement discrete grid search over state-space
        pass
