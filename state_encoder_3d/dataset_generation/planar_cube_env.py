import os
from typing import List

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
    ):
        self._time_step = time_step
        self._scene_directive_path = scene_directive_path
        self._num_cameras = num_cameras
        self._initial_box_position = initial_box_position
        self._initial_finger_position = initial_finger_position
        self._meshcat = None
        self._simulator = None

        self._setup()

    def _setup(self) -> None:
        self._meshcat = StartMeshcat()

        # Setup environment
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._time_step
        )
        parser = get_parser(plant)
        parser.AddAllModelsFromFile(self._scene_directive_path)
        plant.Finalize()

        AddRgbdSensors(builder, plant, scene_graph)

        visualizer_params = MeshcatVisualizerParams()
        visualizer_params.role = Role.kIllustration
        self._visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            self._meshcat,
            visualizer_params,
        )

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        # Set initial cube position
        context = self._simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box = plant.GetBodyByName("box")
        box_model_instance = box.model_instance()
        if box.is_floating():
            plant.SetFreeBodyPose(
                plant_context,
                box,
                RigidTransform([*self._initial_box_position, 0.0]),
            )
        else:
            plant.SetPositions(
                plant_context, box_model_instance, self._initial_box_position
            )

        # Set initial finger position
        sphere = plant.GetBodyByName("sphere")
        sphere_model_instance = sphere.model_instance()
        if sphere.is_floating():
            plant.SetFreeBodyPose(
                plant_context,
                sphere,
                RigidTransform([*self._initial_finger_position, 0.0]),
            )
        else:
            plant.SetPositions(
                plant_context, sphere_model_instance, self._initial_finger_position
            )

        # Set up image generator
        self._image_generator = ImageGenerator(
            max_depth_range=10.0, diagram=diagram, scene_graph=scene_graph
        )

        # Test image generator
        rgb_image, _, _, _ = self._image_generator.get_camera_data(
            camera_name="camera0", context=self._simulator.get_context()
        )
        from PIL import Image

        im = Image.fromarray(rgb_image)
        im.save("test.png")

    def simulate(self) -> None:
        print("Press 'Stop Simulation' in MeshCat to continue.")

        self._visualizer.StartRecording()

        print(f"Meshcat URL: {self._meshcat.web_url()}")

        self._simulator.AdvanceTo(1.0)

        self._visualizer.StopRecording()
        self._visualizer.PublishRecording()
