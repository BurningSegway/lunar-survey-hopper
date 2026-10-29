import math
from pathlib import Path
import unittest

import mujoco
import numpy as np

from jumper_sim.ray_casting import HeightfieldRayCaster, MeshRayCaster
from jumper_sim.gpu_ray_caster import WarpRayCaster, gpu_available
from jumper_sim.lidar import LIVOX_MID360

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"


@unittest.skipUnless(MODEL_PATH.is_file(), "needs the robot model")
@unittest.skipUnless(gpu_available(), "needs an NVIDIA GPU (Warp)")
class WarpRayCasterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.terrain = HeightfieldRayCaster(self.model)
        self.gpu = WarpRayCaster(self.terrain)
        # 1 s of the Mid-360 from 6 m up, tilted 45 deg: lots of grazing rays.
        tilt = math.radians(45.0)
        rotation = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)],
                             [0, math.sin(tilt), math.cos(tilt)]])
        self.origin = np.array([0.0, 0.0, 6.0])
        self.rays = LIVOX_MID360.directions(0, 200000) @ (rotation @ np.diag([1.0, -1.0, -1.0])).T

    def test_matches_the_numpy_caster_on_the_terrain(self) -> None:
        cpu = self.terrain.cast(self.origin, self.rays, 40.0)
        gpu = self.gpu.cast(self.origin, self.rays, 40.0)
        self.assertGreater(np.isfinite(cpu).sum(), 50000)
        # Same hits; float32 on the GPU, so distances agree to ~0.1 mm.
        disagree = np.isfinite(cpu) != np.isfinite(gpu)
        self.assertLess(disagree.sum(), 50)  # rays grazing a triangle edge
        both = np.isfinite(cpu) & np.isfinite(gpu)
        self.assertLess(float(np.max(np.abs(cpu[both] - gpu[both]))), 1e-3)

    def test_follows_objects_placed_on_the_terrain(self) -> None:
        geom = self.model.geom("target_object").id
        # Stand it where a ray ~8 m long meets the ground.
        ground = self.terrain.cast(self.origin, self.rays, 40.0)
        spot = self.origin + self.rays[np.nanargmin(np.abs(ground - 8.0))] * 8.0
        x, y = spot[:2]
        self.model.geom_pos[geom] = (x, y, float(self.terrain.height(np.array([x]), np.array([y]))[0]))
        before = self.gpu.cast(self.origin, self.rays, 40.0)
        self.terrain.objects = [MeshRayCaster.from_model(self.model, geom)]
        cpu = self.terrain.cast(self.origin, self.rays, 40.0)
        gpu = self.gpu.cast(self.origin, self.rays, 40.0)
        on_object = np.isfinite(cpu) & (cpu < before - 0.01)
        self.assertGreater(on_object.sum(), 10)
        np.testing.assert_allclose(gpu[on_object], cpu[on_object], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
