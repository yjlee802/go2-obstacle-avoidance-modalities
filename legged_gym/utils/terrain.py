import glob
import os
import os.path as osp
from itertools import permutations

import cv2
import imageio
import numpy as np
from numpy.random import choice
from scipy import interpolate

from legged_gym.utils import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import tqdm

OBS_DIST_THRESH = 0.5
EUCLID_THRESH = 5.0


def to_shape(a, shape):
    y_, x_ = shape
    y, x = a.shape
    y_pad = y_ - y
    x_pad = x_ - x
    return np.pad(
        a,
        ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
        mode="constant",
    )

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
         # === [NEW] 전방(x) 컷오프: x_cut(m) 이후를 평탄화(장애물 제거) ===
        x_cut_m = getattr(self.cfg, "obstacle_x_cut_m", None)
        if x_cut_m is not None:
            s = float(self.cfg.horizontal_scale)
            x_cut_idx = int(self.border + (float(x_cut_m) / s))
            x_cut_idx = max(0, min(x_cut_idx, self.tot_rows))
            self.height_field_raw[x_cut_idx:, :] = 0
            self.heightsamples = self.height_field_raw

            # heightsamples도 갱신
            self.heightsamples = self.height_field_raw

            # (선택) 1회 디버그
            if not hasattr(self, "_dbg_ycut"):
                print(f"[TERRAIN] obstacle_x_cut_m={x_cut_m}  -> col idx={x_cut_idx}/{self.tot_cols}")
                self._dbg_ycut = True
        # === [NEW] 끝 ===

        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    def block_terrain(self, num_blocks):
        terrain = np.zeros((900, 900))
        rng = np.random.default_rng(12345)
        h, w = terrain.shape
        xs = rng.integers(low=0, high=h, size=num_blocks)
        ys = rng.integers(low=0, high=w, size=num_blocks)

        for i in range(len(xs)):
            width = np.random.choice([2, 3, 4, 5]) * 3
            x, y = xs[i], ys[i]
            if np.random.choice([0, 1]) == 1:
                terrain[x:x+width, y:y+3] = 25
            else:
                terrain[x:x+3, y:y+width] = 25

        return terrain

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self, diff=None, obs_scale=1):
        num_cols = (
            self.cfg.tot_cols if hasattr(self.cfg, "tot_cols") else self.cfg.num_cols
        )
        num_rows = (
            self.cfg.tot_rows if hasattr(self.cfg, "tot_rows") else self.cfg.num_rows
        )
        for j in range(num_cols):
            for i in range(num_rows):
                difficulty = i / self.cfg.num_rows if diff is None else diff
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty, obs_scale)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels, # play.py 시 지형이 train.py 와 다르게 나오는 문제 수정용
                            # length=self.width_per_env_pixels, # 원래 이거였음
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty, obs_scale=1):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels, # play.py 시 지형이 train.py 와 다르게 나오는 문제 수정용
                                # length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.0)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.0)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.0)
        
        
        # elif choice < self.proportions[5]:
        #     # print("MAKING TERRAIN HERE")
        #     num_rectangles = int(200 * difficulty)
        #     # num_rectangles = 0
        #     rectangle_min_size = 2 * obs_scale
        #     rectangle_max_size = 5 * obs_scale
        #     terrain_utils.discrete_obstacles_terrain_cells(
        #         terrain,
        #         # float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"]),
        #         # float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"]),
        #         0.14,
        #         0.15,
        #         rectangle_min_size,
        #         rectangle_max_size,
        #         num_rectangles,
        #         platform_size=3.0,
        #         width = 2 * obs_scale
        #     )
        
        # The core codes of terrain for our project (random boxes) 
        elif choice < self.proportions[5]:
            # print("MAKING TERRAIN HERE")
            num_rectangles = int(300 * difficulty) # was 200
            # num_rectangles = 0
            rectangle_min_size = 2 * obs_scale
            rectangle_max_size = 5 * obs_scale
            terrain_utils.discrete_obstacles_terrain_cells(
                terrain,
                # float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"]),
                # float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"]),
                0.70, # 0.50
                1.00, # 0.70
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
                width = 2 * obs_scale
            )
        elif choice < self.proportions[6]:
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=4.0,
            )
        elif choice < self.proportions[7]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.0)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length / 2.0 - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length / 2.0 + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
    def add_blocks(self):
        BLOCKS_PER_AREA = 1.0
        DIST_THRESH = 0.75
        SPAWN_OBS_THRESH = 1.5
        POTENTIAL_DIMS = [(0.15, 0.15), (0.15, 0.3), (0.3, 0.15)]
        min_block_height = float(os.environ["ISAAC_BLOCK_MIN_HEIGHT"])
        max_block_height = float(os.environ["ISAAC_BLOCK_MAX_HEIGHT"])

        x0, x1, y0, y1 = self.get_terrain_bounds()
        area = (x1 - x0) * (y1 - y0)
        num_blocks = int(area * BLOCKS_PER_AREA)
        # A block is an x, y, s1, s2, and h
        blocks = []
        np.random.seed(int(os.environ["ISAAC_SEED"]))
        print(f"Generating {num_blocks} obstacles..")
        for _ in tqdm.trange(num_blocks):
            success = False
            while not success:
                s1, s2 = POTENTIAL_DIMS[np.random.randint(3)]
                x = np.random.rand() * (x1 - x0) + x0
                y = np.random.rand() * (y1 - y0) + y0
                if (
                    np.linalg.norm(np.array([x, y]) - self.terrain_start)
                    < SPAWN_OBS_THRESH
                ):
                    continue
                if (
                    np.linalg.norm(np.array([x, y]) - self.terrain_goal)
                    < SPAWN_OBS_THRESH
                ):
                    continue
                block_height = np.random.uniform(min_block_height, max_block_height)
                new_block = (x, y, s1, s2, block_height)
                if blocks:
                    blocks_arr = np.array(blocks)[:, :2]
                    new_block_arr = np.array(new_block)[:2]
                    diff = blocks_arr - new_block_arr
                    if min(np.linalg.norm(diff, axis=1)) < DIST_THRESH:
                        continue
                blocks.append(new_block)
                success = True

        for block in blocks:
            self.add_block(*block)

    def add_block(self, x0, y0, s1, s2, h):
        # A rectangular prism has 8 vertices
        new_vertices = [
            (x0, y0, 0.0),
            (x0 + s1, y0, 0.0),
            (x0, y0 + s2, 0.0),
            (x0 + s1, y0 + s2, 0.0),
            (x0, y0, h),
            (x0 + s1, y0, h),
            (x0, y0 + s2, h),
            (x0 + s1, y0 + s2, h),
        ]
        # Spam every possible combination
        new_triangles = list(permutations(range(8), 3))
        self.triangles = np.concatenate(
            [
                self.triangles,
                np.array(new_triangles, dtype=np.uint32) + self.vertices.shape[0],
            ]
        )
        self.vertices = np.concatenate(
            [self.vertices, np.array(new_vertices, dtype=np.float32)]
        )

def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    
def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min