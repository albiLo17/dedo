"""
DeformEnv class is the core class for loading and running various tasks.


Note: this code is for research i.e. quick experimentation; it has minimal
comments for now, but if we see further interest from the community -- we will
add further comments, unify the style, improve efficiency and add unittests.

@contactrika, @yonkshi

"""

import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, load_deformable, 
    load_floor, get_preset_properties)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import (
    DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO,
    TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION)
from ..utils.procedural_utils import (
    gen_procedural_hang_cloth, gen_procedural_hang_cloth_real,
    gen_procedural_button_cloth)
from ..utils.args import preset_override_util
from ..utils.process_camera import ProcessCamera, cameraConfig


class DeformEnv(gym.Env):
    MAX_OBS_VEL = 20.0  # max vel (in m/s) for the anchor observations
    MAX_ACT_VEL = 10.0  # max vel (in m/s) for the anchor actions
    WORKSPACE_BOX_SIZE = 20.0  # workspace box limits (needs to be >=1)
    STEPS_AFTER_DONE = 500     # steps after releasing anchors at the end
    FORCE_REWARD_MULT = 1e-4   # scaling for the force penalties
    FINAL_REWARD_MULT = 400    # multiply the final reward (for sparse rewards)
    SUCESS_REWARD_TRESHOLD = 2.5  # approx. threshold for task success/failure

    def __init__(self, args):
        self.args = args
        self.cam_on = args.cam_resolution > 0
        # Initialize sim and load objects.
        self.sim = bclient.BulletClient(
            connection_mode=pybullet.GUI if args.viz else pybullet.DIRECT)
        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        reset_bullet(self.args, self.sim, debug=args.debug)

        # reset_bullet(args, self.sim, debug=args.debug)
        self.food_packing = self.args.env.startswith('FoodPacking')
        self.num_anchors = 1 if self.food_packing else 2
        res = self.load_objects(self.sim, self.args, debug=True)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = res

        # Step 3: Load floor
        load_floor(self.sim, debug=args.debug)

        self.max_episode_len = self.args.max_episode_len
        # Define sizes of observation and action spaces.
        self.gripper_lims = np.tile(np.concatenate(
            [self.WORKSPACE_BOX_SIZE * np.ones(3),  # 3D pos
             np.ones(3)]), self.num_anchors)         # 3D linvel/MAX_OBS_VEL
        if args.cam_resolution <= 0:  # report gripper positions as low-dim obs
            self.observation_space = gym.spaces.Box(
                -1.0 * self.gripper_lims, self.gripper_lims)
        else:  # RGB WxHxC
            shape = (args.cam_resolution, args.cam_resolution, 3)
            if args.flat_obs:
                shape = (np.prod(shape),)
            self.observation_space = gym.spaces.Box(
                low=0, high=255 if args.uint8_pixels else 1.0,
                dtype=np.uint8 if args.uint8_pixels else np.float16,
                shape=shape)
        act_sz = 3  # 3D linear velocity for anchors
        self.action_space = gym.spaces.Box(  # [-1, 1]
            -1.0 * np.ones(self.num_anchors * act_sz),
            np.ones(self.num_anchors * act_sz))
        if self.args.debug:
            print('Created DeformEnv with obs', self.observation_space.shape,
                  'act', self.action_space.shape)

        # Optional: render frames during make_final_steps (the post-policy
        # settle phase) so external video loggers can show the cloth falling.
        # Toggled by an external callback before/after each eval episode;
        # default off to avoid render cost during training.
        self._record_settle_frames = False
        self._settle_render_kwargs = dict(width=300, height=300)
        self._settle_frame_stride = 1  # every Nth recorded sub-step

        # Point cloud observation initilization
        self.pcd_mode = args.pcd
        if args.pcd:
            self.camera_config = cameraConfig.from_file(args.cam_config_path)
            self.object_ids = res[0]
            self.object_ids.append(res[1])

            print(f"Starting object ids: {self.object_ids}")
            print(f"Deformable ID: {res[1]}")

    @staticmethod
    def unscale_vel(act, unscaled):
        if unscaled:
            return act
        return act*DeformEnv.MAX_ACT_VEL

    @property
    def anchor_ids(self):
        return list(self.anchors.keys())

    @property
    def _cam_viewmat(self):
        dist, pitch, yaw, pos_x, pos_y, pos_z = self.args.cam_viewmat
        cam = {
            'distance': dist,
            'pitch': pitch,
            'yaw': yaw,
            'cameraTargetPosition': [pos_x, pos_y, pos_z],
            'upAxisIndex': 2,
            'roll': 0,
        }
        view_mat = self.sim.computeViewMatrixFromYawPitchRoll(**cam)
        return view_mat

    def get_texture_path(self, file_path):
        # Get either pre-specified texture file or a random one.
        if self.args.use_random_textures:
            parent = os.path.dirname(file_path)
            full_parent_path = os.path.join(self.args.data_path, parent)
            randfile = np.random.choice(list(os.listdir(full_parent_path)))
            file_path = os.path.join(parent, randfile)
        return file_path

    def load_objects(self, sim, args, debug):
        scene_name = self.args.task.lower()
        if scene_name in ['hanggarment', 'bgarments', 'sewing', 'hangproccloth']:
           scene_name = 'hangcloth'  # same hanger for garments and cloths
        elif scene_name == 'hangprocclothreal':
           scene_name = 'hangcloth_real'
        elif scene_name.startswith('button'):
            scene_name = 'button'
        elif scene_name.startswith('dress'):
            scene_name = 'dress'  # same human figure for dress and mask tasks

        # Make v0 the random version
        if args.version == 0:
            args.use_random_textures = True

        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        sim.setAdditionalSearchPath(data_path)

        # Adjust info in DEFORM_INFO and SCENE_INFO for the specific task,
        # then record the appropriate path to deformable into deform_obj.
        if args.override_deform_obj is not None:
            deform_obj = args.override_deform_obj
        elif self.args.task == 'HangProcCloth':  # procedural gen. for hanging
            args.node_density = 15
            if args.version == 0:
                args.num_holes = np.random.randint(2)+1
            elif args.version in [1,2]:
                args.num_holes = args.version
            deform_obj = gen_procedural_hang_cloth(
                self.args, 'procedural_hang_cloth', DEFORM_INFO)

            preset_override_util(args, DEFORM_INFO[deform_obj])
        elif self.args.task == 'HangProcClothReal':  # real-world scale variant
            args.node_density = 15
            if args.version == 0:
                args.num_holes = np.random.randint(2)+1
            elif args.version in [1,2]:
                args.num_holes = args.version
            deform_obj = gen_procedural_hang_cloth_real(
                self.args, 'procedural_hang_cloth_real', DEFORM_INFO)

            preset_override_util(args, DEFORM_INFO[deform_obj])
        elif self.args.task == 'ButtonProc':  # procedural gen. for buttoning
            args.num_holes = 2
            args.node_density = 15
            deform_obj, hole_centers = gen_procedural_button_cloth(
                self.args, 'proc_button_cloth', DEFORM_INFO)
            preset_override_util(args, DEFORM_INFO[deform_obj])

            # Move buttons to match hole position.
            h1, h2 = hole_centers
            h1 = (-h1[1], 0, h1[2]+2)
            h2 = (-h2[1], 0, h2[2]+2)
            buttons = SCENE_INFO['button']
            buttons['entities']['urdf/button_fixed.urdf']['basePosition'] = (
                h1[0], 0.2, h1[2])
            buttons['entities']['urdf/button_fixed2.urdf']['basePosition'] = (
                h2[0], 0.2, h2[2])
            buttons['goal_pos'] = [h1, h2]
        elif self.args.task == 'BGarments':
            if args.version == 0:
                deform_obj = np.random.choice(TASK_INFO[args.task])
            else:
                deform_obj = TASK_INFO[args.task][args.version-1]
            DEFORM_INFO[deform_obj] = DEFORM_INFO['berkeley_garments'].copy()
        elif self.args.task == 'Sewing':
            if args.version == 0:
                deform_obj = np.random.choice(TASK_INFO[args.task])
            else:
                deform_obj = TASK_INFO[args.task][args.version-1]
            DEFORM_INFO[deform_obj] = DEFORM_INFO['sewing_garments'].copy()
        else:
            assert (args.task in TASK_INFO)  # already checked in args
            assert (args.version <= len(TASK_INFO[args.task]))
            if args.version == 0:
                deform_obj = np.random.choice(TASK_INFO[args.task])
                if args.task == 'HangBag':  # select from 100+ obj mesh variants
                    tmp_id0 = np.random.randint(TOTE_MAJOR_VERSIONS)
                    tmp_id1 = np.random.randint(TOTE_VARS_PER_VERSION)
                    deform_obj = f'bags/totes/bag{tmp_id0:d}_{tmp_id1:d}.obj'
                    if deform_obj not in DEFORM_INFO:
                        tmp_key = f'bags/totes/bag{tmp_id0:d}_0.obj'
                        assert (tmp_key in DEFORM_INFO)
                        DEFORM_INFO[deform_obj] = DEFORM_INFO[tmp_key].copy()
            else:
                deform_obj = TASK_INFO[args.task][args.version - 1]


            preset_override_util(args, DEFORM_INFO[deform_obj])
        if deform_obj in DEFORM_INFO:
            preset_override_util(args, DEFORM_INFO[deform_obj])

        # Load deformable object.
        texture_path = args.deform_texture_file
        # Randomize textures for deformables (except YCB food objects).
        if not self.food_packing:
            texture_path = os.path.join(
                data_path, self.get_texture_path(args.deform_texture_file))
        deform_id = load_deform_object(
            sim, deform_obj, texture_path, args.deform_scale,
            args.deform_init_pos, args.deform_init_ori,
            args.deform_bending_stiffness, args.deform_damping_stiffness,
            args.deform_elastic_stiffness, args.deform_friction_coeff,
            not args.disable_self_collision, debug,
            mass=getattr(args, 'deform_mass', 1.0))
        if scene_name == 'button':  # pin cloth edge for buttoning task
            assert ('deform_fixed_anchor_vertex_ids' in DEFORM_INFO[deform_obj])
            pin_fixed(sim, deform_id,
                      DEFORM_INFO[deform_obj]['deform_fixed_anchor_vertex_ids'])

        # HangProcCloth-only: sample a per-episode (dx, dy) shift for the
        # hanger goal so the cloth has to thread a different peg location
        # each reset. The shift is applied to BOTH rigid bodies (hanger
        # cross-piece AND tallrod vertical support, so the peg structure
        # stays physically coherent) AND to the goal_pos (so the reward
        # function, the scripted controller, and obs['goal'] all retarget
        # automatically — every downstream consumer already reads
        # self.goal_pos[0]). Uses np.random, so a prior env.seed() call
        # makes the dxy reproducible across collection/eval.
        # The z component is sampled separately (--randomize_goal_dz) because
        # peg HEIGHT has a different physical meaning and a much tighter safe
        # range than lateral placement: it changes how far the gripper has to
        # lift, so it stops the policy learning a fixed lift amplitude.
        goal_dxy = np.zeros(2, dtype=np.float32)
        goal_dz = np.float32(0.0)
        randomize_r = float(getattr(args, 'randomize_goal_radius', 0.0))
        randomize_dz = float(getattr(args, 'randomize_goal_dz', 0.0))
        _is_hangcloth_scene = scene_name in ('hangcloth', 'hangcloth_real')
        if _is_hangcloth_scene and randomize_r > 0.0:
            goal_dxy = np.random.uniform(
                -randomize_r, randomize_r, size=2).astype(np.float32)
        if _is_hangcloth_scene and randomize_dz > 0.0:
            goal_dz = np.float32(np.random.uniform(-randomize_dz, randomize_dz))
        self._last_goal_dxy = goal_dxy
        self._last_goal_dz = goal_dz
        # Single delta applied to every peg body and to goal_pos, so the peg
        # structure and all downstream consumers stay consistent.
        goal_delta = np.array([goal_dxy[0], goal_dxy[1], goal_dz],
                              dtype=np.float32)
        self._last_goal_delta = goal_delta

        # Load rigid objects.
        rigid_ids = []
        for name, kwargs in SCENE_INFO[scene_name]['entities'].items():
            rgba_color = kwargs['rgbaColor'] if 'rgbaColor' in kwargs else None
            texture_file = None
            if 'useTexture' in kwargs and kwargs['useTexture']:
                texture_file = self.get_texture_path(args.rigid_texture_file)
            # Apply the same dxy shift to hanger + tallrod for hangcloth scenes.
            base_position = list(kwargs['basePosition'])
            if _is_hangcloth_scene:
                base_position[0] = float(base_position[0]) + float(goal_delta[0])
                base_position[1] = float(base_position[1]) + float(goal_delta[1])
                base_position[2] = float(base_position[2]) + float(goal_delta[2])
            id = load_rigid_object(
                sim, os.path.join(data_path, name), kwargs['globalScaling'],
                base_position, kwargs['baseOrientation'],
                kwargs.get('mass', 0.0), texture_file, rgba_color)
            rigid_ids.append(id)

        # Mark the goal and store intermediate info for reward computations.
        goal_poses = SCENE_INFO[scene_name]['goal_pos']
        if _is_hangcloth_scene and (randomize_r > 0.0 or randomize_dz > 0.0):
            goal_poses = [
                [float(p[0]) + float(goal_delta[0]),
                 float(p[1]) + float(goal_delta[1]),
                 float(p[2]) + float(goal_delta[2])]
                for p in goal_poses
            ]
        if args.viz and debug:
            for i, goal_pos in enumerate(goal_poses):
                print(f'goal_pos{i}', goal_pos)
                alpha = 1 if i == 0 else 0.3  # primary vs secondary goal
                create_anchor_geom(sim, goal_pos, mass=0.0,
                                   rgba=(0, 1, 0, alpha), use_collision=False)
        if scene_name == 'foodpacking':
            # Save mesh info used for computing penalty for food packing task.
            _, vertices = get_mesh_data(sim, deform_id)
            vertices = np.array(vertices)
            relative_dist = np.linalg.norm(vertices - vertices[[0]], axis=1)
            self.deform_shape_sample_idx = np.random.choice(np.arange(
                vertices.shape[0]), 20, replace=False)
            self.deform_init_shape = relative_dist[self.deform_shape_sample_idx]

        return rigid_ids, deform_id, deform_obj, np.array(goal_poses)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.stepnum = 0
        self.episode_reward = 0.0
        self.anchors = {}

        if self.args.viz:  # no rendering during load
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        # Reset pybullet sim to clear out deformables and reload objects.
        plane_texture_path = os.path.join(
            self.args.data_path,  self.get_texture_path(
                self.args.plane_texture_file))
        reset_bullet(self.args, self.sim, plane_texture=plane_texture_path)
        res = self.load_objects(self.sim, self.args, self.args.debug)
        self.rigid_ids, self.deform_id, self.deform_obj, self.goal_pos = res

        # Special case for Procedural Cloth tasks that can have two holes:
        # reward is based on the closest hole.
        if self.args.env.startswith('HangProcCloth'):
            self.goal_pos = np.vstack((self.goal_pos, self.goal_pos))

        self.sim.stepSimulation()  # step once to get initial state
        debug_mrks = None
        if self.args.debug and self.args.viz:
           debug_mrks = self.debug_viz_true_loop()

        # Setup dynamic anchors.
        if not self.food_packing:
            self.make_anchors()

        # Set up viz.
        if self.args.viz:  # loading done, so enable debug rendering if needed
            time.sleep(0.1)  # wait for debug visualizer to catch up
            self.sim.stepSimulation()  # step once to get initial state
            self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
            if debug_mrks is not None:
                input('Visualized true loops; press ENTER to continue')
                for mrk_id in debug_mrks:
                    # removeBody doesn't seem to work, so just make invisible
                    self.sim.changeVisualShape(mrk_id, -1,
                                               rgbaColor=[0, 0, 0, 0])

        obs, _ = self.get_obs()
        return obs

    def make_anchors(self):
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        for i in range(self.num_anchors):  # make anchors
            anchor_init_pos = self.args.anchor_init_pos if (i % 2) == 0 else \
                self.args.other_anchor_init_pos
            anchor_id, anchor_pos, anchor_vertices = create_anchor(
                self.sim, anchor_init_pos, i,
                preset_dynamic_anchor_vertices, mesh)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)
            self.anchors[anchor_id] = {'pos': anchor_pos,
                                       'vertices': anchor_vertices}

    def debug_viz_true_loop(self):
        # DEBUG visualize true loop vertices
        # Note: this function can be very slow when the number of ground truth
        # vertices marked is large, because it will create many visual elements.
        # So, use it sparingly (e.g. just only at trajectory start/end).
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return
        true_vs_lists = self.args.deform_true_loop_vertices
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        all_vs = np.array(vertex_positions)
        mrk_ids = []
        clrs = [cm.tab10(i) for i in range(len(true_vs_lists))]
        for l, loop_v_lst in enumerate(true_vs_lists):
            curr_vs = all_vs[loop_v_lst]
            for v in curr_vs:
                mrk_ids.append(create_anchor_geom(
                    self.sim, v, mass=0.0, radius=0.05,
                    rgba=clrs[l], use_collision=False))
        return mrk_ids

    def step(self, action, unscaled=False):
        if self.args.debug:
            print('action', action)
        if not unscaled:
            assert self.action_space.contains(action)
            assert ((np.abs(action) <= 1.0).all()), 'action must be in [-1, 1]'
        action = action.reshape(self.num_anchors, -1)

        # Step through physics simulation.
        for sim_step in range(self.args.sim_steps_per_action):
            self.do_action(action, unscaled)
            self.sim.stepSimulation()

        # Get next obs, reward, done.
        next_obs, done = self.get_obs()
        reward = self.get_reward()
        # Snapshot the per-step (un-multiplied) reward at the terminal frame,
        # so wrappers can shape on the policy-handoff pose BEFORE the gravity
        # settle in make_final_steps mutates the cloth.
        pre_settle_reward = reward
        if done:  # if terminating early use reward from current step for rest
            reward *= (self.max_episode_len - self.stepnum)
        done = (done or self.stepnum >= self.max_episode_len)

        # Update episode info and call make_final_steps if needed.
        if done:
            # Compute final reward by releasing anchors to let the object fall.
            info = self.make_final_steps()
            info['pre_settle_reward'] = pre_settle_reward
            info['pre_settle_dist_m'] = (
                -pre_settle_reward * self.WORKSPACE_BOX_SIZE)
            last_rwd = self.get_reward() * DeformEnv.FINAL_REWARD_MULT
            info['is_success'] = np.abs(last_rwd) < self.SUCESS_REWARD_TRESHOLD
            reward += last_rwd
            info['final_reward'] = reward
            print(f'final_reward: {reward:.4f}')
        else:
            info = {}

        self.episode_reward += reward  # update episode reward

        if self.args.debug and self.stepnum % 10 == 0:
            print(f'step {self.stepnum:d} reward {reward:0.4f}')
            if done:
                print(f'episode reward {self.episode_reward:0.4f}')
            
        self.stepnum += 1

        return next_obs, reward, done, info

    def do_action(self, action, unscaled):
        # Action is num_anchors x 3 for 3D velocity for anchors/grippers.
        # Assume action in [-1,1], convert to [-MAX_ACT_VEL, MAX_ACT_VEL].
        for i in range(self.num_anchors):
            command_anchor_velocity(
                self.sim, self.anchor_ids[i],
                DeformEnv.unscale_vel(action[i], unscaled))

    def make_final_steps(self):
        # We do no explicitly release the anchors, since this can create a jerk
        # and large forces.
        # release_anchor(self.sim, self.anchor_ids[0])
        # release_anchor(self.sim, self.anchor_ids[1])
        change_anchor_color_gray(self.sim, self.anchor_ids[0])
        change_anchor_color_gray(self.sim, self.anchor_ids[1])
        info = {'final_obs': []}
        if self._record_settle_frames:
            info['settle_frames'] = []
        sub_step_idx = 0
        for sim_step in range(DeformEnv.STEPS_AFTER_DONE):
            # For lasso pull the string at the end to test lasso loop.
            # For other tasks noop action to let the anchors fall.
            if self.args.task.lower() == 'lasso':
                if sim_step % self.args.sim_steps_per_action == 0:
                    action = [10*DeformEnv.MAX_ACT_VEL,
                              10*DeformEnv.MAX_ACT_VEL, 0]
                    self.do_action(action, unscaled=True)
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                next_obs, _ = self.get_obs()
                info['final_obs'].append(next_obs)
                if (self._record_settle_frames and
                        sub_step_idx % self._settle_frame_stride == 0):
                    info['settle_frames'].append(self.render(
                        mode='rgb_array', **self._settle_render_kwargs))
                sub_step_idx += 1
        return info

    def get_pcd_obs(self):
        """ Grab Pointcloud observations based from the camera config. """

        # Grab pcd observation from the camera_config camera
        segmented_pcd, segmented_ids, img = ProcessCamera.render(
            self.sim, self.camera_config, width=self.args.cam_resolution,
            height=self.args.cam_resolution, object_ids=self.object_ids,
            return_rgb=True, retain_unknowns=True,
            debug=False)

        # Process the RGB image
        img = img[...,:-1] # drop the alpha
        if self.args.uint8_pixels:
            img = img.astype(np.uint8)  # already in [0,255]
        else:
            img = img.astype(np.float32)/255.0  # to [0,1]
            img = np.clip(img, 0, 1)
        if self.args.flat_obs:
            img = img.reshape(-1)
        atol = 0.0001
        if ((img < self.observation_space.low-atol).any() or
            (img > self.observation_space.high+atol).any()):
            print('img', img.shape, f'{np.min(img):e}, n{np.max(img):e}')
            assert self.observation_space.contains(img)

        # Package and return
        obs = {'img': img,
               'pcd': segmented_pcd,
               'ids': segmented_ids
              }

        return obs 
    
    def get_obs(self):
        grip_obs = self.get_grip_obs()
        done = False
        grip_obs = np.nan_to_num(np.array(grip_obs))
        if (np.abs(grip_obs) > self.gripper_lims).any():  # at workspace lims
            if self.args.debug:
                print('clipping grip_obs', grip_obs)
            grip_obs = np.clip(
                grip_obs, -1.0*self.gripper_lims, self.gripper_lims)
            done = True
        if self.args.cam_resolution <= 0:
            obs = grip_obs
        else:
            obs = self.render(mode='rgb_array', width=self.args.cam_resolution,
                              height=self.args.cam_resolution)
            if self.args.uint8_pixels:
                obs = obs.astype(np.uint8)  # already in [0,255]
            else:
                obs = obs.astype(np.float32)/255.0  # to [0,1]
                obs = np.clip(obs, 0, 1)
        if self.args.flat_obs:
            obs = obs.reshape(-1)
        atol = 0.0001
        if ((obs < self.observation_space.low-atol).any() or
            (obs > self.observation_space.high+atol).any()):
            print('obs', obs.shape, f'{np.min(obs):e}, n{np.max(obs):e}')
            assert self.observation_space.contains(obs)

        return obs, done

    def get_grip_obs(self):
        anc_obs = []
        for i in range(self.num_anchors):
            pos, _ = self.sim.getBasePositionAndOrientation(
                self.anchor_ids[i])
            linvel, _ = self.sim.getBaseVelocity(self.anchor_ids[i])
            anc_obs.extend(pos)
            anc_obs.extend((np.array(linvel)/DeformEnv.MAX_OBS_VEL))
        return anc_obs

    def get_reward(self):
        _, vertex_positions = get_mesh_data(self.sim, self.deform_id)
        dist = []
        if not hasattr(self.args, 'deform_true_loop_vertices'):
            return 0.0  # no reward info without info about true loops
        # Compute distance from loop/hole to the corresponding target.
        num_holes_to_track = min(
            len(self.args.deform_true_loop_vertices), len(self.goal_pos))
        for i in range(num_holes_to_track):  # loop through goal vertices
            true_loop_vertices = self.args.deform_true_loop_vertices[i]
            goal_pos = self.goal_pos[i]
            pts = np.array(vertex_positions)
            cent_pts = pts[true_loop_vertices]
            cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]  # remove nans
            if len(cent_pts) == 0 or np.isnan(cent_pts).any():
                dist = self.WORKSPACE_BOX_SIZE*num_holes_to_track
                dist *= DeformEnv.FINAL_REWARD_MULT
                # Save a screenshot for debugging.
                # obs = self.render(mode='rgb_array', width=300, height=300)
                # pth = f'nan_{self.args.env}_s{self.stepnum}.npy'
                # np.save(os.path.join(self.args.logdir, pth), obs)
                break
            cent_pos = cent_pts.mean(axis=0)
            dist.append(np.linalg.norm(cent_pos - goal_pos))

        if self.args.env.startswith('HangProcCloth'):
            dist = np.min(dist)
        else:
            dist = np.mean(dist)
        rwd = -1.0 * dist / self.WORKSPACE_BOX_SIZE
        return rwd

    def render(self, mode='rgb_array', width=300, height=300):
        assert (mode == 'rgb_array')
        w, h, rgba_px, _, _ = self.sim.getCameraImage(
            width=width, height=height,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=self._cam_viewmat, **DEFAULT_CAM_PROJECTION)
        # If getCameraImage() returns a tuple instead of numpy array that
        # means that pybullet was installed without numpy being present.
        # Uninstall pybullet, then install numpy, then install pybullet.
        assert (isinstance(rgba_px, np.ndarray)), 'Install numpy, then pybullet'
        img = rgba_px[:, :, 0:3]
        return img


class HangProcClothRealEnv(DeformEnv):
    """Real-world-scale variant of HangProcCloth.

    Identical to DeformEnv but with WORKSPACE_BOX_SIZE reduced to 1.0 m
    so that reward magnitudes and observation bounds match the metre-scale
    coordinate system (hanger at ~0.32 m, cloth starting at ~0.27 m).

    Also uses a smaller anchor radius (5 mm vs the default 50 mm) so that
    the gripper spheres do not collide with each other or with the cloth
    mesh elements (which are ~19 mm wide at this scale).
    """
    WORKSPACE_BOX_SIZE = 1.0
    ANCHOR_RADIUS = 0.005  # 5 mm — scaled down from 50 mm for metre-scale env

    # NOTE: do_action is intentionally NOT overridden. The base-class
    # force-PD controller (command_anchor_velocity, ±CTRL_MAX_FORCE=10 N)
    # is used. An earlier kinematic resetBaseVelocity override existed to
    # beat gravity drift when the cloth was 1 kg (~5.9 N/anchor, too close
    # to the 10 N cap). The cloth is now 0.1 kg (deform_mass), so the
    # gravity load is ~0.5 N/anchor with ample control headroom. Force
    # control also preserves cloth↔anchor coupling: when the cloth resists,
    # the anchor naturally slows, keeping the sheet taut — kinematic
    # control decoupled this and whipped the cloth into a crumple.

    def make_final_steps(self):
        # Override: hold anchors stationary (v=0) during the gravity settle so
        # they do not fall through the floor (use_collision=False). The cloth
        # drapes naturally around the hanger peg while the anchors hover.
        change_anchor_color_gray(self.sim, self.anchor_ids[0])
        change_anchor_color_gray(self.sim, self.anchor_ids[1])
        info = {'final_obs': []}
        if self._record_settle_frames:
            info['settle_frames'] = []
        sub_step_idx = 0
        for sim_step in range(DeformEnv.STEPS_AFTER_DONE):
            for anchor_id in self.anchor_ids:
                self.sim.resetBaseVelocity(
                    anchor_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self.sim.stepSimulation()
            if sim_step % self.args.sim_steps_per_action == 0:
                next_obs, _ = self.get_obs()
                info['final_obs'].append(next_obs)
                if (self._record_settle_frames and
                        sub_step_idx % self._settle_frame_stride == 0):
                    info['settle_frames'].append(self.render(
                        mode='rgb_array', **self._settle_render_kwargs))
                sub_step_idx += 1
        return info

    def make_anchors(self):
        from ..utils.anchor_utils import (
            attach_anchor, create_anchor_geom, ANCHOR_RGBA_ACTIVE)
        from ..utils.init_utils import get_preset_properties
        from ..utils.mesh_utils import get_mesh_data
        from ..utils.task_info import DEFORM_INFO
        import numpy as np
        preset_dynamic_anchor_vertices = get_preset_properties(
            DEFORM_INFO, self.deform_obj, 'deform_anchor_vertices')
        _, mesh = get_mesh_data(self.sim, self.deform_id)
        mesh = np.array(mesh)
        for i in range(self.num_anchors):
            anchor_vertices = preset_dynamic_anchor_vertices[i]
            anchor_pos = mesh[anchor_vertices].mean(axis=0)
            # use_collision=False: prevent the anchor sphere from colliding
            # with nearby cloth vertices (which causes physics explosions at
            # metre scale where the 5 mm sphere is close to cloth elements).
            anchor_id = create_anchor_geom(
                self.sim, anchor_pos,
                mass=0.1, radius=self.ANCHOR_RADIUS,
                rgba=ANCHOR_RGBA_ACTIVE, use_collision=False)
            attach_anchor(self.sim, anchor_id, anchor_vertices, self.deform_id)
            self.anchors[anchor_id] = {'pos': anchor_pos,
                                       'vertices': anchor_vertices}
