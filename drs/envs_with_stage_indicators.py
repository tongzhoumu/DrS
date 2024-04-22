from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.registration import register_env
import numpy as np
from collections import OrderedDict

class DrS_BaseEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "semi_sparse")

    def compute_stage_indicator(self):
        raise NotImplementedError()

    def _get_obs_state_dict(self) -> OrderedDict:
        ret = super()._get_obs_state_dict()
        ret['extra'].update(self.compute_stage_indicator())
        return ret
    
    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            eval_info = self.evaluate(**kwargs)
            return float(eval_info["success"])
        elif self._reward_mode == "dense":
            return self.compute_dense_reward(**kwargs)
        elif self._reward_mode == "normalized_dense":
            return self.compute_normalized_dense_reward(**kwargs)
        elif self._reward_mode == "semi_sparse":
            # reward build from stage indicators
            return self.compute_semi_sparse_reward(**kwargs)
        else:
            raise NotImplementedError(self._reward_mode)
        
    def compute_semi_sparse_reward(self, info, **kwargs):
        stage_indicators = self.compute_stage_indicator()
        eval_info = self.evaluate(**kwargs)
        return sum(stage_indicators.values()) + float(eval_info["success"])


############################################
# Pick And Place
############################################

from mani_skill2.envs.pick_and_place.pick_single import PickSingleYCBEnv, PickSingleEGADEnv

@register_env("PickAndPlace_DrS_learn-v0", max_episode_steps=100)
class PickAndPlace_DrS_learn(PickSingleYCBEnv, DrS_BaseEnv):
    def check_obj_placed(self):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        return np.linalg.norm(obj_to_goal_pos) <= self.goal_thresh

    def compute_stage_indicator(self):
        return {
            'is_grasped': float(self.agent.check_grasp(self.obj)),
            'is_obj_placed': float(self.check_obj_placed()),
        }

@register_env("PickAndPlace_DrS_reuse-v0", max_episode_steps=100)
class PickAndPlace_DrS_reuse(PickSingleEGADEnv, PickAndPlace_DrS_learn):
    pass


############################################
# Turn Faucet
############################################

from mani_skill2.envs.misc.turn_faucet import (
    TurnFaucetEnv, transform_points, load_json
)
from mani_skill2 import PACKAGE_ASSET_DIR

class TurnFaucetEnv_DrS(TurnFaucetEnv, DrS_BaseEnv):
    
    def _get_obs_extra(self) -> OrderedDict:
        ret = super()._get_obs_extra()
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        ret.update(
            handle_center=np.mean(pcd, axis=0),
            lfinger_center=np.mean(pcd1, axis=0),
            rfinger_center=np.mean(pcd2, axis=0),
            target_joint_qvel=self.faucet.get_qvel()[self.target_joint_idx],
        )
        return ret

    def _initialize_task(self):
        super()._initialize_task()
        self._last_angle = self.current_angle

    def step_action(self, action):
        self._last_angle = self.current_angle
        super().step_action(action)
    
    def compute_stage_indicator(self):
        delta_angle = self.current_angle - self._last_angle
        success = self.evaluate()['success']

        return {
            'handle_move': float((delta_angle > 1e-3) or success),
        }
    

@register_env("TurnFaucet_DrS_learn-v0", max_episode_steps=100)
class TurnFaucetEnv_DrS_learn(TurnFaucetEnv_DrS):
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        model_ids = ('5028','5063','5034','5000','5006','5039','5056','5020','5027','5041')
        super().__init__(*args, model_ids=model_ids, **kwargs)

@register_env("TurnFaucet_DrS_reuse-v0", max_episode_steps=100)
class TurnFaucetEnv_DrS_reuse(TurnFaucetEnv_DrS):
    def __init__(
        self,
        *args,
        model_ids = (),
        **kwargs,
    ):
        model_json = f"{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_faucet_train.json"
        model_db = load_json(model_json)
        exclude_model_ids = ('5028','5063','5034','5000','5006','5039','5056','5020','5027','5041')
        model_ids = sorted(model_db.keys())
        model_ids = [model_id for model_id in model_ids if model_id not in exclude_model_ids]
        super().__init__(*args, model_ids=model_ids, **kwargs)


############################################
# Open Cabinet Door
############################################
        
from mani_skill2.envs.ms1.open_cabinet_door_drawer import (
    OpenCabinetDoorEnv, 
    vectorize_pose, 
    clip_and_normalize,
    Pose,
    sdist,
)
import sapien.core as sapien
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import get_pairwise_contact_impulse, get_entity_by_name
from mani_skill2.agents.robots.mobile_panda import MobilePandaSingleArm
import trimesh

class MobilePandaSingleArm_with_utils(MobilePandaSingleArm):
    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85):
        # This function is migrated from Panda Agent
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
        rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle = compute_angle_between(ldirection, limpulse)
        rangle = compute_angle_between(rdirection, rimpulse)

        lflag = (
            np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
        )
        rflag = (
            np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle
        )

        return all([lflag, rflag])

@register_env("OpenCabinetDoor_DrS_learn-v0", model_ids = ('1000','1065'), max_episode_steps=100)
@register_env("OpenCabinetDoor_DrS_reuse-v0", model_ids = ('1034','1052','1078'), max_episode_steps=100)
class OpenCabinetDoorEnv_DrS(OpenCabinetDoorEnv, DrS_BaseEnv):
    agent: MobilePandaSingleArm_with_utils

    def _load_agent(self):
        self.agent = MobilePandaSingleArm_with_utils(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        links = self.agent.robot.get_links()
        self.tcp: sapien.Link = get_entity_by_name(links, "right_panda_hand_tcp")

    def _initialize_task(self):
        super()._initialize_task()
        self._set_target_handle_info()

    def _load_articulations(self):
        super()._load_articulations()
        if self._reward_mode not in ["dense", "normalized_dense"]:
            self.cabinet.set_pose(Pose())
            self._set_cabinet_handles_mesh()
            self._compute_handles_grasp_poses()

    def _compute_grasp_poses(self, mesh: trimesh.Trimesh, pose: sapien.Pose):
        # we didn't modify this function, just save one varible from this function
        mesh2: trimesh.Trimesh = mesh.copy()
        mesh2.apply_transform(pose.to_transformation_matrix())
        extents = mesh2.extents
        if extents[1] > extents[2]:  # horizontal handle
            closing = np.array([0, 0, 1])
        else:  # vertical handle
            closing = np.array([0, 1, 0])
        self.extents = extents # save this

        approaching = [1, 0, 0]
        grasp_poses = [
            self.agent.build_grasp_pose(approaching, closing, [0, 0, 0]),
            self.agent.build_grasp_pose(approaching, -closing, [0, 0, 0]),
        ]
        pose_inv = pose.inv()
        grasp_poses = [pose_inv * x for x in grasp_poses]

        return grasp_poses

    def _get_obs_extra(self):
        # A unified obs for different cabinet instances 

        obs = OrderedDict()
        eval_info = self.evaluate()
        # robot info
        obs.update(
            **self.agent.get_fingers_info(),
            ee_pose=vectorize_pose(self.agent.hand.pose),
        )
        # object info
        obs.update(
            target_joint_axis=self.target_joint_axis,
            target_link_cmass_pos=self.target_link_pos,
            target_link_pose=vectorize_pose(self.target_link.pose),
            target_angle_to_go=clip_and_normalize(self.link_qpos, 0, self.target_qpos),
            targe_link_qvel=self.link_qvel,
            link_vel_norm=eval_info["link_vel_norm"],
            link_ang_vel_norm=eval_info["link_ang_vel_norm"],
            handle_direction=float(self.extents[1] > self.extents[2]),
        )
        # relation between robot and object
        handle_pose = self.target_link.pose
        ee_coords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        handle_pcd = transform_points(
            handle_pose.to_transformation_matrix(), self.target_handle_pcd
        )
        disp_ee_to_handle = sdist.cdist(ee_coords.reshape(-1, 3), handle_pcd)
        dist_ee_to_handle = disp_ee_to_handle.reshape(2, -1).min(-1)  # [2]
        ee_center_at_world = ee_coords.mean(0)  # [10, 3]
        ee_center_at_handle = transform_points(
            handle_pose.inv().to_transformation_matrix(), ee_center_at_world
        )
        dist_ee_center_to_handle = self.target_handle_sdf.signed_distance(
            ee_center_at_handle
        )
        dist_ee_center_to_handle = dist_ee_center_to_handle.max()
        obs.update(
            dist_ee_to_handle=dist_ee_to_handle,
            dist_ee_center_to_handle=dist_ee_center_to_handle,
        )

        return obs

    def compute_stage_indicator(self):
        open_enough = self.evaluate()['open_enough']
        return {
            'is_grasp': float(self.agent.check_grasp(self.target_link) or open_enough),
            'open_enough': float(open_enough),
        }
    
    ##################################################################
    # The following code is to fix a bug in MS1 envs (0.5.3)
    ##################################################################
    def reset(self, seed=None, options=None):
        self._prev_actor_pose_dict = {}
        return super().reset(seed, options)
    
    def check_actor_static(self, actor: sapien.Actor, max_v=None, max_ang_v=None):
        """Check whether the actor is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues.
        """

        from mani_skill2.utils.geometry import angle_distance

        pose = actor.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            prev_actor_pose, prev_step, prev_actor_static = self._prev_actor_pose_dict[actor.id]
            if prev_step == self._elapsed_steps:
                return prev_actor_static
            assert prev_step == self._elapsed_steps - 1, (prev_step, self._elapsed_steps)
            dt = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - prev_actor_pose.p) <= max_v * dt
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(prev_actor_pose, pose) <= max_ang_v * dt
            )

        # CAUTION: carefully deal with it for MPC
        actor_static = flag_v and flag_ang_v
        self._prev_actor_pose_dict[actor.id] = (pose, self._elapsed_steps, actor_static)
        return actor_static