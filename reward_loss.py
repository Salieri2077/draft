# -*- coding: utf-8 -*-


"""Reward Loss."""
from dataclasses import dataclass
import logging
from torchpilot import logger
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as torch_nn_func
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import Polygon as ShpPolygon
from shapely.affinity import rotate, scale
from shapely.strtree import STRtree
from torchpilot.model.loss.base_loss import BaseLoss
from torchpilot.utils.registries import LOSSES
from shapely.affinity import rotate, scale
from torchpilot.utils.torch_compile import compile_wrapper

from tpp_onemodel.constant import LANE_DIRECTION_TO_TLD_ID_MAPPING
from tpp_onemodel.data.dataset.plannn2_dataset_utils.common import ScenarioEnum
from tpp_onemodel.utils.collision_utils import agent_reward_collision_check
from tpp_onemodel.utils.collision_utils import lane_reward_collision_check
from tpp_onemodel.utils.collision_utils import point_to_trajectory_distance
from tpp_onemodel.utils.reward_utils import Turntype, judge_intersection_and_maneuver
from tpp_onemodel.utils.reward_utils import calc_path_length
from tpp_onemodel.utils.reward_utils import calc_path_length_from_point
from tpp_onemodel.utils.reward_utils import calc_path_point_heading
from tpp_onemodel.utils.reward_utils import determine_path_turn_type
from tpp_onemodel.utils.reward_utils import find_nearest_point
from tpp_onemodel.utils.reward_utils import (
    get_distance,
    get_distance_with_index,
    get_all_distances_with_indices,
    get_min_distance_line_details,
)
from tpp_onemodel.utils.reward_utils import get_lane_direction
from tpp_onemodel.utils.reward_utils import get_lat_distance
from tpp_onemodel.utils.reward_utils import get_lon_lat_distance
from tpp_onemodel.utils.reward_utils import polygon_rear_padding
from tpp_onemodel.utils.reward_utils import project_to_line
from tpp_onemodel.utils.reward_utils import get_lines_distance
from tpp_onemodel.utils.reward_utils import get_lon_lat_distance
from tpp_onemodel.utils.reward_utils import polygon_rear_padding
from tpp_onemodel.utils.reward_utils import get_lines_distance
from tpp_onemodel.utils.reward_utils import get_map_cls_to_3cls
from tpp_onemodel.utils.reward_utils import get_projection_on_segment
from tpp_onemodel.utils.reward_utils import get_relative_pose_from_obj
from tpp_onemodel.utils.reward_utils import get_vector_angle
from tpp_onemodel.utils.reward_utils import get_xyyaw_from_polygon
from tpp_onemodel.utils.reward_utils import is_segments_intersection
from tpp_onemodel.utils.reward_utils import line_intersection_point
from tpp_onemodel.utils.reward_utils import match_road_sign_from_path
from tpp_onemodel.utils.reward_utils import normalize_angle
from tpp_onemodel.utils.reward_utils import polygon_to_segments
from tpp_onemodel.utils.reward_utils import road_arrow_to_lane_direction_mapping
from tpp_onemodel.utils.reward_utils import segment_intersect
from tpp_onemodel.utils.reward_utils import split_points_to_segments
from tpp_onemodel.utils.reward_utils import calculate_signed_lateral_distance

from tpp_onemodel.data.dataset.plannn2_dataset_utils.data_gate_selector import build_gate_gt_fusion

from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import distance_ranges_to_target_scene
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import ROAD_SPLIT_MAIN_ACTION_MAPPING
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import ROAD_SPLIT_ASSIST_ACTION_MAPPING
from tpp_onemodel.utils.reward_utils import signed_distance
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import distance_ranges_to_current_link_only
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import LEFT_RIGHT_TURN_MAIN_ACTION_MAPPING
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import NONE_ASSIST_ACTION_MAPPING

from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils.misc import filter_opened_gate_sobj

from tpp_onemodel.utils.slow_follow_detector_v3_2 import SlowFollowSegment


class BaseRewardLoss(BaseLoss):
    def __init__(self, funcs_cfg, loss_scales) -> None:
        """Init."""
        super().__init__()
        self.reward_funcs = {key: LOSSES.build(cfg) for key, cfg in funcs_cfg.items()}
        self.loss_scales = loss_scales


class AgentCollisionReward(BaseLoss):
    def __init__(self, ignore_collision_afterwards) -> None:
        """Init."""
        super().__init__()
        self.ignore_collision_afterwards = ignore_collision_afterwards

    def forward(self, traj_sample_pred: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Od collision reward."""
        traj_pred_pos = traj_sample_pred["traj_pred_pos"]
        traj_pred_vel = traj_sample_pred["traj_pred_vel"]
        traj_pred_yaw = traj_sample_pred["traj_pred_yaw"]
        _, traj_num, time_step, _ = traj_pred_pos.shape

        # traj_pred_pos = traj_pred_refined[:, :, :, :2].detach()  # [B, N, T, 2, M]
        # traj_pred_pos = traj_pred_pos.permute(0, 1, 4, 2, 3).reshape(B, N * M, T, 2)
        # traj_pred_vel = traj_pred_refined[:, :, :, -3:-1].detach()  # [B, N, T, 2, M]
        # traj_pred_vel = traj_pred_vel.permute(0, 1, 4, 2, 3).reshape(B, N * M, T, 2)
        # traj_pred_yaw = traj_pred_refined[:, :, :, -1:].detach()  # [B, N, T, 1, M]
        # traj_pred_yaw = traj_pred_yaw.permute(0, 1, 4, 2, 3).reshape(B, N * M, T, 1)
        ego_bbox_info = model_inputs["egos_size"]
        ego_gt_mask = model_inputs["egos_pred_mask"]
        agent_gts = model_inputs["bboxes_pred_gt"]
        agent_bbox_infos = model_inputs["bboxes_det_pred_full_info_gt"]  # B,A,T,D
        agent_gt_masks = model_inputs["bboxes_pred_mask"]
        # back_mask = torch.logical_and(agent_bbox_infos[:, :, 0, 0] < 0, agent_bbox_infos[:, :, 0, 1].abs() < 4.8)
        # back_mask = agent_bbox_infos[:, :, 0, 0] < 0
        # agent_gt_masks = agent_gt_masks * (~back_mask[:, :, None]).float()  # [B, 64, 35]
        vel_mask = torch.logical_and(
            agent_bbox_infos[:, :, 0, 10]
            > model_inputs["ego_vel_pred_gt"][:, 0, 0].unsqueeze(dim=1).expand_as(agent_bbox_infos[:, :, 0, 10]),
            agent_bbox_infos[:, :, 0, 7] >= 400,
        )
        ego_straight_mask = (
            torch.abs(
                model_inputs["egos_pred_gt"][:, 0, :, 1]
                - torch.mean(model_inputs["egos_pred_gt"][:, 0, :, 1], dim=-1, keepdim=True)
            )
            > 0.4
        ).sum(-1) == 0
        vel_mask = vel_mask & ego_straight_mask[:, None]
        agent_gt_masks = agent_gt_masks * (~vel_mask[:, :, None]).float()
        agent_gts = agent_gts[:, :, :time_step]
        agent_bbox_infos = agent_bbox_infos[:, :, :time_step]
        agent_gt_masks = agent_gt_masks[:, :, :time_step]
        ego_gt_mask = ego_gt_mask[:, :, :time_step]

        collision_check = agent_reward_collision_check(
            ego_pred_pos=traj_pred_pos,
            ego_pred_yaw=traj_pred_yaw,
            ego_pred_vel=traj_pred_vel,
            ego_bbox_info=ego_bbox_info,
            agent_gts=agent_gts,
            agent_bbox_infos=agent_bbox_infos,
            model_inputs=model_inputs,  # used for debug
        )  # [B, 64, 81, 35]

        # Apply agent mask
        agent_gt_masks = agent_gt_masks.unsqueeze(2).expand(*collision_check.shape)
        collision_check = collision_check * agent_gt_masks  # [B, 64, 81, 35]
        collision_label = collision_check.sum(dim=1) > 0  # [B, 81, 35]
        collision_label = collision_label.float()  # [B, 81, 35]

        # ignore the time steps after collision
        if self.ignore_collision_afterwards:
            collision_check_label_cummax = torch.cummax(collision_label, dim=-1)[0]  # [B, 81, 35]
            collision_ignore_mask = (
                collision_check_label_cummax == collision_label
            ).float()  # [B, 81, 35], 0 means ignore
        else:
            # collision_ignore_mask = torch.full_like(collision_label, 1.0)  # 0 means ignore
            collision_ignore_mask = ego_gt_mask.repeat(1, traj_num, 1)  # 0 means ignore

        # collision_label = collision_label.reshape(B, N, M, T).permute(0, 1, 3, 2)
        # collision_ignore_mask = collision_ignore_mask.reshape(B, N, M, T).permute(0, 1, 3, 2)
        collision_label = torch.where(collision_ignore_mask.bool(), collision_label, -1)

        return collision_label.detach(), collision_ignore_mask.detach(), collision_check, agent_gts


class LaneCollisionReward(BaseLoss):
    def __init__(self, ignore_collision_afterwards) -> None:
        """Init."""
        super().__init__()
        self.ignore_collision_afterwards = ignore_collision_afterwards

    def forward(self, traj_sample_pred: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Lane collision reward."""
        traj_pred_pos = traj_sample_pred["traj_pred_pos"]
        traj_pred_vel = traj_sample_pred["traj_pred_vel"]
        traj_pred_yaw = traj_sample_pred["traj_pred_yaw"]

        ego_bbox_info = model_inputs["egos_size"]
        lane_points = model_inputs["lane_points"]  # [B, 100, 50, 2]
        lane_types = model_inputs["lane_types"]  # [B, 100, 50], -1 for invalid, 0 for lane, 1 for road edge
        lane_attr = model_inputs["lane_attr"]  # [B, 100, 50, 2]

        collision_check, closest_lane_corners = lane_reward_collision_check(
            traj_pred_pos,
            traj_pred_yaw,
            traj_pred_vel,
            ego_bbox_info,
            lane_points,
            lane_types,
            lane_attr,
        )  # [B, 81, 35]
        collision_label = collision_check.float()  # [B, 81, 35]

        # ignore the time steps after collision,  0 means ignore
        if self.ignore_collision_afterwards:
            collision_check_label_cummax = torch.cummax(collision_label, dim=-1)[0]  # [B, 81, 35]
            collision_ignore_mask = (collision_check_label_cummax == collision_label).float()  # [B, 81, 35]
        else:
            collision_ignore_mask = torch.full_like(collision_label, 1.0)

        # collision_label = collision_label.reshape(B, N, M, T).permute(0, 1, 3, 2)
        # collision_ignore_mask = collision_ignore_mask.reshape(B, N, M, T).permute(0, 1, 3, 2)

        return collision_label.detach(), collision_ignore_mask.detach(), collision_check, closest_lane_corners


class TrajectoryCollisionLoss(BaseLoss):
    def __init__(self, reward_funcs_cfg, loss_scales, num_future_frames=35) -> None:
        """Init."""
        super().__init__()
        self.reward_funcs = {key: LOSSES.build(cfg) for key, cfg in reward_funcs_cfg.items()}
        self.loss_scales = loss_scales
        self.num_future_frames = num_future_frames
        self.agent_future_frames = 15

    def get_agent_collision_loss(self, model_pred, collision_label, collision_ignore_mask):
        """Agent collision value loss."""
        collision_label_pred = model_pred  # [B, 81, 35]
        collision_label_loss = torch_nn_func.binary_cross_entropy_with_logits(
            collision_label_pred, collision_label, reduction="none"
        )  # [B, 81, 35]
        # collision_ignore_mask[:, :, self.agent_future_frames :] = 0  # 0 means ignore
        collision_weight = torch.full_like(collision_ignore_mask, 1.0)
        collision_positive_weight = 10  # Enhance collision weights
        collision_weight[collision_label > 0] *= collision_positive_weight
        collision_weight[collision_ignore_mask == 0] = 0.0
        collision_loss = collision_label_loss * collision_weight
        collision_loss = collision_loss.sum() / (collision_weight.sum() + 1)
        return collision_loss

    def get_lane_collision_loss(self, model_pred, collision_label, collision_ignore_mask):
        """Lane collision value loss."""
        collision_label_pred = model_pred  # [B, 81, 35]
        collision_label_loss = torch_nn_func.binary_cross_entropy_with_logits(
            collision_label_pred, collision_label, reduction="none"
        )  # [B, 81, 35]
        collision_weight = torch.full_like(collision_ignore_mask, 1.0)
        collision_positive_weight = 10  # Enhance collision weights
        collision_weight[collision_label > 0] *= collision_positive_weight
        collision_weight[collision_ignore_mask == 0] = 0.0
        collision_loss = collision_label_loss * collision_weight
        collision_loss = collision_loss.sum() / (collision_weight.sum() + 1)
        return collision_loss

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Calculate collision value loss."""
        traj_sample_pred = model_outputs  # A dict
        # traj_pred_refined = traj_sample_pred["traj_pred_refined"]  # [B, N, T, D, M], [10, 1, 35, 8, 12]
        traj_pred_agent_collision = traj_sample_pred["traj_pred_agent_collision"].squeeze(-1)  # [B, N, T, M]
        traj_pred_lane_collision = traj_sample_pred["traj_pred_lane_collision"].squeeze(-1)  # [B, N, T, M]

        loss_dict = {}
        for key, reward_func in self.reward_funcs.items():
            if "agent_collision_reward" == key:
                with torch.no_grad():
                    collision_label, collision_ignore_mask = compile_wrapper(reward_func)(
                        traj_sample_pred={
                            "traj_pred_pos": traj_sample_pred["traj_pred_pos"],
                            "traj_pred_vel": traj_sample_pred["traj_pred_vel"],
                            "traj_pred_yaw": traj_sample_pred["traj_pred_yaw"],
                        },
                        model_inputs={
                            "egos_size": model_inputs["egos_size"],
                            "egos_pred_gt": model_inputs["egos_pred_gt"],
                            "egos_pred_mask": model_inputs["egos_pred_mask"],
                            "bboxes_pred_gt": model_inputs["bboxes_pred_gt"],
                            "bboxes_det_pred_full_info_gt": model_inputs["bboxes_det_pred_full_info_gt"],
                            "bboxes_pred_mask": model_inputs["bboxes_pred_mask"],
                            "ids": model_inputs["ids"],
                            "timestamps": model_inputs["timestamp"],
                            "subclip_ids": model_inputs["subclip_ids"],
                            "ego_vel_pred_gt": model_inputs["ego_vel_pred_gt"],
                        },
                    )[:2]
                # traj_pred_agent_collision = traj_pred_agent_collision.sigmoid()
                agent_collision_loss = self.get_agent_collision_loss(
                    traj_pred_agent_collision, collision_label, collision_ignore_mask
                )
                loss_dict["traj_od_collision_loss"] = agent_collision_loss * self.loss_scales[key]

            if "lane_collision_reward" == key:
                with torch.no_grad():
                    collision_label, collision_ignore_mask = reward_func(model_outputs, model_inputs)[:2]
                # traj_pred_lane_collision = traj_pred_lane_collision.sigmoid()  # [B, N, 35, 2], OD collision
                lane_collision_loss = self.get_lane_collision_loss(
                    traj_pred_lane_collision, collision_label, collision_ignore_mask
                )
                loss_dict["traj_lane_collision_loss"] = lane_collision_loss * self.loss_scales[key]

            # if "navi_follow_reward" == key:
            #     with torch.no_grad():
            #         navi_follow_min_dists = reward_func(model_outputs, model_inputs)
            #     navi_follow_loss = self.l1_loss_func(traj_pred_navi_follow, navi_follow_min_dists)
            #     navi_follow_loss = navi_follow_loss.mean()
            #     loss_dict["traj_navi_follow_loss"] = navi_follow_loss * self.loss_scales[key]

            # if "ego_jerk_reward" == key:
            #     with torch.no_grad():
            #         ego_jerk = reward_func(model_outputs, model_inputs)
            #     jerk_loss = self.l1_loss_func(traj_pred_jerk, ego_jerk)
            #     loss_dict["traj_ego_jerk_loss"] = jerk_loss * self.loss_scales[key]

        return loss_dict


class CollisionReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps

    def check_cross_laneline(self, laneline_penalties, check_seconds=2):
        """检查压线."""
        check_ts = self.fps * check_seconds
        return np.any(laneline_penalties[-min(check_ts, len(laneline_penalties)) :] < 0)

    def check_rear_collision(self, ego_box, ego_yaw, dynamic_boxes, dynamic_yaws, dynamic_cids):
        """
        判断哪些动态物体追尾了 ego 车.

        :param ego_box: shape (4, 2)  -> ego 车的四边形坐标, 顺序为: [(-y, -x), (-y, x), (y, x), (y, -x)]
        :param ego_yaw: shape (1,)       -> ego 车的朝向（弧度）
        :param dynamic_boxes: shape (n, 4, 2)  -> n 个动态物体的四边形坐标
        :param dynamic_yaws: shape (n,)       -> n 个动态物体的朝向（弧度）
        :return: 形状为 (n,) 的布尔张量，表示是否发生追尾
        """
        # 只保留角度差值在 ±30°（±pi/6）范围内的动态物体
        angle_diff = dynamic_yaws - ego_yaw
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        angle_condition = (np.abs(angle_diff) <= np.pi / 6) | (
            np.array([get_map_cls_to_3cls(cid) for cid in dynamic_cids]) == "pedestrian"
        )

        # 计算 ego 车后半区域
        rear_box = ego_box.copy()
        rear_box[1] = (rear_box[1] + rear_box[0]) / 2
        rear_box[2] = (rear_box[2] + rear_box[3]) / 2
        rear_poly = Polygon(rear_box)

        # 计算 ego 车前半区域
        front_box = ego_box.copy()
        front_box[0] = (front_box[1] + front_box[0]) / 2
        front_box[3] = (front_box[2] + front_box[3]) / 2
        front_poly = Polygon(front_box)

        # 计算相交情况和相交位置
        rear_collision_mask = np.zeros(dynamic_boxes.shape[0], dtype=np.bool_)
        for i, dyn_box in enumerate(dynamic_boxes):
            if angle_condition[i]:
                dyn_poly = Polygon(dyn_box)
                if rear_poly.intersects(dyn_poly) and not front_poly.intersects(dyn_poly):
                    rear_collision_mask[i] = True

        return rear_collision_mask

    def forward(
        self,
        ego_polygon,
        raw_env,
        ts,
        history_t_num,
        rear_collision_tids_all,
        laneline_penalties=None,
        is_ego_touch_solid_line_now=None,
        lc_ahead_obj_id = None,
    ):
        """
        计算碰撞Reward.

        Args:
            polygon: n x 4 x 2
            raw_env: dict
        Returns:
            collision_with_static: bool
            collision_with_dyn: bool
            collision_penalty: float
        """
        # compute ego pose
        polygon = ego_polygon[ts - history_t_num]
        ego_poly = Polygon(polygon)
        ego_xyzlwhyaw = get_xyyaw_from_polygon(polygon[None, :])[0]
        ego_yaw = ego_xyzlwhyaw[6]

        # compute obstacles
        lines = []
        sta_polygons = []
        god_polygon_lines = []
        rear_dyn_polygons = []
        normal_dyn_polygons = []
        sta_dyn_polygons = []

        # sobjs = raw_env[ts]["sobjs_polygon"]
        _, sobjs = filter_opened_gate_sobj(raw_env[ts])
        sta_polygons.append(sobjs)

        road_edge = raw_env[ts]["road_edge"]
        nomap_road_edge = raw_env[ts]["nomap_road_edge"]
        for p in (road_edge + nomap_road_edge):
            lines.append(p[:, :2])

        god_polygon = raw_env[ts]["god_polygon"]
        for p in god_polygon:
            god_polygon_lines.append(p[:, :2])

        if ts < len(raw_env[0]["dobjs_full"]):
            _, dobjs = raw_env[0]["dobjs_full"][ts]
            dobjs_polygon = raw_env[0]["dobjs_polygon"][ts]
            dobjs_tid = dobjs[:, 9]
            rear_dobjs_polygon = dobjs_polygon[np.isin(dobjs_tid, list(rear_collision_tids_all))]
            normal_dobjs_polygon = dobjs_polygon[~np.isin(dobjs_tid, list(rear_collision_tids_all))]
            normal_dobjs = dobjs[~np.isin(dobjs_tid, list(rear_collision_tids_all))]
            normal_dobjs_yaw = normal_dobjs[:, 6]
            normal_dobjs_cid = normal_dobjs[:, 7]
            normal_dobjs_tid = normal_dobjs[:, 9]
            # 换道侧前车后向padding
            if lc_ahead_obj_id is not None and lc_ahead_obj_id[ts - history_t_num] > 0.0:
                lc_ahead_obj_tid = lc_ahead_obj_id[ts - history_t_num]
                if np.any(normal_dobjs_tid == lc_ahead_obj_tid):
                    normal_dobjs_polygon[normal_dobjs_tid == lc_ahead_obj_tid] = polygon_rear_padding(normal_dobjs_polygon[normal_dobjs_tid == lc_ahead_obj_tid][0], 3.0)
            rear_dyn_polygons.append(rear_dobjs_polygon)
            normal_dyn_polygons.append(normal_dobjs_polygon)

            vx = dobjs[:, 10]
            vy = dobjs[:, 11]
            dobjs_speed = np.sqrt(vx**2 + vy**2)
            dobjs_speed_mask = dobjs_speed < 0.5
            sta_dyn_polygon = dobjs_polygon[(~np.isin(dobjs_tid, list(rear_collision_tids_all))) & dobjs_speed_mask]
            sta_dyn_polygons.append(sta_dyn_polygon)

        sta_polygons = np.concatenate(sta_polygons, axis=0) if len(sta_polygons) > 0 else np.zeros((0, 4, 2))
        rear_dyn_polygons = (
            np.concatenate(rear_dyn_polygons, axis=0) if len(rear_dyn_polygons) > 0 else np.zeros((0, 4, 2))
        )
        normal_dyn_polygons = (
            np.concatenate(normal_dyn_polygons, axis=0) if len(normal_dyn_polygons) > 0 else np.zeros((0, 4, 2))
        )
        sta_dyn_polygons = (
            np.concatenate(sta_dyn_polygons, axis=0) if len(sta_dyn_polygons) > 0 else np.zeros((0, 4, 2))
        )

        # 过滤掉追尾自车的动态物体
        if len(normal_dyn_polygons) > 0:
            rear_collision_mask = self.check_rear_collision(
                polygon, ego_yaw, normal_dyn_polygons, normal_dobjs_yaw, normal_dobjs_cid
            )
            if sum(rear_collision_mask):
                # 不能过滤的追尾: (自车变道 & 动态目标不是后车) or (动态目标极大概率不是后车)   # 根据轨迹重叠度判断是否是后车
                # 满足原始条件 或者 自车当前帧压线
                ego_cross_laneline = is_ego_touch_solid_line_now
                if (not ego_cross_laneline) and laneline_penalties is not None:
                    ego_cross_laneline = is_ego_touch_solid_line_now or self.check_cross_laneline(
                        laneline_penalties
                    )  # [50m内压线; 或者更远的地方连续10帧压线]
                # else:
                #     ego_cross_laneline = np.zeros_like(laneline_penalties)

                rear_collision_tids = normal_dobjs_tid[rear_collision_mask]
                rear_collision_cids = normal_dobjs_cid[rear_collision_mask]
                follow_mask = np.ones(rear_collision_tids.shape[0], dtype=np.bool_)
                ego_traj = ego_polygon[max(ts - history_t_num - 5 * self.fps, 0) : ts - history_t_num].mean(axis=1)
                ego_traj_dist = np.linalg.norm(ego_traj[-1] - ego_traj[0]) if len(ego_traj) > 0 else 0.0
                if ego_traj_dist > 10 and len(ego_traj) > 10:
                    tree = cKDTree(ego_traj)
                    for i, tid in enumerate(rear_collision_tids):
                        dobj_traj = []
                        for ti in range(max(0, ts - len(ego_traj)), ts, 1):
                            ti_dobjs = raw_env[0]["dobjs_full"][ti][1]
                            if tid in ti_dobjs[:, 9]:
                                dobj_traj.append(ti_dobjs[ti_dobjs[:, 9] == tid][0, :2])
                        if len(dobj_traj) < 3:
                            continue
                        distances, _ = tree.query(np.array(dobj_traj), k=1)  # 每个他车点到最近自车点的距离
                        match_ratio = np.sum(distances < 1.5) / len(dobj_traj)  # 最近距离在1.5m内的轨迹点认为匹配上
                        if (ego_cross_laneline and match_ratio < 0.2) or (
                            match_ratio < 0.05 and get_map_cls_to_3cls(rear_collision_cids[i]) == "pedestrian"
                        ):
                            follow_mask[i] = False

                rear_collision_mask[np.flatnonzero(rear_collision_mask)[~follow_mask]] = False
                if sum(rear_collision_mask) > 0:
                    rear_dyn_polygons = np.concatenate(
                        [rear_dyn_polygons, normal_dyn_polygons[rear_collision_mask]], axis=0
                    )
                    normal_dyn_polygons = normal_dyn_polygons[~rear_collision_mask]
                    rear_collision_tids_all.update(normal_dobjs_tid[rear_collision_mask])

        # compute ego-obstacle distance
        mindist_edge_start = False
        shape_lines = MultiLineString(lines)
        shape_sta_polygons = MultiPolygon([Polygon(p) for p in sta_polygons])
        shape_normal_dyn_polygons = MultiPolygon([Polygon(p) for p in normal_dyn_polygons])
        shape_rear_dyn_polygons = MultiPolygon([Polygon(p) for p in rear_dyn_polygons])
        shape_god_polygon_lines = MultiLineString(god_polygon_lines)
        shape_sta_dyn_polygons = MultiPolygon([Polygon(p) for p in sta_dyn_polygons])

        # static_distance = get_distance(ego_poly, shape_lines)
        static_distance, idex = get_distance_with_index(ego_poly, shape_lines)
        collision_with_static = static_distance == 0
        if len(lines) > 0 and lines[idex].shape[0] > 0:
            lines_start_pts = [lines[idex][: min(20, lines[idex].shape[0]), :]]
            # lines_start_pts = np.array([lines_start_pts.tolist()], dtype=np.float32)
            lines_mindist = MultiLineString(lines_start_pts)
            static_distance_min_start = get_distance(ego_poly, lines_mindist)
            if static_distance_min_start <= static_distance:
                mindist_edge_start = True

        if not collision_with_static:
            static_distance2 = get_distance(ego_poly, shape_sta_polygons)
            collision_with_static = static_distance2 == 0
            static_distance = min(static_distance, static_distance2)
        if not collision_with_static:
            static_distance3 = get_distance(ego_poly, shape_god_polygon_lines)
            collision_with_static = static_distance3 == 0
            static_distance = min(static_distance, static_distance3)

        normal_dyn_distance = get_distance(ego_poly, shape_normal_dyn_polygons)
        collision_with_dyn_normal = normal_dyn_distance == 0
        rear_distance = get_distance(ego_poly, shape_rear_dyn_polygons)
        collision_with_dyn_rear = rear_distance == 0
        sta_dyn_distance = get_distance(ego_poly, shape_sta_dyn_polygons)
        collision_with_sta_dyn = sta_dyn_distance == 0

        if collision_with_static or collision_with_dyn_normal:
            collision_penalty = 30
        else:
            collision_penalty = 0
        # min_distance = min(static_distance, normal_distance)
        return (
            collision_with_static,
            collision_with_dyn_normal,
            collision_with_dyn_rear,
            collision_with_sta_dyn,
            collision_penalty,
            static_distance,
            mindist_edge_start,
            normal_dyn_distance,
            rear_collision_tids_all,
        )


class TTCReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps

    def _calc_rela_vel_dict(self, raw_env, ego_polygon, ts):
        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        # 当前帧与上一帧的目标ID与中心点
        ts_ids = dobjs_full[ts][1][:, 9].astype(int)
        ts_centers = np.mean(dobjs_polygon[ts], axis=1)  # (N, 2)

        pre_ids = dobjs_full[ts - 1][1][:, 9].astype(int)
        pre_centers = np.mean(dobjs_polygon[ts - 1], axis=1)  # (M, 2)

        # 取两帧公共ID并对齐索引
        common_ids, idx_ts, idx_pre = np.intersect1d(ts_ids, pre_ids, assume_unique=False, return_indices=True)

        if common_ids.size == 0:
            return {}

        agent_v = (ts_centers[idx_ts] - pre_centers[idx_pre]) * self.fps  # (K, 2)
        ego_v = (np.mean(ego_polygon[ts], axis=0) - np.mean(ego_polygon[ts - 1], axis=0)) * self.fps  # (2,)

        rel_v_x = agent_v[:, 0] - ego_v[0]  # (K,)

        id2rela_vel = dict(zip(common_ids, rel_v_x))
        return id2rela_vel

    def _preprocess_cross_vru(self, raw_env, ts, ignore_track_ids, ego_polygon, id2rela_vel, max_penalty, direct_light, bypass_flag=False, vru_preds=None, vru_preds_plg=None):
        """
        当前方有横穿VRU风险时，直接返回相应的惩罚和目标vru id
        """
        if bypass_flag:
            return 0, None, False
        max_vru_penalty = 0
        target_vru_id = None
        vru_caution = False
        ego_in_junction = False
        min_distance = 2.5
        if not np.isnan(direct_light) or direct_light >= 0:
            ego_in_junction = True
        # reward param
        ego_width_buffer = 1.5 if ego_in_junction else 1.2
        vru_width_buffer = 0.3
        vru_history_frame_save_num = 1
        ego_collision_index_decay_coef = -4.0
        min_check_size = 18 if ego_in_junction else 20
        velocity_threshold = -2.0
        lateral_threshold = 2.5
        ego_straight_threshold = 1.0
        min_preempt_frame_threshold = 10 if ego_in_junction else 5

        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        vru_ids = set()
        _, dobjs = dobjs_full[ts]
        ignore_mask = np.isin(dobjs[:, 9].astype(int), ignore_track_ids)
        dobjs_cid = dobjs[:, 7]  # n,
        ego_cur_center = np.mean(ego_polygon[ts], axis=0)
        ego_traj = np.mean(ego_polygon, axis=1)
        ego_straight_mask = (
            abs(ego_cur_center[1] - np.mean(ego_traj[ts : min(ts + min_check_size, len(ego_polygon)), 1], axis=0))
            < ego_straight_threshold
        )
        vru_track_mask = (dobjs_cid >= 300) & ~ignore_mask  # n,
        if np.any(vru_track_mask):
            vru_ids.update(dobjs[vru_track_mask, 9].astype(int).tolist())
            vru_ids = list(vru_ids)
            if vru_preds is None:
                vru_preds = {}
            if vru_preds_plg is None:
                vru_preds_plg = {}
            ego_pred = []
            ego_pred_plg = []
        else:
            return max_vru_penalty, target_vru_id, vru_caution

        # ignore front faster vru
        ids_to_remove = []
        for i in range(0, len(dobjs)):
            id = dobjs[i, 9].astype(int)
            if not np.isin(id, vru_ids):
                continue
            vru_cur_center = np.mean(dobjs_polygon[ts][i], axis=0)
            dobjs_direct = vru_cur_center - ego_cur_center
            vru_at_ego_front = np.logical_and(abs(dobjs_direct[1]) < lateral_threshold, dobjs_direct[0] > 0)
            vru_is_faster = False
            if id in id2rela_vel.keys():
                vru_is_faster = id2rela_vel[id] > velocity_threshold
            if ego_straight_mask and vru_at_ego_front and vru_is_faster:
                # vru_ids.remove(id)
                ids_to_remove.append(id)
        for id in ids_to_remove:
            if id in vru_ids:
                vru_ids.remove(id)
        if len(vru_ids) == 0:
            return max_vru_penalty, target_vru_id, vru_caution

        # # 车速越小惩罚越小
        # velocity_scale = 1.0
        # if ego_max_velocity > 0:
        #     vxvy = np.mean(ego_polygon[ts], axis=0) - np.mean(ego_polygon[ts - 1], axis=0)
        #     vxvy = math.sqrt(vxvy[0]**2 + vxvy[1]**2) * self.fps
        #     velocity_scale = 0.2 + 0.8 * max(min(vxvy / ego_max_velocity, 1.0), 0.0)

        # ego的预测轨迹
        for i in range(ts, min(len(ego_polygon), ts + min_check_size + 1)):
            ego_center = np.mean(ego_polygon[i], axis=0)
            ego_pred.append(ego_center)
            ego_pred_plg.append(ego_polygon[i])

        # vru的预测轨迹
        start_idx = max(ts - vru_history_frame_save_num, 0)
        end_idx = min(len(dobjs_full), ts + min_check_size + 1)

        if vru_preds is None:
            vru_preds = {}
        if vru_preds_plg is None:
            vru_preds_plg = {}

        vru_preds_internal = vru_preds
        vru_preds_plg_internal = vru_preds_plg

        is_first_init = len(vru_preds_internal) == 0

        if is_first_init:
            for i in range(start_idx, end_idx):
                _, dobjs = dobjs_full[i]
                for j in range(len(dobjs)):
                    obj_id = dobjs[j, 9].astype(int)
                    if np.isin(obj_id, vru_ids):
                        vru_polygon = dobjs_polygon[i][j] # 4, 2
                        vru_center = np.mean(vru_polygon, axis=0) # 2,
                        if obj_id not in vru_preds_internal:
                            vru_preds_internal[obj_id] = []
                            vru_preds_plg_internal[obj_id] = []
                        vru_preds_internal[obj_id].append((i, vru_center))
                        vru_preds_plg_internal[obj_id].append((i, vru_polygon))
        else:
            for obj_id in list(vru_preds_internal.keys()):
                if obj_id in vru_preds_internal:
                    vru_preds_internal[obj_id] = [(t_idx, center) for t_idx, center in vru_preds_internal[obj_id] if t_idx >= start_idx]
                    if len(vru_preds_internal[obj_id]) == 0:
                        del vru_preds_internal[obj_id]
                if obj_id in vru_preds_plg_internal:
                    vru_preds_plg_internal[obj_id] = [(t_idx, plg) for t_idx, plg in vru_preds_plg_internal[obj_id] if t_idx >= start_idx]
                    if len(vru_preds_plg_internal[obj_id]) == 0:
                        del vru_preds_plg_internal[obj_id]

            max_t_idx = -1
            for obj_id in vru_preds_internal.keys():
                if len(vru_preds_internal[obj_id]) > 0:
                    obj_max_t_idx = max(t_idx for t_idx, _ in vru_preds_internal[obj_id])
                    max_t_idx = max(max_t_idx, obj_max_t_idx)

            new_start_idx = max(max_t_idx + 1, start_idx)
            for i in range(new_start_idx, end_idx):
                _, dobjs = dobjs_full[i]
                for j in range(len(dobjs)):
                    obj_id = dobjs[j, 9].astype(int)
                    dobjs_cid = dobjs[j, 7]
                    if dobjs_cid >= 300:
                        vru_polygon = dobjs_polygon[i][j] # 4, 2
                        vru_center = np.mean(vru_polygon, axis=0) # 2,
                        if obj_id not in vru_preds_internal:
                            vru_preds_internal[obj_id] = []
                            vru_preds_plg_internal[obj_id] = []
                        if len(vru_preds_internal[obj_id]) == 0 or vru_preds_internal[obj_id][-1][0] < i:
                            vru_preds_internal[obj_id].append((i, vru_center))
                            vru_preds_plg_internal[obj_id].append((i, vru_polygon))
                        elif vru_preds_internal[obj_id][-1][0] == i:
                            vru_preds_internal[obj_id][-1] = (i, vru_center)
                            vru_preds_plg_internal[obj_id][-1] = (i, vru_polygon)

        vru_preds_calc = {}
        vru_preds_plg_calc = {}
        for obj_id in vru_preds_internal.keys():
            if obj_id in vru_ids:
                # 只保留checksize内的数据
                filtered_items = [(t_idx, center) for t_idx, center in vru_preds_internal[obj_id] if start_idx <= t_idx < end_idx]
                vru_preds_calc[obj_id] = [center for _, center in filtered_items]
                if obj_id in vru_preds_plg_internal:
                    filtered_items_plg = [(t_idx, plg) for t_idx, plg in vru_preds_plg_internal[obj_id] if start_idx <= t_idx < end_idx]
                    vru_preds_plg_calc[obj_id] = [plg for _, plg in filtered_items_plg]

        # 遍历vru，计算碰撞时间
        caution_vru_ids = []
        for obj_id, vru_pred in vru_preds_calc.items():
            # 先通过总体判断ego和vru的预测轨迹是否有交点来筛选vru
            check_size = min(min(len(vru_pred), len(ego_pred)), min_check_size)
            if check_size < 2:
                continue
            ego_polyline = LineString(np.asarray(ego_pred[:check_size])[:, :2]).buffer(ego_width_buffer)
            vru_polyline = LineString(np.asarray(vru_pred[:check_size])[:, :2]).buffer(vru_width_buffer)
            if not ego_polyline.intersects(vru_polyline):
                continue

            # 计算自车到达交点的index
            ego_collision_index = check_size
            for i in range(0, check_size - 1):
                ego_segment = LineString([ego_pred[i], ego_pred[i + 1]]).buffer(ego_width_buffer)
                if ego_segment.intersects(vru_polyline):
                    ego_collision_index = i
                    break

            # 计算vru到达交点的index
            vru_collision_index = check_size
            for j in range(0, check_size - 1):
                vru_segment = LineString([vru_pred[j], vru_pred[j + 1]]).buffer(vru_width_buffer)
                if vru_segment.intersects(ego_polyline):
                    vru_collision_index = j
                    break

            # 归一化index，计算相对时间差
            ego_collision_index_norm = ego_collision_index / check_size
            vru_collision_index_norm = vru_collision_index / check_size
            delta_index_norm = abs(ego_collision_index_norm - vru_collision_index_norm)
            delta_index_norm = min(max(delta_index_norm, 0.0), 1.0)

            # 随着碰撞发生的时间变远，惩罚不断衰减
            if ego_collision_index >= check_size or vru_collision_index >= check_size:
                collision_time_coef = 0
            else:
                # collision_time_coef = np.exp(ego_collision_index_decay_coef * ego_collision_index / len(ego_polygon))
                collision_time_coef = 1 - ego_collision_index / len(ego_polygon)

            if ego_in_junction:
                    if ego_collision_index_norm <= vru_collision_index_norm - min_preempt_frame_threshold / check_size:
                        vru_plgs = vru_preds_plg_calc.get(int(obj_id), [])
                        if len(vru_plgs) > ego_collision_index + vru_history_frame_save_num and len(ego_pred_plg) > ego_collision_index:
                            ego_corners = ego_pred_plg[ego_collision_index]
                            vru_corners = vru_plgs[ego_collision_index + vru_history_frame_save_num]
                            if (len(ego_corners) >= 3 and len(vru_corners) >= 3 and
                                all(len(p) == 2 for p in ego_corners) and
                                all(len(p) == 2 for p in vru_corners)):
                                ego_plg = Polygon(ego_corners)
                                vru_plg = Polygon(vru_corners)
                                if ego_plg.is_valid and vru_plg.is_valid:
                                    distance = ego_plg.distance(vru_plg)
                                else:
                                    distance = 10
                            else:
                                distance = 10
                            if distance < min_distance:
                                delta_time_coef = 1 - delta_index_norm / 2
                            else:
                                delta_time_coef = (1 - delta_index_norm) / 2
                        else:
                            delta_time_coef = (1 - delta_index_norm) / 2
                    elif ego_collision_index_norm <= vru_collision_index_norm:
                        delta_time_coef = 1 - delta_index_norm / 2
                    else:
                        delta_time_coef = 1 - delta_index_norm
                    caution_vru_ids.append(obj_id)
                    cur_vru_penalty = delta_time_coef * collision_time_coef * max_penalty
            else:
                if ego_straight_mask:
                    cur_vru_penalty = (1 - delta_index_norm) * collision_time_coef * max_penalty / 1.5
                else:
                    # 如果ego比vru更早到达交点，应该给出更大的惩罚让自车减速
                    if ego_collision_index_norm <= vru_collision_index_norm - min_preempt_frame_threshold / check_size:
                        # coef = -4.0
                        delta_time_coef = (1 - delta_index_norm) / 2
                    else:
                        # coef = -3.0
                        delta_time_coef = 1 - delta_index_norm
                    cur_vru_penalty = delta_time_coef * collision_time_coef * max_penalty / 1.5

            max_vru_penalty, target_vru_id = max(
                (max_vru_penalty, target_vru_id), (cur_vru_penalty, obj_id), key=lambda x: x[0]
            )
        caution_vru_ids = list(caution_vru_ids)
        if target_vru_id is not None and len(caution_vru_ids) > 0:
            if np.isin(target_vru_id, caution_vru_ids):
                vru_caution = True
        return max_vru_penalty, target_vru_id, vru_caution

    def preprocess_cross(self, raw_env, ts, ignore_track_ids, ego_polygon, max_penalty, min_check_size, gt_polygon):
        max_cross_penalty = 0
        target_cross_id = None
        direct_light = -1
        if ts < 40:
            direct_light = judge_intersection_and_maneuver(ego_polygon[: ts + 1], raw_env, ts)
        else:
            direct_light = judge_intersection_and_maneuver(ego_polygon[ts - 40 : ts], raw_env, ts)
        path_turn_type = determine_path_turn_type(
            gt_polygon[::5].mean(axis=-2), threshold_degree=50, fuzzy_threshold_degree=30
        )

        if ((direct_light == 1) and (Turntype.LEFT in path_turn_type)) or (
            (direct_light == 2) and (Turntype.STRAIGHT in path_turn_type)
        ):
            ego_width_buffer = 1.2
            if direct_light == 1:
                cross_width_buffer = 1.0
                min_check_size *= 3.5
                min_check_size = max(int(min_check_size), 35)
            else:
                cross_width_buffer = 0.5
                min_check_size *= 2.5
                min_check_size = int(min_check_size)
        elif {Turntype.STRAIGHT} == path_turn_type:
            ego_width_buffer = 1.2
            cross_width_buffer = 0.5
            min_check_size *= 2.5
            min_check_size = int(min_check_size)
        else:
            return 0, None, False, direct_light

        cross_history_frame_save_num = 1

        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        _, dobjs = dobjs_full[ts]
        cross_preds = {}
        ego_pred = []

        # 过滤mask
        ignore_mask = np.isin(dobjs[:, 9].astype(int), ignore_track_ids)
        polygon = dobjs_polygon[ts][~ignore_mask].copy()
        dobjs = dobjs[~ignore_mask]

        # 前方mask
        egoplg_t = ego_polygon[ts]
        ego_center = np.mean(egoplg_t, axis=0, keepdims=True)
        rf = egoplg_t[1]  # 右前
        lf = egoplg_t[2]  # 左前
        fc = (rf + lf) * 0.5
        edge = lf - rf
        n = np.array([-edge[1], edge[0]])
        ego_ctr = ego_center.squeeze()
        if np.dot(n, (fc - ego_ctr)) < 0:
            n = -n
        n = n / (np.linalg.norm(n) + 1e-9)
        rel = polygon - fc
        signed = rel @ n
        front_mask = (signed > 0).any(axis=1)  # 任一角点在前方

        # 角度 mask
        obj_centers = polygon.mean(axis=1)
        rf_o = polygon[:, 1, :]
        lf_o = polygon[:, 2, :]
        fc_o = (rf_o + lf_o) * 0.5
        edge_o = lf_o - rf_o
        n_obj = np.stack([-edge_o[:, 1], edge_o[:, 0]], axis=1)
        forward_vec_o = fc_o - obj_centers
        flip = np.sum(n_obj * forward_vec_o, axis=1) < 0
        n_obj[flip] = -n_obj[flip]
        n_obj_unit = n_obj / (np.linalg.norm(n_obj, axis=1, keepdims=True) + 1e-9)
        cosang = np.clip(n_obj_unit @ n, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosang))
        cross_mask = (ang_deg > 30.0) & (ang_deg < 150.0)
        oncome_mask = ang_deg >= 150.0

        # 速度mask
        dobjs_vel_i = np.sqrt(dobjs[:, 10] ** 2 + dobjs[:, 11] ** 2)
        dyn_mask = dobjs_vel_i > 0.05

        # 车辆mask
        vehicle_mask = dobjs[:, 7].astype(int) < 300

        # 候选的 cross 目标
        if ((direct_light == 1) and (Turntype.LEFT in path_turn_type)) or (
            (direct_light == 2) and (Turntype.STRAIGHT in path_turn_type)
        ):
            valid_mask = vehicle_mask & dyn_mask & (cross_mask | oncome_mask) & front_mask
        else:
            valid_mask = vehicle_mask & dyn_mask & cross_mask & front_mask

        if not np.any(valid_mask):
            return 0, None, False, direct_light
        cross_ids = dobjs[valid_mask, 9].astype(int)

        # ego的预测轨迹
        end_idx = min(len(ego_polygon), ts + min_check_size + 1)
        ego_v_ts = np.linalg.norm(np.mean(ego_polygon[ts] - ego_polygon[ts - 1], axis=-2)) * self.fps
        slow_mode = (ego_v_ts < 5.0 / 3.6) and (ego_v_ts > 0.01)
        for i in range(ts, end_idx):
            poly = ego_polygon[i]
            ego_center = np.mean(poly[(1, 2), :], axis=0)
            ego_pred.append(ego_center)
            # 按 end_idx-1 时刻的速度和朝向往前多插若干个 ego_center
            if slow_mode and i == end_idx - 1:
                front_center = np.mean(poly[[1, 2], :], axis=0)
                rear_center = np.mean(poly[[0, 3], :], axis=0)
                heading_vec = front_center - rear_center
                norm = np.linalg.norm(heading_vec)

                if norm > 1e-6 and ego_v_ts > 0.01:
                    heading_unit = heading_vec / norm
                    ego_v_i = max(0.4, np.linalg.norm(np.mean(ego_polygon[i] - ego_polygon[i - 1], axis=-2)))
                    step = heading_unit * (ego_v_i / self.fps)
                    # 速度越低，额外插入的点越多
                    max_v = 5.0 / 3.6
                    n_extra = int((1 - np.clip(ego_v_ts / max_v, 0, 1)) * 10)
                    min_check_size += min(len(ego_polygon), end_idx + n_extra) - end_idx

                    extra_center = ego_center.copy()
                    for _ in range(n_extra):
                        extra_center = extra_center + step
                        ego_pred.append(extra_center.copy())
        if len(ego_pred) < 2:
            return 0, None, False, direct_light

        # cross的预测轨迹
        cross_start_idx = max(ts - cross_history_frame_save_num, 0)
        cross_end_idx = min(len(dobjs_full), ts + min_check_size + 1)
        for i in range(cross_start_idx, cross_end_idx):
            _, dobjs = dobjs_full[i]
            for j in range(len(dobjs)):
                obj_id = dobjs[j, 9].astype(int)
                if np.isin(obj_id, cross_ids):
                    cross_polygon = dobjs_polygon[i][j]  # 4, 2
                    cross_center = np.mean(cross_polygon, axis=0)  # 2,
                    if i == cross_start_idx:
                        cross_center = np.mean(cross_polygon[(0, 3), :], axis=0)
                    if i == cross_end_idx - 1:
                        cross_center = np.mean(cross_polygon[(1, 2), :], axis=0)
                    cross_preds.setdefault(int(obj_id), []).append(cross_center)

        # 遍历cross，计算碰撞时间
        for obj_id, cross_pred in cross_preds.items():
            # 先通过总体判断ego和cross的预测轨迹是否有交点来筛选cross
            check_size = min(min(len(cross_pred), len(ego_pred)), min_check_size)
            if check_size < 2:
                continue
            ego_polyline = LineString(np.asarray(ego_pred[:check_size])[:, :2]).buffer(ego_width_buffer)
            cross_polyline = LineString(np.asarray(cross_pred[:check_size])[:, :2]).buffer(cross_width_buffer)
            if not ego_polyline.intersects(cross_polyline):
                continue

            # 计算自车到达交点的index
            ego_collision_index = check_size
            for i in range(0, check_size - 1):
                ego_segment = LineString([ego_pred[i], ego_pred[i + 1]]).buffer(ego_width_buffer)
                if ego_segment.intersects(cross_polyline):
                    ego_collision_index = i
                    break

            # 计算cross到达交点的index
            cross_collision_index = check_size
            for j in range(0, check_size - 1):
                cross_segment = LineString([cross_pred[j], cross_pred[j + 1]]).buffer(cross_width_buffer)
                if cross_segment.intersects(ego_polyline):
                    cross_collision_index = j
                    break

            # 随着碰撞发生的时间变远，惩罚不断衰减
            if ego_collision_index == check_size or cross_collision_index == check_size:
                collision_time_coef = 1
            else:
                collision_time_coef = ego_collision_index / check_size

            # 归一化index，计算相对时间差
            ego_collision_index_norm = ego_collision_index / check_size
            cross_collision_index_norm = cross_collision_index / check_size
            delta_index_norm = abs(ego_collision_index_norm - cross_collision_index_norm)
            delta_index_norm = min(max(delta_index_norm, 0.0), 1.0)

            cur_cross_penalty = (1.0 - delta_index_norm) * (1.0 - collision_time_coef) * max_penalty / 1.5
            max_cross_penalty, target_cross_id = max(
                (max_cross_penalty, target_cross_id), (cur_cross_penalty, obj_id), key=lambda x: x[0]
            )

        turn_left_cross_flag = False
        straight_cross_flag = False
        if direct_light == 1 and max_cross_penalty > 0:
            turn_left_cross_flag = True
        elif (direct_light == 2 and max_cross_penalty > 0) or \
            ({Turntype.STRAIGHT} == path_turn_type and max_cross_penalty > 0):
            straight_cross_flag = True

        return max_cross_penalty, target_cross_id, (turn_left_cross_flag, straight_cross_flag), direct_light

    def has_cutin_car(self ,polygons, dobjs_full, check_size_agt, ignore_track_ids, ts ,dobjs_polygon, ego_polygon, raw_env,vxvy, truck_max_penalty):
        # 做自车的速度隔离， 自车30kph以下 不进入过滤
        if vxvy < 8.33:
            return False
        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]

        cur_raw_env = raw_env[ts]
        lane_lines = cur_raw_env["lane_lines"]

        ego_straight_mask = True
        for pt , cur_index in lane_lines:
            polyline = LineString(pt[:, :2])
            cur_ego_polygon = Polygon(ego_polygon[ts])
            if polyline.intersects(cur_ego_polygon):
                ego_straight_mask = False
                break


        # 动态（多帧窗口）
        t_end_agt = min(ts + check_size_agt, len(dobjs_full))
        has_cutin_car_flag = False
        for i in range(ts, t_end_agt):
            _, dobjs_i = dobjs_full[i]
            tids_i = dobjs_i[:, 9].astype(int)
            keep = ~np.isin(tids_i, list(ignore_track_ids))

            vx_i = dobjs_i[:, 10]
            vy_i = dobjs_i[:, 11]
            yaw_i = dobjs_i[:, 6]
            cid_i = dobjs_i[:, 7]
            obs_speed_i = np.sqrt(vx_i**2 + vy_i**2)

            ego_center = np.mean(ego_polygon[ts], axis=0, keepdims=True)


            ## 判断ego前一定区域内的前车
            if i < len(ego_polygon):
                egoplg_t = ego_polygon[i]
            else:
                egoplg_t = ego_polygon[-1]
            dobjs_center = dobjs_i[:,0:2]
            rea_center = np.mean(egoplg_t[[0, 3]], axis=0, keepdims=True)
            ego_direct = rea_center - ego_center
            ego_direct = ego_direct / np.linalg.norm(ego_direct, axis=1, keepdims=True)
            dobjs_direct = dobjs_center - ego_center

            agent_at_ego_front = None
            agent_at_ego_front = np.logical_and(dobjs_direct[:, 0] > 0, dobjs_direct[..., 0] < 40)
            is_truck =  (dobjs_i[:, 3] > 6)  # 车长大于6米
            front_truck = np.logical_and(agent_at_ego_front, abs(dobjs_direct[:, 1]) < 4.5) & is_truck
            front_car = np.logical_and(agent_at_ego_front, abs(dobjs_direct[:, 1]) < 3)


            egoplg_t = ego_polygon[i] if i < len(ego_polygon) else ego_polygon[-1]
            rf = egoplg_t[1]   # 右前
            lf = egoplg_t[2]   # 左前
            fc = (rf + lf) * 0.5  # 前中心，用作分界点

            # 构造“向前”的法向量：取前沿边(右前->左前)的法向，并与(前中心-自车中心)同向
            edge = lf - rf                         # 前沿边向量
            n = np.array([-edge[1], edge[0]])      # 其一侧法向
            ego_ctr = ego_center.squeeze()         # (2,)
            if np.dot(n, (fc - ego_ctr)) < 0:
                n = -n
            n = n / (np.linalg.norm(n) + 1e-9)     # 归一化

            # 计算每个目标多边形四点到“前方法向”的有符号投影
            obj_pts = dobjs_polygon[i]             # (N,4,2)
            rel = obj_pts - fc                     # (N,4,2)
            # signed = rel @ n                       # (N,4)  每个角点在法向上的符号距离

            # 判断目标是不是4个角点位于ego的同一侧，并且其中一个角点横向距离自车很近

            hori_rel = rel[...,1]
            obs_in_left_truck = (hori_rel > -1).all(axis=1) & is_truck  # 4个点在同一侧
            obs_in_left_car = (hori_rel > 0).all(axis=1)   # 4个点在同一侧
            obs_in_left_corner_constraint = ((hori_rel > -1) & (hori_rel < 1.4)).any(axis=1) #其中一个点在
            obs_in_right_truck = (hori_rel < 1).all(axis=1) & is_truck
            obs_in_right_car = (hori_rel < 0).all(axis=1)
            obs_in_right_corner_constraint = ((hori_rel > -1.4) & (hori_rel < 1)).any(axis=1)
            # obs 速度大于 自车2 m/s 以上则不考虑
            obs_spd_mask_ts = obs_speed_i < (vxvy + 2)

            # 目标层面：heading， 角点约束， 速度 > 4m/s 过滤 误绕行的case
            heading_cut_in_right = (yaw_i > 0.174) & (yaw_i < 0.785) & (obs_in_right_car | obs_in_right_truck) & obs_in_right_corner_constraint & (obs_speed_i > 4) & obs_spd_mask_ts
            heading_cut_in_left = (yaw_i < -0.174) & (yaw_i > -0.785) & (obs_in_left_car | obs_in_left_truck) & obs_in_left_corner_constraint & (obs_speed_i > 4 ) & obs_spd_mask_ts

            cid_mask = (cid_i < 300) # 只管veh

            cut_in_obs_right_mask = (front_truck | front_car) * heading_cut_in_right * cid_mask * ego_straight_mask
            cut_in_obs_left_mask = (front_truck | front_car)  * heading_cut_in_left * cid_mask * ego_straight_mask

            tids_cut_in_padding_left = tids_i[keep * cut_in_obs_right_mask]    #从右侧cut in padding 左侧
            tids_cut_in_padding_right = tids_i[keep * cut_in_obs_left_mask]    #从左侧cut in padding 右侧

            if tids_cut_in_padding_left.size > 0 or tids_cut_in_padding_right.size > 0:
                has_cutin_car_flag = True
        return has_cutin_car_flag
    def _is_ego_straight(self, ego_polygon, raw_env, min_check_size, ts):
        """
        判断自车是否为直行
        """

        ego_traj = np.mean(ego_polygon[ts:min(ts + min_check_size, len(ego_polygon))], axis=1)
        if len(ego_traj) < 3:
            return False

        # 1.用是否压线判断
        cur_raw_env = raw_env[ts]
        lane_lines = cur_raw_env["lane_lines"]
        ego_not_crossing_line = True
        for pt, cur_index in lane_lines:
            if len(pt) < 2:
                continue
            polyline = LineString(pt[:, :2])
            for i in range(ts, min(min_check_size + ts, len(ego_polygon))):
                if polyline.intersects(Polygon(ego_polygon[i])):
                    ego_not_crossing_line = False
                    break
            if ego_not_crossing_line == False:
                break

        # 2.用航向角判断
        angle_threshold = math.pi / 18
        straight_ratio_threshold = 0.85
        yaw_sequence = []
        for i in range(len(ego_traj) - 1):
            dx = ego_traj[i + 1][0] - ego_traj[i][0]
            dy = ego_traj[i + 1][1] - ego_traj[i][1]
            yaw = math.atan2(dy, dx)
            yaw_sequence.append(yaw)
        if len(yaw_sequence) < 2:
            return False

        yaw_changes = []
        point_states = [True]
        for i in range(len(yaw_sequence) - 1):
            dyaw = yaw_sequence[i + 1] - yaw_sequence[i]
            while dyaw > math.pi:
                dyaw -= 2 * math.pi
            while dyaw < -math.pi:
                dyaw += 2 * math.pi
            yaw_changes.append(abs(dyaw))
            # 判断单个点是否直行
            is_straight_point = abs(dyaw) < angle_threshold
            point_states.append(is_straight_point)

        # 计算平均航向角变化率
        avg_yaw_change = np.mean(yaw_changes) if yaw_changes else 0.0

        # 判断整体是否为直行状态
        # 如果超过80%的点都处于直行状态，则认为整体是直行
        straight_ratio = sum(point_states) / len(point_states)
        overall_straight = straight_ratio > straight_ratio_threshold and avg_yaw_change < angle_threshold

        # 3.老方法
        ego_cur_center = np.mean(ego_polygon[ts], axis=0)
        ego_straight_threshold = 1.0
        ego_straight_mask = (
                abs(
                    ego_cur_center[1]
                    - np.mean(ego_traj[:, 1], axis=0)
                )
                < ego_straight_threshold
            )

        # 4.如果没压线或航向角变化在阈值内，则为直行
        is_ego_straight = ego_straight_mask or overall_straight or ego_not_crossing_line
        return is_ego_straight
    def preprocess_encounter(self, raw_env, ts, ignore_track_ids, ego_polygon, max_penalty, min_check_size):

        max_encounter_penalty = 0
        target_encounter_id = None

        ego_width_buffer = 1.2
        encounter_width_buffer = 1.4

        encounter_history_frame_save_num = int(4 * self.fps)

        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        _, dobjs = dobjs_full[ts]
        encounter_preds = {}
        ego_pred = []

        # 过滤mask
        ignore_mask = np.isin(dobjs[:, 9].astype(int), ignore_track_ids)
        polygon = dobjs_polygon[ts][~ignore_mask].copy()
        dobjs = dobjs[~ignore_mask]

        # 前方mask
        egoplg_t = ego_polygon[ts]
        ego_center = np.mean(egoplg_t, axis=0, keepdims=True)
        rf = egoplg_t[1]           # 右前
        lf = egoplg_t[2]           # 左前
        fc = (rf + lf) * 0.5
        edge = lf - rf
        n = np.array([-edge[1], edge[0]])
        ego_ctr = ego_center.squeeze()
        if np.dot(n, (fc - ego_ctr)) < 0:
            n = -n
        n = n / (np.linalg.norm(n) + 1e-9)
        rel = polygon - fc
        signed = rel @ n
        front_mask = (signed > 0).any(axis=1)     # 任一角点在前方

        # 角度 mask 150到-150度
        obj_centers = polygon.mean(axis=1)
        rf_o = polygon[:, 1, :]
        lf_o = polygon[:, 2, :]
        fc_o = (rf_o + lf_o) * 0.5
        edge_o = lf_o - rf_o
        n_obj = np.stack([-edge_o[:, 1], edge_o[:, 0]], axis=1)
        forward_vec_o = fc_o - obj_centers
        flip = (np.sum(n_obj * forward_vec_o, axis=1) < 0)
        n_obj[flip] = -n_obj[flip]
        n_obj_unit = n_obj / (np.linalg.norm(n_obj, axis=1, keepdims=True) + 1e-9)
        cosang = np.clip(n_obj_unit @ n, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cosang))
        encounter_mask = ang_deg >= 150.0

        dobjs_vx = dobjs[:, 10]  # 纵向速度vx
        velocity_mask = dobjs_vx < -1.0

        # 车辆mask
        vehicle_mask = dobjs[:, 7].astype(int) < 300

        # 左侧mask
        hori_rel = rel[...,1]
        # obs_in_left = (hori_rel > 0).all(axis=1)
        obs_in_left = (hori_rel > 0).sum(axis=1) >= 3

        # 候选的会车目标
        valid_mask = (vehicle_mask & velocity_mask & encounter_mask & front_mask & obs_in_left)

        if not np.any(valid_mask):
            return 0, None
        encounter_ids = dobjs[valid_mask, 9].astype(int)

        for i in range(ts, min(len(ego_polygon), ts + min_check_size + 1)):
            ego_center = np.mean(ego_polygon[i][(1, 2), :], axis=0)
            ego_pred.append(ego_center)
        if len(ego_pred) < 2:
            return 0, None

        encounter_start_idx = max(ts - encounter_history_frame_save_num, 0)
        encounter_end_idx = min(len(dobjs_full), ts + min_check_size + 1)
        for i in range(encounter_start_idx, encounter_end_idx):
            _, dobjs = dobjs_full[i]
            for j in range(len(dobjs)):
                obj_id = dobjs[j, 9].astype(int)
                if np.isin(obj_id, encounter_ids):
                    encounter_polygon = dobjs_polygon[i][j] # 4, 2
                    encounter_center = np.mean(encounter_polygon, axis=0) # 2,
                    if i == encounter_start_idx:
                        encounter_center = np.mean(encounter_polygon[(1, 2), :], axis=0)
                    if i == encounter_end_idx - 1:
                        encounter_center = np.mean(encounter_polygon[(0, 3), :], axis=0)
                    encounter_preds.setdefault(int(obj_id), []).append(encounter_center)

        for obj_id, encounter_pred in encounter_preds.items():
            check_size = min(min(len(encounter_pred), len(ego_pred)), min_check_size)
            if check_size < 2:
                continue
            ego_polyline = LineString(np.asarray(ego_pred[:check_size])[:, :2]).buffer(ego_width_buffer)
            encounter_polyline = LineString(np.asarray(encounter_pred[:check_size])[:, :2]).buffer(encounter_width_buffer)
            if not ego_polyline.intersects(encounter_polyline):
                continue

            # 计算自车到达交点的index
            ego_collision_index = check_size
            for i in range(0, check_size - 1):
                ego_segment = LineString([ego_pred[i], ego_pred[i + 1]]).buffer(ego_width_buffer)
                if ego_segment.intersects(encounter_polyline):
                    ego_collision_index = i
                    break

            # 计算encounter到达交点的index
            encounter_collision_index = len(encounter_pred)
            for j in range(0, len(encounter_pred) - 1):
                encounter_segment = LineString([encounter_pred[j], encounter_pred[j + 1]]).buffer(encounter_width_buffer)
                if encounter_segment.intersects(ego_polyline):
                    encounter_collision_index = j
                    break

            # 随着碰撞发生的时间变远，惩罚不断衰减
            if ego_collision_index == check_size or encounter_collision_index == len(encounter_pred):
                collision_time_coef = 1
            else:
                collision_time_coef = ego_collision_index / check_size

            # 归一化index，计算相对时间差
            ego_collision_index_norm = ego_collision_index / check_size
            encounter_collision_index_norm = encounter_collision_index / len(encounter_pred)
            delta_index_norm = abs(ego_collision_index_norm - encounter_collision_index_norm)
            delta_index_norm = min(max(delta_index_norm, 0.0), 1.0)

            cur_encounter_penalty = (1.0 - delta_index_norm) * (1.0 - collision_time_coef) * max_penalty
            max_encounter_penalty, target_encounter_id = max(
                (max_encounter_penalty, target_encounter_id),
                (cur_encounter_penalty, obj_id),
                key=lambda x: x[0]
            )
        return max_encounter_penalty, target_encounter_id

    def forward(
        self,
        gt_polygon,
        ego_polygon,
        raw_env,
        ts,
        rear_collision_tids,
        max_penalty=5.0,
        min_distance=0.3,
        max_distance=0.8,
        ego_max_velocity=-1.0,
        ego_check_seconds=1.2,
        agt_check_seconds=1.2,
        ignore_rear_obj=False,
        bypass_flag=False,
        vru_preds=None,
        vru_preds_plg=None,
    ):
        """
        检查ego_check_seconds时间内的规划轨迹，与agt_check_seconds时间内的任意目标，是否会发生碰撞.

        Args:
            ego_max_velocity: 当ego_max_velocity大于0时，惩罚会根据自车速度衰减
        """
        # 车速越小惩罚越小
        min_speed = 3.0
        max_speed = 20.0
        velocity_scale = 1.0
        vxvy = np.mean(ego_polygon[ts], axis=0) - np.mean(ego_polygon[ts - 1], axis=0)
        vxvy = math.sqrt(vxvy[0] ** 2 + vxvy[1] ** 2) * self.fps
        if max_speed > 0:
            velocity_scale = 0.5 + 0.5 * min(max((vxvy - min_speed) / (5.55 - min_speed), 0.0), 1.0)

        # ego speed aware ttc
        max_padding_ttc = 2.0
        ego_check_seconds_speed_aware = (
            min(max((vxvy - min_speed) / (max_speed - min_speed), 0.0), 1.0) * max_padding_ttc + ego_check_seconds
        )

        # ego speed aware dist
        min_distance = 0.2
        max_distance = min(max(vxvy / 20, 0.0), 1.0) * 0.3 + 0.3

        # 当前帧的curb
        road_edge = raw_env[ts]["road_edge"]
        nomap_road_edge = raw_env[ts]["nomap_road_edge"]
        shape_lines = MultiLineString([p[:, :2] for p in (road_edge + nomap_road_edge)])

        # 当前帧的静态目标
        # polygons = [raw_env[ts]["sobjs_polygon"]]
        _, polygons = filter_opened_gate_sobj(raw_env[ts])
        polygons = [polygons]

        # 时间窗口内的动态目标
        check_size = int(agt_check_seconds * self.fps)
        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        ignore_track_ids = rear_collision_tids.copy()

        """
                    ^
                    │
                2———│———1
                │   │   │
        <——————————ego————————————
                │   │   │
                3———│———0
                    │
        后向60度以内的目标忽略，cos(60°)=0.5
        距离0.8m以内的目标忽略，意味着已经碰撞
        """
        objplg = dobjs_polygon[ts]
        if ignore_rear_obj and len(objplg) > 0:
            egoplg = ego_polygon[ts]
            ego_center = np.mean(egoplg, axis=0, keepdims=True)
            rea_center = np.mean(egoplg[[0, 3]], axis=0, keepdims=True)
            obj_center = np.mean(objplg, axis=1)
            ego_direct = rea_center - ego_center
            ego_direct = ego_direct / np.linalg.norm(ego_direct, axis=1, keepdims=True)

            obj_vertices_direct = objplg - ego_center
            obj_vertices_norm = np.linalg.norm(obj_vertices_direct, axis=2, keepdims=True)
            obj_vertices_direct = obj_vertices_direct / (obj_vertices_norm + 1e-8)
            cos_direct_vertices = (obj_vertices_direct @ ego_direct.T).squeeze(-1)
            cos_direct = np.all(cos_direct_vertices > 0.5, axis=1)
            obj_direct = obj_center - ego_center
            dst_center = np.linalg.norm(obj_direct, axis=1, keepdims=True)
            obj_direct = obj_direct / dst_center
            cos_direct = (obj_direct @ ego_direct.T).flatten()
            dst_center = dst_center.flatten()
            ignore_mask = (cos_direct > 0.5) | (dst_center < 0.8)

            # 忽略自车直行时，前方速度比自车快的目标
            ## 直行判断
            ego_traj = np.mean(ego_polygon, axis=1, keepdims=True)

            ego_straight_mask = self._is_ego_straight(ego_polygon, raw_env, 18, ts)
            ## 判断ego前车
            dobjs_direct = obj_center - ego_center
            agent_at_ego_front = None
            agent_at_ego_front = np.logical_and(dobjs_direct[:, 0] > 0, dobjs_direct[..., 0] < 40)
            agent_at_ego_front = np.logical_and(agent_at_ego_front, abs(dobjs_direct[:, 1]) < 3)  ## @zhd251118 从1m改成3m
            ##角度计算
            dobjs_vec = objplg[:, 1] - objplg[:, 0]
            dobjs_vec_norm = np.linalg.norm(dobjs_vec, axis=1, keepdims=True)
            dobjs_vec = dobjs_vec / dobjs_vec_norm
            cos_direct = abs(dobjs_vec @ ego_direct.T).flatten()
            cos_direct_mask = cos_direct > 0.9397
            ##速度判断
            _, dobjs = dobjs_full[ts]
            dobjs_speed_x = dobjs[:, 10]
            ego_vx, _ = (np.mean(egoplg, axis=0) - np.mean(ego_polygon[ts - 1], axis=0)) * self.fps

            dobjs_speed_mask = (dobjs_speed_x > ego_vx + 1) & (ego_vx > 10.55)
            ##路口隔离
            direct_light = -1
            if ts < 15:
                direct_light = judge_intersection_and_maneuver(ego_polygon[:ts + 1], raw_env, ts)
            else:
                direct_light = judge_intersection_and_maneuver(ego_polygon[ts - 15: ts], raw_env, ts)

            intersection_mask = ~np.isnan(direct_light)

            front_dobjs_mask = ego_straight_mask * agent_at_ego_front * cos_direct_mask * dobjs_speed_mask * (~intersection_mask)
            ignore_mask = np.logical_or(ignore_mask, front_dobjs_mask)

            if np.any(ignore_mask):
                _, dobjs = dobjs_full[ts]
                ignore_track_ids.update(dobjs[ignore_mask, 9].astype(int).tolist())
        ignore_track_ids = list(ignore_track_ids)

        id2rela_vel = self._calc_rela_vel_dict(raw_env, ego_polygon, ts)

        # junction scene exclude conditions
        is_junction_scenario = False
        is_main_to_aux_scenario = False
        is_narrow_road_scenario = False

        manner_scene_list = raw_env[ts].get("manner_scene_info", None)
        link_info = raw_env[ts].get("link_info", None)
        if link_info:
            if link_info[0] is not None:
                is_narrow_road_scenario = link_info[0].get("road_class", 0) >= 7

        if manner_scene_list is not None:
            manner_scene_list.sort(key=lambda x: x["distance_to_entry"])
        for scene in manner_scene_list:
            distance_to_entry = scene["distance_to_entry"]
            distance_to_exit = scene["distance_to_exit"]
            type = scene["scene_type"]

            if type == 2 and distance_to_entry < 150.0 and distance_to_exit > -50.0:
                is_junction_scenario = True
                break

            if type == 5 and distance_to_entry < 150.0 and distance_to_exit > -50.0:
                is_main_to_aux_scenario = True
                break

        # if not (is_junction_scenario or is_main_to_aux_scenario):
        if is_narrow_road_scenario:
            max_distance_mindist_sta = 0.05
        else:
            max_distance_mindist_sta = 0.8

        max_cross_penalty, cross_id, cross_flag, direct_light = self.preprocess_cross(
            raw_env=raw_env,
            ts=ts,
            ignore_track_ids=ignore_track_ids,
            ego_polygon=ego_polygon,
            max_penalty=max_penalty,
            min_check_size=int(ego_check_seconds_speed_aware * self.fps),
            gt_polygon=gt_polygon,
        )

        max_vru_penalty, vru_id, vru_caution = self._preprocess_cross_vru(
            raw_env=raw_env,
            ts=ts,
            ignore_track_ids=ignore_track_ids,
            ego_polygon=ego_polygon,
            id2rela_vel=id2rela_vel,
            max_penalty=max_penalty,
            direct_light=direct_light,
            bypass_flag=bypass_flag,
            vru_preds=vru_preds,
            vru_preds_plg=vru_preds_plg,
        )

        max_encounter_penalty = 0
        if is_narrow_road_scenario:
            max_encounter_penalty, encounter_id = self.preprocess_encounter(
                raw_env=raw_env,
                ts=ts,
                ignore_track_ids=ignore_track_ids,
                ego_polygon=ego_polygon,
                max_penalty=max_penalty,
                min_check_size=int(ego_check_seconds_speed_aware * self.fps) * 2,
            )

        # # 选取动态目标
        # for i in range(ts, min(ts + check_size, len(dobjs_full))):
        #     _, dobjs = dobjs_full[i]
        #     ignore_mask = np.isin(dobjs[:, 9].astype(int), ignore_track_ids)
        #     polygon = dobjs_polygon[i][~ignore_mask]
        #     polygons.append(polygon)

        # # 汇总所有需要check的目标
        # polygons = np.concatenate(polygons, axis=0)
        # shape_polygons = MultiPolygon([Polygon(p) for p in polygons])

        # 当前帧GOD polygon
        god_polygon = raw_env[ts]["god_polygon"]
        god_shape_lines = MultiLineString([p[:, :2] for p in god_polygon])

        # 计算是否碰撞
        check_size = int(ego_check_seconds_speed_aware * self.fps)
        min_ts, min_dist = 0, float("inf")
        ttc_sta_flag = False
        for i in range(ts, min(ts + check_size, len(ego_polygon))):
            _, dobjs = dobjs_full[i]
            ignore_mask = np.isin(dobjs[:, 9].astype(int), ignore_track_ids)
            polygon = dobjs_polygon[i][~ignore_mask]
            shape_polygons = MultiPolygon([Polygon(p) for p in polygon])

            ego_poly = Polygon(ego_polygon[i])
            line_distance = get_distance(ego_poly, shape_lines)
            polygon_distance = get_distance(ego_poly, shape_polygons)
            god_polygon_distance = get_distance(ego_poly, god_shape_lines)
            dist = min(line_distance, polygon_distance, god_polygon_distance)
            if dist < min_dist:
                if line_distance < polygon_distance:
                    ttc_sta_flag = True
                min_ts, min_dist = i, dist
                if min_dist < min_distance:
                    break
        if min_dist > max_distance:
            ttc_penalty = 0
        else:
            # 距离越远惩罚越小
            distance_scale = (max_distance - min_dist) / (max_distance - min_distance)
            distance_scale = max(min(distance_scale, 1.0), 0.0)

            # 时间越后惩罚越小
            time_scale = 1.0 - (min_ts - ts) / check_size
            time_scale = max(min(time_scale, 1.0), 0.0)

            ttc_penalty = max_penalty * distance_scale * time_scale
        if ttc_sta_flag:
            ttc_penalty *= 0.05
        if max_vru_penalty > ttc_penalty and vru_id is not None:
            ttc_penalty = max_vru_penalty
        if max_cross_penalty > ttc_penalty and cross_id is not None:
            ttc_penalty = max_cross_penalty

        has_cutin_car_flag = self.has_cutin_car(polygons, dobjs_full, check_size, ignore_track_ids ,ts ,dobjs_polygon, ego_polygon, raw_env, vxvy ,max_penalty)

        if max_encounter_penalty > ttc_penalty and encounter_id is not None:
            ttc_penalty = max_encounter_penalty
        return ttc_penalty, velocity_scale, has_cutin_car_flag, max_encounter_penalty, max_distance_mindist_sta, cross_flag, vru_caution

class MinDistReward(BaseLoss):
    def find_local_minima(self, a):
        """找到 1D ndarray 中的局部极小值。对于平坦极小区间，返回该区间最后一个位置的索引.

        参数
        ----
        a : ndarray
            输入的一维数组

        返回
        ----
        minima_indices : list of int
            局部极小值的索引
        minima_values : list
            对应的极小值
        """
        a = np.asarray(a)
        n = a.size
        if n == 0:
            return [], []

        # 计算相邻元素差异，定位相等值区间的边界
        diffs = np.diff(a)
        run_breaks = np.where(diffs != 0)[0]
        run_starts = np.concatenate(([0], run_breaks + 1))
        run_ends = np.concatenate((run_breaks, [n - 1]))  # 区间结束（包含）

        minima_indices = []

        for start, end in zip(run_starts, run_ends):
            val = a[start]
            # 边界处用 +inf 填充，保证边缘元素也可成为极小值
            left = a[start - 1] if start > 0 else np.inf
            right = a[end + 1] if end < n - 1 else np.inf

            # 严格小于两侧邻居
            if val < left and val < right:
                minima_indices.append(end)

        minima_values = a[minima_indices]
        return minima_indices, minima_values

    def find_local_minima_static(self, a, sta_min_distance_start_edge_flag):
        """找到 1D ndarray 中的局部极小值。对于平坦极小区间，返回该区间最后一个位置的索引.

        参数
        ----
        a : ndarray
            输入的一维数组

        返回
        ----
        minima_indices : list of int
            局部极小值的索引
        minima_values : list
            对应的极小值
        """
        a = np.asarray(a)
        n = a.size
        if n == 0:
            return [], []

        # 计算相邻元素差异，定位相等值区间的边界
        diffs = np.diff(a)
        run_breaks = np.where(diffs != 0)[0]
        run_starts = np.concatenate(([0], run_breaks + 1))
        run_ends = np.concatenate((run_breaks, [n - 1]))  # 区间结束（包含）

        minima_indices = []

        for start, end in zip(run_starts, run_ends):
            val = a[start]
            # 边界处用 +inf 填充，保证边缘元素也可成为极小值
            left = a[start - 1] if start > 0 else np.inf
            right = a[end + 1] if end < n - 1 else np.inf

            for i in range(start, end + 1):
                if sta_min_distance_start_edge_flag[i] and val < 0.8:
                    minima_indices.append(i)
            # 严格小于两侧邻居
            if val < left and val < right:
                ##如果是距离路沿的顶点过近，则需要保留区间的所有索引
                ##否则保存区间的起点
                if len(minima_indices) <= 0:
                    minima_indices.append(start)

        minima_values = a[minima_indices]
        return minima_indices, minima_values

    def forward(self,
        sta_min_distance_list,
        sta_min_distance_start_edge_flag_list,
        dyn_min_distance_list,
        sta_ttc_max_distance_list,
        min_distance_reward,
        sta_min_distance_reward,
        dyn_min_distance_reward,
        velocity_scale_list
    ):
        """Forward."""
        # min_distance = np.array(min_distance_list)
        # min_distance[min_distance >= 0.8] = 1.5
        # minima_indices, minima_values = self.find_local_minima(min_distance)
        # for ind, min_dis in zip(minima_indices, minima_values):
        sta_min_distance = np.array(sta_min_distance_list)
        sta_min_distance_start_edge_flag = np.array(sta_min_distance_start_edge_flag_list)
        dyn_min_distance = np.array(dyn_min_distance_list)

        sta_min_distance[sta_min_distance >= 0.8] = 1.5
        dyn_min_distance[dyn_min_distance >= 0.8] = 1.5

        sta_minima_indices, sta_minima_values = self.find_local_minima_static(
            sta_min_distance, sta_min_distance_start_edge_flag
        )
        dyn_minima_indices, dyn_minima_values = self.find_local_minima(dyn_min_distance)
        for ind, min_dis in zip(sta_minima_indices, sta_minima_values):
            if min_dis == 0 or ind == 0:
                continue
            # dis_penalty = max(0, 5 - 6.25 * min_dis)
            dis_penalty = max(0, 5 - 5 / (sta_ttc_max_distance_list[ind + 1] + 1e-6) * min_dis)
            dis_penalty *= 0.5
            dis_penalty *= velocity_scale_list[ind + 1]
            dis_penalty = min(dis_penalty, 2.5)
            sta_min_distance_reward[ind + 1] = -dis_penalty
        for ind, min_dis in zip(dyn_minima_indices, dyn_minima_values):
            if min_dis == 0 or ind == 0:
                continue
            dis_penalty = max(0, 5 - 6.25 * min_dis)
            # dis_penalty *= velocity_scale_list[ind + 1]
            # dis_penalty *= 1.5
            # dis_penalty = min(dis_penalty, 5)
            # min_distance_reward[ind + 1] = -dis_penalty
            dyn_min_distance_reward[ind + 1] = -dis_penalty

        choose_dyn = dyn_min_distance_reward < sta_min_distance_reward
        min_distance_reward[:] = np.where(choose_dyn, dyn_min_distance_reward, sta_min_distance_reward)
        sta_min_distance_reward[choose_dyn] = 0
        dyn_min_distance_reward[~choose_dyn] = 0

        return min_distance_reward, sta_min_distance_reward, dyn_min_distance_reward

        ### 原版
        # min_distance = np.array(min_distance_list)
        # min_distance[min_distance >= 0.8] = 1.5
        # minima_indices, minima_values = self.find_local_minima(min_distance)
        # for ind, min_dis in zip(minima_indices, minima_values):
        #     if min_dis == 0 or ind == 0:
        #         continue
        #     dis_penalty = max(0, 5 - 6.25 * min_dis)
        #     dis_penalty *= velocity_scale_list[ind + 1]
        #     min_distance_reward[ind + 1] = -dis_penalty
        # return min_distance_reward


class CrossSolidLineReward(BaseLoss):
    def check_intersection_with_lines(self, pred_ego_polygon, raw_env):
        """Check intersection with lines.

        Args:
            pred_ego_polygon: n x 4 x 2  自车预测位置polygon
            raw_env
        Returns:
            indecies: 与车道线有交互的自车polygon的索引, list
            attr: 交互的车道线属性, map {indecies: attr}
        """
        indecies = []
        attr = {}
        lane_lines = raw_env["lane_lines"]  # list

        for i, polygon in enumerate(pred_ego_polygon):
            filter_lines = []
            ego_center_pt = polygon.mean(axis=0).reshape(1, 2)

            for p, a in lane_lines:
                mask = np.linalg.norm(p[:, :2] - ego_center_pt, axis=1) < 5.0
                filter_pt = p[mask, :]
                filter_lines.append([p[mask, :], a[mask, :]])

            for filter_pt, filter_attr in filter_lines:
                if filter_pt.shape[0] < 2:
                    continue
                polyline = LineString(filter_pt[:, :2])
                ego_polygon = Polygon(polygon)
                if polyline.intersects(ego_polygon):
                    indecies.append(i)
                    attr.update({i: filter_attr[filter_attr.shape[0] // 2, :]})  # 取中点
                    break

        return indecies, attr

    def forward(self, pred_polygon, cur_env, pred_ts, lane_nr_remain_distance, continue_intersection_with_lines_frames):
        """计算压线Reward."""
        pred_ego_polygon_index, intersection_line_attr = self.check_intersection_with_lines(
            np.expand_dims(pred_polygon, 0), cur_env
        )
        line_penalty = 0
        if pred_ego_polygon_index:
            continue_intersection_with_lines_frames += 1
            t, c = intersection_line_attr[pred_ego_polygon_index[0]]
            # if 0 <= lane_nr_remain_distance < 50:
            #     line_penalty += 2
            if continue_intersection_with_lines_frames >= 10:
                if t in [1, 5, 7, 8]:
                    line_penalty += 1
        else:
            continue_intersection_with_lines_frames = 0
        return -line_penalty, continue_intersection_with_lines_frames

class ToggleReward(BaseLoss):

    def calculate_signed_lateral_distance(self, polygon, point, line):
        """
        计算带符号的横向距离
        正数：点在车道线右侧
        负数：点在车道线左侧
        """
        if isinstance(line, LineString):
            line_coords = np.array(line.coords)
        else:
            line_coords = line

        if len(line_coords) < 2:
            return float('inf')

        # 找到最近的线段
        min_distance = float('inf')
        closest_segment_index = -1
        projected_point = None

        for i in range(len(line_coords) - 1):
            p1 = line_coords[i]
            p2 = line_coords[i + 1]
            dist, proj = self.point_to_line_segment_distance(point, p1, p2)

            if dist < min_distance:
                min_distance = dist
                closest_segment_index = i
                projected_point = proj

        if closest_segment_index == -1:
            return min_distance

        # 获取最近线段的向量方向
        p1 = line_coords[closest_segment_index]
        p2 = line_coords[closest_segment_index + 1]
        line_direction = p2 - p1
        line_direction = line_direction / np.linalg.norm(line_direction)  # 单位化

        # 计算法向量（垂直于车道线方向）
        normal_vector = np.array([-line_direction[1], line_direction[0]])

        # 计算点到投影点的向量
        point_to_projection = point - projected_point

        # 计算点积来判断左右
        dot_product = np.dot(point_to_projection, normal_vector)

        # 赋予符号
        signed_distance = min_distance * np.sign(dot_product)

        return signed_distance

    def point_to_line_segment_distance(self, point, line_start, line_end):
        """
        计算点到线段的距离和投影点
        """
        line_vec = line_end - line_start
        point_vec = point - line_start

        line_length = np.linalg.norm(line_vec)
        if line_length == 0:
            return np.linalg.norm(point_vec), line_start

        # 计算投影比例
        t = np.dot(point_vec, line_vec) / (line_length ** 2)

        # 限制投影在线段范围内
        t = max(0, min(1, t))

        # 计算投影点
        projection = line_start + t * line_vec

        # 计算距离
        distance = np.linalg.norm(point - projection)

        return distance, projection

    def check_intersection_with_lines(self, pred_ego_polygon, raw_env):
        """Check intersection with lines.

        Args:
            pred_ego_polygon: n x 4 x 2  自车预测位置polygon
            raw_env
        Returns:
            indecies: 与车道线有交互的自车polygon的索引, list
            attr: 交互的车道线属性, map {indecies: attr}
        """
        indecies = []
        attr = {}
        solid_line_flag_for_toggle = False
        solid_dash_line_flag_for_toggle = False
        dash_solid_line_flag_for_toggle = False
        lane_lines = raw_env["lane_lines"]  # list
        lateral_distances = {}  # 新增：存储横向距离

        for i, polygon in enumerate(pred_ego_polygon):
            filter_lines = []
            ego_center_pt = polygon.mean(axis=0).reshape(1, 2)

            for p, a in lane_lines:
                mask = np.linalg.norm(p[:, :2] - ego_center_pt, axis=1) < 5.0
                filter_pt = p[mask, :]
                filter_lines.append([p[mask, :], a[mask, :]])

            for filter_pt, filter_attr in filter_lines:
                if filter_pt.shape[0] < 2:
                    continue
                polyline = LineString(filter_pt[:, :2])
                ego_polygon = Polygon(polygon)
                if polyline.intersects(ego_polygon):
                    lateral_distance = self.calculate_signed_lateral_distance(
                        polygon, ego_center_pt[0], polyline
                    )
                    indecies.append(i)
                    attr.update({i: filter_attr[filter_attr.shape[0] // 2, :]})  # 取中点
                    for j in range(0, filter_attr.shape[0]):
                        t, c = filter_attr[j, :]
                        if t in [1, 5, 7, 8]:
                            solid_line_flag_for_toggle = True
                            break
                    for j in range(0, filter_attr.shape[0]):
                        t, c = filter_attr[j, :]
                        if t in [3]:
                            solid_dash_line_flag_for_toggle = True
                            break
                    for j in range(0, filter_attr.shape[0]):
                        t, c = filter_attr[j, :]
                        if t in [4]:
                            dash_solid_line_flag_for_toggle = True
                            break

                    lateral_distances.update({i: lateral_distance})
                    break
        return indecies, attr, lateral_distances, solid_line_flag_for_toggle, solid_dash_line_flag_for_toggle, dash_solid_line_flag_for_toggle

    def check_near_solidline(self, pred_ego_polygon, raw_env):
        """Check intersection with lines.

        Args:
            pred_ego_polygon: n x 4 x 2  自车预测位置polygon
            raw_env
        Returns:
            indecies: 与车道线有交互的自车polygon的索引, list
            attr: 交互的车道线属性, map {indecies: attr}
        """
        indecies = []
        attr = {}
        solid_line_flag = False
        solid_dash_line_flag = False
        dash_solid_line_flag = False
        lane_lines = raw_env["lane_lines"]  # list
        lateral_distances = {}  # 新增：存储横向距离

        for i, polygon in enumerate(pred_ego_polygon):
            filter_lines = []
            ego_center_pt = polygon.mean(axis=0).reshape(1, 2)

            # 左右最近line： (距离, 属性)
            left_near  = (np.inf, None)
            right_near = (np.inf, None)

            for p, a in lane_lines:
                mask = np.linalg.norm(p[:, :2] - ego_center_pt, axis=1) < 5.0
                filter_pt = p[mask, :]
                filter_lines.append([p[mask, :], a[mask, :]])

            for filter_pt, filter_attr in filter_lines:
                if filter_pt.shape[0] < 2:
                    continue
                polyline = LineString(filter_pt[:, :2])
                ego_polygon = Polygon(polygon)

                lateral_distance = self.calculate_signed_lateral_distance(
                    polygon, ego_center_pt[0], polyline
                )

                mid_attr = filter_attr[filter_attr.shape[0] // 2, :]

                if lateral_distance < 0:
                    if abs(lateral_distance) < left_near[0]:
                        left_near = (abs(lateral_distance), mid_attr)

                else:
                    if lateral_distance < right_near[0]:
                        right_near = (lateral_distance, mid_attr)

            if left_near[1] is not None or right_near[1] is not None:
                indecies.append(i)
                # 用字典把左右两条都带回去
                attr[i] = {
                    'left':  left_near[1],   # 左边第一条属性
                    'right': right_near[1]   # 右边第一条属性
                }

        return indecies, attr

    def forward(self, pred_polygon, cur_env, pred_ts, turn_signal, last_lateral_distance, np_plus_lcc_status, last_ego_center_pt):
        """计算拨杆Reward."""
        pred_ego_polygon_index, intersection_line_attr, lateral_distances, solid_line_flag_for_toggle, solid_dash_line_flag_for_toggle, dash_solid_line_flag_for_toggle = self.check_intersection_with_lines(
            np.expand_dims(pred_polygon, 0), cur_env
        )

        np_lcc_penalty = 0
        turn_signal_reward = 0

        cur_ego_center_pt = pred_polygon.mean(axis=0).reshape(1, 2)
        delta_y = cur_ego_center_pt[0, 1] - last_ego_center_pt[0, 1]
        last_ego_center_pt = cur_ego_center_pt
        new_intersect_status = False
        if pred_ego_polygon_index:
            new_intersect_status = True

        if pred_ego_polygon_index:
            if np_plus_lcc_status:
                np_lcc_penalty += 4
            t, c = intersection_line_attr[pred_ego_polygon_index[0]]
            turn_direction = 0
            if abs(last_lateral_distance) < 5:
                if lateral_distances[pred_ego_polygon_index[0]] > last_lateral_distance:
                    turn_direction = 1
                elif lateral_distances[pred_ego_polygon_index[0]] < last_lateral_distance:
                    turn_direction = 2
            last_lateral_distance = lateral_distances[pred_ego_polygon_index[0]]

            if turn_signal == turn_direction and ((turn_signal != 0 and not solid_line_flag_for_toggle) or (solid_dash_line_flag_for_toggle and turn_signal == 1) or (dash_solid_line_flag_for_toggle and turn_signal == 2)):
                turn_signal_reward += -2

            if turn_signal == turn_direction and ((turn_signal != 0 and solid_line_flag_for_toggle) or (solid_dash_line_flag_for_toggle and turn_signal == 2) or (dash_solid_line_flag_for_toggle and turn_signal == 1)):
                turn_signal_reward += 1

            # 拨杆变道折返惩罚
            if(turn_signal != 0 and turn_signal != turn_direction):
                turn_signal_reward += 1

        else:
            pred_ego_polygon_index_near, intersection_line_attr_near = self.check_near_solidline(np.expand_dims(pred_polygon, 0), cur_env)
            if pred_ego_polygon_index_near and 0 in intersection_line_attr_near:
                left_attr  = intersection_line_attr_near[0]['left']   # 左边第一条线的属性
                right_attr = intersection_line_attr_near[0]['right']  # 右边第一条线的属性

                if (turn_signal == 1 and delta_y > 0 and left_attr is not None and left_attr[0] not in [1,4,5,7,8]) or (turn_signal == 2 and delta_y < 0 and right_attr is not None and right_attr[0] not in [1,3,5,7,8]):
                    turn_signal_reward += -2
                elif (turn_signal == 1 and delta_y < 0 and left_attr is not None and left_attr[0] not in [1,4,5,7,8]) or (turn_signal == 2 and delta_y > 0 and right_attr is not None and right_attr[0] not in [1,3,5,7,8]):
                    turn_signal_reward += 2
        return -np_lcc_penalty, -turn_signal_reward, last_lateral_distance, last_ego_center_pt, new_intersect_status


class NaviFollowReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Navi follow reward."""
        # ego_motion_preds = model_outputs["planner_pred"]["pos_sampled"]  # [B, 130, 25, 2]
        ego_motion_preds = model_outputs["traj_pred_pos"][..., :25, :]  # [B, 130, 25, 2]
        bs, num_trajs, num_future_step, _ = ego_motion_preds.shape
        ego_pred_points = ego_motion_preds
        # .view(bs, num_trajs, -1, 2)  # [B, 81, 35, 2]
        ego_pred_last_points = ego_pred_points[:, :, -1, :]  # [B, 130, 2]

        navi_path_points = model_inputs["navi_path_points"]  # [B, 1, 30, 2]
        navi_path_types = model_inputs["navi_path_types"]  # [B, 1, 30]
        bs, num_paths, num_points, _ = navi_path_points.shape
        new_navi_path_points = navi_path_points.clone()
        navi_path_valid_mask = (navi_path_types == -1).float()  # [B, 1, 30], 0 means invalid
        ones_column = torch.ones_like(navi_path_valid_mask[..., -1:])  # [B, 1, 1]
        last_valid_idx = torch.cat([navi_path_valid_mask, ones_column], dim=-1).argmax(dim=-1) - 1
        last_valid_point = new_navi_path_points[torch.arange(bs), torch.arange(num_paths), last_valid_idx[:, 0]]
        padding_last_valid_points = last_valid_point.reshape(bs, num_paths, 1, 2).expand(-1, -1, num_points, -1)
        new_navi_path_points[navi_path_types == -1] = padding_last_valid_points[navi_path_types == -1]

        navi_path_last_points = new_navi_path_points[:, :, -1, :]  # [B, 1, 2]
        ego_to_navi_dist = point_to_trajectory_distance(ego_pred_last_points, new_navi_path_points)  # [B, 81, 1]
        navi_to_ego_dist = point_to_trajectory_distance(navi_path_last_points, ego_pred_points)  # [B, 1, 81]
        min_dists = torch.stack([ego_to_navi_dist.squeeze(2), navi_to_ego_dist.squeeze(1)], dim=2)  # [B, 81, 2]
        min_dists = min_dists.min(dim=-1, keepdim=True)[0].repeat(1, 1, num_future_step)  # [B, 81]

        path_semantic = model_inputs["navi_path_lc_attr"][..., 0].squeeze()
        lc_type = model_inputs["navi_path_lc_attr"][..., 2].squeeze()
        min_dists *= ((lc_type != 0) | (path_semantic != 1)).float().view(bs, 1, 1)

        return -min_dists.detach()


class CPFollowReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Contral point follow reward."""
        return


class EfficiencyLoss(BaseRewardLoss):
    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Forward."""
        loss_dict = {}
        for key, reward_func in self.reward_funcs.items():
            loss = reward_func(model_outputs, model_inputs)
            # from IPython import embed; embed()
            loss_dict[key] = loss * self.loss_scales[key]

        return loss_dict


class SmoothReward(BaseLoss):
    def __init__(
        self,
        max_loss=10.0,
        dt=0.2,  # 采样时间间隔（秒）
        # 时间线性衰减参数
        time_decay_start=1.0,  # 时间衰减起始权重
        time_decay_end=0.5,  # 时间衰减结束权重
        # 纵向加速度参数
        long_acc_comfort=(-1.0, 1.0),  # 舒适区范围 (m/s²)
        long_acc_tolerance=(-2.0, 2.0),  # 容忍区范围 (m/s²)
        long_acc_scale_range=(-5.0, 5.0),  # 归一化scale范围 (m/s²)
        long_acc_k1=0.5,  # 容忍区损失系数
        long_acc_k2=2.0,  # 强惩罚区损失系数
        # 纵向加加速度参数
        long_jerk_comfort=(-1.0, 1.0),  # 舒适区范围 (m/s³)
        long_jerk_tolerance=(-2.0, 2.0),  # 容忍区范围 (m/s³)
        long_jerk_scale_range=(-5.0, 5.0),  # 归一化scale范围 (m/s³)
        long_jerk_k1=0.8,  # 容忍区损失系数
        long_jerk_k2=3.2,  # 强惩罚区损失系数
        # 横向加速度参数
        lat_acc_comfort=(-0.5, 0.5),  # 舒适区范围 (m/s²)
        lat_acc_tolerance=(-1.0, 1.0),  # 容忍区范围 (m/s²)
        lat_acc_scale_range=(-3.0, 3.0),  # 归一化scale范围 (m/s²)
        lat_acc_k1=0.6,  # 容忍区损失系数
        lat_acc_k2=2.4,  # 强惩罚区损失系数
        # 横向加加速度参数
        lat_jerk_comfort=(-0.6, 0.6),  # 舒适区范围 (m/s³)
        lat_jerk_tolerance=(-1.2, 1.2),  # 容忍区范围 (m/s³)
        lat_jerk_scale_range=(-3.0, 3.0),  # 归一化scale范围 (m/s³)
        lat_jerk_k1=0.7,  # 容忍区损失系数
        lat_jerk_k2=2.8,  # 强惩罚区损失系数
        # 全局权重
        loss_weights=(1.0, 1.0, 0.8, 0.6),  # (acc, jerk, lat_acc, lat_jerk)
    ) -> None:
        """Init."""
        super().__init__()
        self.max_loss = max_loss
        self.dt = dt
        # 时间衰减参数（新增）
        self.time_decay_start = time_decay_start
        self.time_decay_end = time_decay_end

        # 纵向参数
        self.long_acc_comfort = long_acc_comfort
        self.long_acc_tolerance = long_acc_tolerance
        self.long_acc_scale_range = long_acc_scale_range
        self.long_acc_k1 = long_acc_k1
        self.long_acc_k2 = long_acc_k2

        # 纵向加加速度
        self.long_jerk_comfort = long_jerk_comfort
        self.long_jerk_tolerance = long_jerk_tolerance
        self.long_jerk_scale_range = long_jerk_scale_range
        self.long_jerk_k1 = long_jerk_k1
        self.long_jerk_k2 = long_jerk_k2

        # 横向参数
        self.lat_acc_comfort = lat_acc_comfort
        self.lat_acc_tolerance = lat_acc_tolerance
        self.lat_acc_scale_range = lat_acc_scale_range
        self.lat_acc_k1 = lat_acc_k1
        self.lat_acc_k2 = lat_acc_k2

        # 横向加加速度
        self.lat_jerk_comfort = lat_jerk_comfort
        self.lat_jerk_tolerance = lat_jerk_tolerance
        self.lat_jerk_scale_range = lat_jerk_scale_range
        self.lat_jerk_k1 = lat_jerk_k1
        self.lat_jerk_k2 = lat_jerk_k2

        # 损失权重
        self.acc_weight, self.jerk_weight, self.lat_acc_weight, self.lat_jerk_weight = loss_weights

    def _generate_linear_weights(self, time, device):
        # 生成线性衰减时间权重(从time_decay_start到time_decay_end)
        # 创建线性衰减序列
        weights = torch.linspace(
            self.time_decay_start,
            self.time_decay_end,
            time,
            device=device,
            dtype=torch.float32,
        )
        # 归一化处理（保持权重总和为1）
        return weights / weights.sum()

    def _three_zone_loss(self, value, comfort_range, tolerance_range, scale_range, k1, k2, global_weight, time_weights):
        """
        集成权重计算的三区段损失函数.

        :param value: 输入值张量
        :param comfort_range: 舒适区范围
        :param tolerance_range: 容忍区范围
        :param k1: 容忍区损失系数
        :param k2: 强惩罚区损失系数
        :param global_weight: 全局损失权重
        :param time_weights: 时间衰减权重
        :return: 标量损失值
        """
        comfort_min, comfort_max = comfort_range
        tolerance_min, tolerance_max = tolerance_range
        scale_range_min, scale_range_max = scale_range
        assert comfort_min <= comfort_max
        assert tolerance_min <= tolerance_max
        assert comfort_min >= tolerance_min
        assert comfort_max <= tolerance_max
        assert scale_range_min <= tolerance_min
        assert scale_range_max >= tolerance_max

        # 转换为张量
        comfort_min_t = torch.tensor(comfort_min, dtype=value.dtype, device=value.device)
        comfort_max_t = torch.tensor(comfort_max, dtype=value.dtype, device=value.device)
        tolerance_min_t = torch.tensor(tolerance_min, dtype=value.dtype, device=value.device)
        tolerance_max_t = torch.tensor(tolerance_max, dtype=value.dtype, device=value.device)

        # 初始化损失张量
        loss_tensor = torch.zeros_like(value)

        # 1. 低于舒适区的情况
        under_comfort = value < comfort_min_t

        # a) 容忍区（舒适区下方）
        under_tolerance = under_comfort & (value >= tolerance_min_t)
        diff_under = comfort_min_t - value
        loss_tensor[under_tolerance] = k1 * torch.pow(diff_under[under_tolerance], 2)

        # b) 强惩罚区（容忍区下方）
        under_penalty = value < tolerance_min_t
        penalty_value_gap_for_min = (k2 - k1) * (tolerance_min - comfort_min) ** 2
        loss_tensor[under_penalty] = k2 * torch.pow(diff_under[under_penalty], 2) - penalty_value_gap_for_min

        # 2. 高于舒适区的情况
        over_comfort = value > comfort_max_t

        # a) 容忍区（舒适区上方）
        over_tolerance = over_comfort & (value <= tolerance_max_t)
        diff_over = value - comfort_max_t
        loss_tensor[over_tolerance] = k1 * torch.pow(diff_over[over_tolerance], 2)

        # b) 强惩罚区（容忍区上方）
        over_penalty = value > tolerance_max_t
        penalty_value_gap_for_max = (k2 - k1) * (tolerance_max - comfort_max) ** 2
        loss_tensor[over_penalty] = k2 * torch.pow(diff_over[over_penalty], 2) - penalty_value_gap_for_max

        # 设定loss scale范围并scale loss
        loss_scale_range_for_min = k2 * (scale_range_min - comfort_min) ** 2 - penalty_value_gap_for_min
        loss_scale_range_for_max = k2 * (scale_range_max - comfort_max) ** 2 - penalty_value_gap_for_max
        loss_scale_range_max = max(loss_scale_range_for_min, loss_scale_range_for_max)
        loss_tensor = torch.clamp(loss_tensor / loss_scale_range_max, min=0.0, max=1.0)

        # 应用时间权重并计算最终损失
        weighted_loss = loss_tensor * time_weights
        summed_loss = weighted_loss.sum(dim=2) * self.max_loss
        final_loss = global_weight * summed_loss

        # from IPython import embed; embed()
        return final_loss.view(final_loss.shape[0], -1)

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Trajectory smooth reward."""
        planner_pred = model_outputs["planner_pred"]
        ego_long_acc = planner_pred["long_acc_sampled"]
        long_vel = planner_pred["long_vel_sampled"]  # 纵向速度
        yaw_rate = planner_pred["yaw_rate_sampled"]  # 横摆角速度

        bs, num_trajs, self.time_num, _ = ego_long_acc.shape
        device = ego_long_acc.device

        # ===== 1. 生成线性衰减权重 =====
        time_weights = self._generate_linear_weights(self.time_num, device).view(1, 1, self.time_num, 1)
        jerk_weights = time_weights[:, :, :-1, :]  # jerk权重少一个时间步

        # ===== 2. 计算横纵向动力学量 =====
        self.lat_acc = long_vel * yaw_rate  # 横向加速度 = 纵向速度 × 横摆角速度
        self.lat_jerk = torch.diff(self.lat_acc, dim=2) / self.dt  # 横向加加速度 = 横向加速度的差分
        self.long_jerk = torch.diff(ego_long_acc, dim=2) / self.dt  # 纵向加加速度

        # ===== 3. 纵向加速度损失 =====
        self.acc_loss = self._three_zone_loss(
            ego_long_acc,
            self.long_acc_comfort,
            self.long_acc_tolerance,
            self.long_acc_scale_range,
            self.long_acc_k1,
            self.long_acc_k2,
            self.acc_weight,
            time_weights,
        )

        # ===== 4. 纵向加加速度损失 =====
        self.jerk_loss = self._three_zone_loss(
            self.long_jerk,
            self.long_jerk_comfort,
            self.long_jerk_tolerance,
            self.long_jerk_scale_range,
            self.long_jerk_k1,
            self.long_jerk_k2,
            self.jerk_weight,
            jerk_weights,
        )

        # ===== 5. 横向加速度损失 =====
        self.lat_acc_loss = self._three_zone_loss(
            self.lat_acc,
            self.lat_acc_comfort,
            self.lat_acc_tolerance,
            self.lat_acc_scale_range,
            self.lat_acc_k1,
            self.lat_acc_k2,
            self.lat_acc_weight,
            time_weights,
        )

        # ===== 6. 横向加加速度损失 =====
        self.lat_jerk_loss = self._three_zone_loss(
            self.lat_jerk,
            self.lat_jerk_comfort,
            self.lat_jerk_tolerance,
            self.lat_jerk_scale_range,
            self.lat_jerk_k1,
            self.lat_jerk_k2,
            self.lat_jerk_weight,
            jerk_weights,
        )

        # reward值应该为loss的负值
        final_reward = (
            -1.0
            / (self.acc_weight + self.jerk_weight + self.lat_acc_weight + self.lat_jerk_weight)
            * (self.acc_loss + self.jerk_loss + self.lat_acc_loss + self.lat_jerk_loss)
        )

        # for i in range(final_reward.shape[0]):
        #     for j in range(final_reward.shape[1]):
        #         motion_reward = final_reward[i][j]
        #         if motion_reward < -3:
        #             from IPython import embed; embed()

        return final_reward


class ConsistancyReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """History trajectory consistancy reward."""
        return


class ComfortLoss(BaseRewardLoss):
    def get_loss(self):
        """Get loss."""
        return

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Forward."""
        loss_dict = {}
        for key, reward_func in self.reward_funcs.items():
            loss = -reward_func(model_outputs, model_inputs)
            loss_dict[key] = loss * self.loss_scales[key]

        return loss_dict


class LaneViolationReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Lane Line violation reward."""
        return


class CentralizationReward(BaseLoss):
    """
    计算自车在车道中的居中奖励，基于与最近车道线的距离和方向对齐度。
    """

    # 定义常量，提高可维护性
    LANE_LINE_SEARCH_RADIUS = 10.0      # 车道线搜索半径
    ROAD_EDGE_SEARCH_RADIUS = 8.0      # 路沿搜索半径
    MAX_DISTANCE_THRESHOLD = 4.0
    MIN_LANE_WIDTH = 0.8
    MAX_LANE_WIDTH = 5.5

    def _find_nearest_boundaries(self, ego_point, lane_lines, road_edge):
        """
        寻找左右两侧最近的车道边界
        优先级：车道线 > 路沿

        Returns:
            tuple: (left_width, right_width)，如果没找到返回 inf
        """
        is_park_odd = False
        left_width = float('inf')
        right_width = float('inf')
        ego_pos = np.array([ego_point.x, ego_point.y])

        # 1. 遍历车道线一次，同时找左右（必须遍历所有，找最近）
        for points, _ in lane_lines:
            if len(points) < 2:
                continue

            # 快速过滤：检查是否有任何点在搜索半径内
            distances_to_ego = np.linalg.norm(points[:, :2] - ego_pos, axis=1)
            if np.min(distances_to_ego) > self.LANE_LINE_SEARCH_RADIUS:
                continue

            signed_dist = calculate_signed_lateral_distance(ego_point, points[:, :2])
            if signed_dist == float('inf'):
                continue

            if abs(signed_dist) < self.MAX_DISTANCE_THRESHOLD:
                if signed_dist < 0:  # 左侧
                    left_width = min(left_width, abs(signed_dist))
                elif signed_dist > 0:  # 右侧
                    right_width = min(right_width, abs(signed_dist))

        # 2. 按需遍历路沿（必须遍历所有，找最近）
        need_left = left_width > self.MAX_DISTANCE_THRESHOLD
        need_right = right_width > self.MAX_DISTANCE_THRESHOLD

        if need_left or need_right and is_park_odd:
            for edge_points in road_edge:
                if len(edge_points) < 2:
                    continue

                # 快速过滤：检查是否有任何点在搜索半径内
                distances_to_ego = np.linalg.norm(edge_points[:, :2] - ego_pos, axis=1)
                if np.min(distances_to_ego) > self.ROAD_EDGE_SEARCH_RADIUS:
                    continue

                signed_dist = calculate_signed_lateral_distance(ego_point, edge_points[:, :2])
                if signed_dist == float('inf'):
                    continue

                # 只在缺失侧更新，但必须遍历所有找最小值
                if abs(signed_dist) < self.MAX_DISTANCE_THRESHOLD:
                    if signed_dist < 0 and need_left:  # 左侧缺失
                        left_width = min(left_width, abs(signed_dist))
                    elif signed_dist > 0 and need_right:  # 右侧缺失
                        right_width = min(right_width, abs(signed_dist))

        return left_width, right_width

    def forward(self, pred_ego_polygon, raw_env, centralization_reward_weight):
        """
        计算居中奖励。

        Args:
            pred_ego_polygon (np.ndarray): 自车多边形顶点，形状为 (n, 2) 的二维数组，表示n个顶点的x、y坐标。
            raw_env (dict): 环境数据，需包含键 'lane_lines' 和 'road_edge'。
            centralization_reward_weight (float): 居中奖励的基础权重。
        Returns:
            double:
                - centralization_reward (float): 计算出的居中奖励值。
        """
        centralization_reward = 0.0
        # 1. 输入检查
        if pred_ego_polygon is None or len(pred_ego_polygon) < 4:
            return centralization_reward
        if not isinstance(raw_env, dict) or "lane_lines" not in raw_env or "road_edge" not in raw_env:
            return centralization_reward
        lane_lines = raw_env["lane_lines"]
        road_edge = raw_env["road_edge"]
        if len(lane_lines) == 0 and len(road_edge) == 0:
            return centralization_reward

        ego_center = np.mean(pred_ego_polygon, axis=0)
        ego_point = Point(ego_center[0], ego_center[1])

        # 寻找最近的中心线，判断不应该奖励居中的类型
        centerlines = raw_env.get("centerlines", [])
        centerline_dist = float('inf')
        centerline_type = 999
        found_valid_centerline = False
        # 不应该奖励居中的车道类型
        no_centralization_lanes = {
            3,   # BICYCLE_LANE (非机动车道)
            6,   # LANE_SIDEWALK (人行道)
            19,  # EMERGENCY_LANE (应急车道)
            21,  # PARNKING_LANE (停车道)
            26,  # NON_DRIVING_ZONE (非行驶区)
            27,  # BORROWING_ZONE (借道区)
            32,  # REVERSE_LANE (逆向车道)
            33,  # TURNING_WAITING_AREA (待转区)
        }

        # 寻找最近中心线及类型
        for centerline in centerlines:
            if not centerline or len(centerline) < 2:
                continue
            centerline_points = np.asarray(centerline[0][:, :2])
            if len(centerline_points) < 2:
                continue
            signed_dist = calculate_signed_lateral_distance(ego_point, centerline_points)
            if abs(signed_dist) < centerline_dist:
                centerline_dist = abs(signed_dist)
                centerline_type = centerline[1] if len(centerline) > 1 else 999
                found_valid_centerline = True

        # 逻辑优化：明确处理各种情况
        if found_valid_centerline and centerline_dist < 3.0:
            # 距离中心线很近，需要检查类型
            if centerline_type in no_centralization_lanes:
                # 这些车道不应该奖励居中
                return centralization_reward

        # 寻找两侧最近车道线/路沿
        left_width, right_width = self._find_nearest_boundaries(ego_point, lane_lines, road_edge)

        #  距离条件检查
        if left_width > self.MAX_DISTANCE_THRESHOLD or right_width > self.MAX_DISTANCE_THRESHOLD:
            return centralization_reward
        lane_width = left_width + right_width
        if lane_width < self.MIN_LANE_WIDTH or lane_width > self.MAX_LANE_WIDTH:
            return centralization_reward
        dist_diff = abs(left_width - right_width)

        # 计算最终居中奖励
        reward_ratio = (lane_width - dist_diff) / lane_width
        if reward_ratio >= 0.9:
            # 高区：线性映射到 [0.9^4, 1.2]，强调完美居中
            t = (reward_ratio - 0.9) / 0.1
            reward = (0.9 ** 4) + t * (1.2 - (0.9 ** 4))  # 约 0.656 + t * 0.544
        else:
            # 低区：4次方，保持低区梯度
            reward = reward_ratio ** 4
        centralization_reward = reward * centralization_reward_weight

        return centralization_reward


class RegulationLoss(BaseRewardLoss):
    def get_loss(self):
        """Get loss."""
        return

    def forward(self, model_outputs: Dict[str, Any], model_inputs: Dict[str, Any]):
        """Forward."""
        loss_dict = {}
        for key, reward_func in self.reward_funcs.items():
            loss = self.get_loss()
            loss_dict[key] = loss * self.loss_scales[key]

        return loss_dict


class VelocityReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, speed, speed_limit, reward_max=50.0):
        """Forward."""
        # 计算速度提高的reward
        velocity_reward = np.clip(speed / 10.0, -reward_max, reward_max)
        speed_limit_penalty = max(speed - speed_limit, 0.0)
        return (velocity_reward, speed_limit_penalty)

class ProgressReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def spd_func(self, x, max_reward=0.2, left_peak=0.9, right_peak=1.0, complete_zero=1.1):
        """Calc spd reward value."""
        if x <= left_peak:
            res = np.exp(-0.5 * ((x - left_peak) / 0.3) ** 2)
        elif x <= right_peak:
            return max_reward
        elif x > complete_zero:
            res = 0
        else:
            h = 1.15 - right_peak
            a = 2 / h**3
            b = -3 / h**2
            dx = x - right_peak
            res = a * dx**3 + b * dx**2 + 1
        return res * max_reward

    def forward(self, speed, speed_limit, reward_max=0.2):
        """Forward."""
        # 计算速度映射reward
        if speed <= 0:
            return 0.01
        elif 0 < speed < 1:
            return 0
        elif 1 <= speed:
            spd = speed / (speed_limit + 1e-5)
            reward = self.spd_func(spd, reward_max)
            return reward


class SpeedLimitReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def forward(self, speed, speed_limit, fine_tune=False):
        """Forward."""
        overspeed_flg = 0
        penalty = 0
        if speed_limit > 0:
            if speed > speed_limit:
                overspeed_flg = 1
                if fine_tune:
                    # 40kph以下不限制，以上按10% 20%两档
                    if speed / max(speed_limit, 40.0) > 1.2:
                        penalty = 4.0
                    elif speed / max(speed_limit, 40.0) > 1.1:
                        penalty = 2.0
                    else:
                        penalty = 0.1
                else:
                    penalty = 0.1

        return overspeed_flg, penalty

class SlowFollowReward(BaseLoss):
    def __init__(self,
                 linear_penalty_cfg:dict=dict(
                    vru_max_penalty=2.0,
                    vru_min_penalty=0.0,
                    vru_penalty_interval=1,
                    veh_max_penalty=4.0,
                    veh_min_penalty=0.0,
                    veh_penalty_interval=1
                 ),
                 min_active_speed_mps:float=1.0,
                 max_penalty:float=4,
                 ratio_thres:float=0.9,
                 mode:str='linear', # ['linear', 'speed_diff']
                 ) -> None:
        """Init."""
        super().__init__()
        self.max_penalty=max_penalty
        self.ratio_thres=ratio_thres
        self.min_active_speed_mps=min_active_speed_mps

        self.linear_penalty_cfg= linear_penalty_cfg

        assert mode in ['linear', 'speed_diff'], f"Unsupported mode {mode}!"
        self.mode=mode

    def forward(
        self,
        speeds_kph: np.ndarray,
        speed_limits_kph: np.ndarray,
        scenario_id:ScenarioEnum,
        reward_results:Dict,
        is_checker_follow_slow:bool,
        is_human_follow_slow:bool,
        follow_slow_segs:List[SlowFollowSegment],
        safe_drive_mask:np.array,
        raw_env_list:List,
    )->np.ndarray:
        if is_checker_follow_slow:
            assert follow_slow_segs is not None and len(follow_slow_segs) > 0, "detect slow follow scene, but no segments?"

            return self.compute_with_follow_slow_segments( speeds_kph, speed_limits_kph, follow_slow_segs, scenario_id, reward_results, raw_env_list, safe_drive_mask)
        elif is_human_follow_slow:
            return self.compute_by_balance_rewards( speeds_kph, speed_limits_kph, reward_results, scenario_id)

        return np.zeros_like(speed_limits_kph)

    def compute_with_follow_slow_segments(
        self,
        speeds_kph: np.ndarray,
        speed_limits_kph: np.ndarray,
        follow_slow_segs:List[SlowFollowSegment],
        scenario_id:ScenarioEnum,
        reward_results:Dict,
        raw_env_list:List,
        safe_drive_mask: np.array,
    )->np.ndarray:
        if ScenarioEnum.kFollowSlow!= scenario_id:
            return np.zeros_like(speed_limits_kph)

        if 'linear'==self.mode:
            penalty = self._compute_linear_penalty(speeds_kph, speed_limits_kph, follow_slow_segs, reward_results, raw_env_list, safe_drive_mask)
        else:
            assert False, f"Unsupported mode {self.mode}!"
        return penalty

    def compute_by_balance_rewards(
        self,
        speeds_kph: np.ndarray,
        speed_limits_kph: np.ndarray,
        reward_results,
        scenario_id:ScenarioEnum,
    )->np.ndarray:
        if ScenarioEnum.kFollowSlowByHuman!= scenario_id:
            return np.zeros_like(speeds_kph)
        ratio = speeds_kph / (speed_limits_kph + 1e-3)
        ratio = np.clip((self.ratio_thres - ratio) / (self.ratio_thres), a_min=0, a_max=1.0)

        vru_min_penalty = self.linear_penalty_cfg['vru_min_penalty']
        vru_max_penalty = self.linear_penalty_cfg['vru_max_penalty']
        vru_penalty = vru_min_penalty + (ratio * (vru_max_penalty - vru_min_penalty) if vru_max_penalty > vru_min_penalty else 0)

        #safe: ttc_reward,traffic_light_reward,min_distance_reward,speed_limit_reward
        is_safe = ((reward_results["ttc_reward"] > -0.1)
                   & (reward_results["traffic_light_reward"] > -0.1)
                   & (reward_results["min_distance_reward"] > -0.1)
                   & (reward_results["speed_limit_reward"] >-0.1))
        #logic cross_solid_line_reward dangerous_lc_penalty  navi_lane_reward  navi_reward wrong_way_reward
        is_logic = ((reward_results["danger_lc_reward"] > -0.1)
                    & (reward_results["navi_reward"] >-0.1)
                    & (reward_results["wrong_way_reward"] >-0.1))
                   #(reward_results["cross_solid_line_reward"][t] > -0.1
        #progress_reward
        is_no_progress_reward = reward_results["progress_reward"] <2.0

        mask = (~is_safe) | (~is_logic) | (~is_no_progress_reward)
        vru_penalty[mask[1:]]=0

        # no collision
        is_no_collision = ~np.any(reward_results["collision_reward"] <= -10)
        vru_penalty *= is_no_collision

        return vru_penalty

    def _compute_linear_penalty(
        self,
        speeds_kph: np.ndarray,
        speed_limits_kph: np.ndarray,
        follow_slow_segs:List[SlowFollowSegment],
        reward_results:Dict,
        raw_env_list:List,
        safe_drive_mask: np.array,
    ):

        penalty = np.zeros_like(speeds_kph)

        ratio = speeds_kph / (speed_limits_kph + 1e-3)
        ratio = np.clip((self.ratio_thres - ratio) / (self.ratio_thres), a_min=0, a_max=1.0)

        vru_min_penalty = self.linear_penalty_cfg['vru_min_penalty']
        vru_max_penalty = self.linear_penalty_cfg['vru_max_penalty']
        vru_penalty = vru_min_penalty + (ratio * (vru_max_penalty - vru_min_penalty) if vru_max_penalty > vru_min_penalty else 0)

        veh_min_penalty = self.linear_penalty_cfg['veh_min_penalty']
        veh_max_penalty = self.linear_penalty_cfg['veh_max_penalty']
        veh_penalty = veh_min_penalty + (ratio * (veh_max_penalty - veh_min_penalty) if veh_max_penalty > veh_min_penalty else 0)

        vru_penalty_interval = self.linear_penalty_cfg['vru_penalty_interval']
        veh_penalty_interval = self.linear_penalty_cfg['veh_penalty_interval']
        for seg in follow_slow_segs:
            if seg.is_vru_block:
                penalty[seg.start_idx:seg.end_idx+1:vru_penalty_interval] = vru_penalty[seg.start_idx:seg.end_idx+1:vru_penalty_interval]
            elif seg.is_vehicle_block:
                penalty[seg.start_idx:seg.end_idx+1:veh_penalty_interval] = veh_penalty[seg.start_idx:seg.end_idx+1:veh_penalty_interval]

        penalty[(np.abs(speed_limits_kph) < 1e-3) | (np.abs(speeds_kph) < self.min_active_speed_mps*3.6) | (np.abs(speeds_kph)>=self.ratio_thres*speed_limits_kph)] = 0
        for ts in range(safe_drive_mask.shape[0]):
            if safe_drive_mask[ts] is False:
                penalty[ts] = penalty[ts] * 2
        self._clear_penalty_by_if_need(penalty, raw_env_list, reward_results)

        return penalty

    def _calc_near_junc_mask(self,raw_env_list):
        mask = np.zeros(len(raw_env_list), dtype=bool)
        for i, raw_env in enumerate(raw_env_list):
            manner_scene_list = raw_env.get("manner_scene_info", None)
            link_info = raw_env.get("link_info", None)
            is_narrow_road_scenario = False
            if link_info is not None and len(link_info)>0 and link_info[0] is not None:
                is_narrow_road_scenario = link_info[0].get("road_class", 0) >= 7

            is_junction_scenario = False
            is_main_to_aux_scenario = False
            if not is_narrow_road_scenario and manner_scene_list is not None and len(manner_scene_list)>0:
                manner_scene_list.sort(key=lambda x: x["distance_to_entry"])
                scene = manner_scene_list[0]
                for scene in manner_scene_list:
                    distance_to_entry = scene["distance_to_entry"]
                    distance_to_exit = scene["distance_to_exit"]
                    scene_type = scene["scene_type"]
                    # 2: junction, 5: main_to_aux
                    if 2==scene_type and distance_to_entry<=0 and distance_to_exit >= 0:
                        is_junction_scenario = True
                        break
                    if 5==scene_type and distance_to_entry<=0 and distance_to_exit >= 0:
                        is_main_to_aux_scenario = True
                        break
            mask[i] = is_junction_scenario or is_main_to_aux_scenario
        return mask

    def _clear_penalty_by_if_need(self, penalty:np.ndarray, raw_env_list:List, reward_results:Dict):
        has_wrong_way = np.any(np.abs(reward_results["wrong_way_reward"] >-0.1))
        has_collision = np.any(reward_results["collision_reward"] <= -10)
        pass_junc_mask = np.any(self._calc_near_junc_mask(raw_env_list))
        if has_collision or (has_wrong_way and pass_junc_mask):
            penalty[:] = 0.0

class AccelerationReward(BaseLoss):
    def __init__(self, fps) -> None:
        """Init."""
        super().__init__()

        self.fps = fps

    def _get_ax_from_polygon(self, polygons):
        """Calc ax."""
        obj0, obj1, obj2 = get_xyyaw_from_polygon(polygons)
        dx0 = get_relative_pose_from_obj(obj0, obj1)[0, 3]
        dx1 = get_relative_pose_from_obj(obj1, obj2)[0, 3]
        vx0 = dx0 * self.fps
        vx1 = dx1 * self.fps
        ax = (vx1 - vx0) * self.fps
        return ax

    def get_ax_penalty(self, ax):
        """Calc ax panalty."""
        comfort_acc_upbound = 1.6
        emergency_acc_limit = (-6, 6)
        # No penalty within comfort bound
        if abs(ax) <= comfort_acc_upbound:
            return 0

        # Linear penalty between comfort and emergency bounds
        if emergency_acc_limit[0] <= ax <= emergency_acc_limit[1]:
            if ax > 0:
                # For positive acceleration
                return (((ax - comfort_acc_upbound) / (emergency_acc_limit[1] - comfort_acc_upbound)) ** 2) * 5
            else:
                # For negative acceleration (deceleration)
                return (
                    ((abs(ax) - comfort_acc_upbound) / (abs(emergency_acc_limit[0]) - comfort_acc_upbound)) ** 2
                ) * 5

        return 5

    def get_ay_penalty(self, ay, speed):
        """Calc ay panalty."""
        comfort_acc_upbound = 2.5

        # define different acc bound based on speed
        if speed < 40:
            comfort_acc_upbound = 2.0
        elif speed < 60:
            comfort_acc_upbound = 1.7
        elif speed < 90:
            comfort_acc_upbound = 1.2
        else:
            comfort_acc_upbound = 0.8

        # emergency_acc_limit = (-10, 10)
        # No penalty within comfort bound
        if abs(ay) <= comfort_acc_upbound:
            return 0

        return min(5.0, abs(ay) - comfort_acc_upbound)


    def forward(self, dx_list, dy_list, dyaw_list, pred_ts, last_ax, ego_polygon, ts):
        # ax penalty
        ax = self._get_ax_from_polygon(ego_polygon[ts - 2 : ts + 1])
        ax_penalty = self.get_ax_penalty(ax)

        # ay penalty
        ay = 0.0
        ay_penalty = 0.0
        jx = 0
        jy = 0

        comfort_dec_upbound = 0

        if pred_ts >= 1 and pred_ts < len(dx_list) - 1:
            dx = dx_list[pred_ts + 1]
            dy = dy_list[pred_ts + 1]
            dyaw = dyaw_list[pred_ts + 1]
            dx_prev = dx_list[pred_ts]
            dy_prev = dy_list[pred_ts]
            dy = dy_list[pred_ts + 1]
            vy = dy * self.fps
            vy_prev = dy_prev * self.fps
            ddy = (vy - vy_prev) * self.fps  # dot dot y
            ay = ddy + dx * self.fps * dyaw * self.fps
            ay_penalty = self.get_ay_penalty(ay, dx * self.fps * 3.6)

            # ---- 当前帧加速度 ----
            dyaw_prev = dyaw_list[pred_ts]
            dx_prev_prev = dx_list[pred_ts - 1]
            dy_prev_prev = dy_list[pred_ts - 1]
            dx_prev = dx_list[pred_ts]
            dy_prev = dy_list[pred_ts]
            ax_prev = (dx_prev - dx_prev_prev) * self.fps * self.fps
            ay_prev = ((dy_prev - dy_prev_prev) * self.fps * self.fps) + dx_prev * self.fps * dyaw_prev * self.fps

            # ---- jerk ----
            jx = (ax - ax_prev)
            jy = (ay - ay_prev)


        return (ax, ax_penalty, ay, ay_penalty, comfort_dec_upbound, jx, jy)

class JerkReward(BaseLoss):
    def __init__(self, fps) -> None:
        """Init."""
        super().__init__()

        self.fps = fps

    def _get_ax_from_polygon(self, polygons):
        """Calc ax."""
        obj0, obj1, obj2 = get_xyyaw_from_polygon(polygons)
        dx0 = get_relative_pose_from_obj(obj0, obj1)[0, 3]
        dx1 = get_relative_pose_from_obj(obj1, obj2)[0, 3]
        vx0 = dx0 * self.fps
        vx1 = dx1 * self.fps
        ax = (vx1 - vx0) * self.fps
        return ax

    def get_jx_penalty(self, ax, jx):
        """Calc jx panalty."""
        comfort_jerk_limit = 1
        emergency_jerk_limit = 3.0
        jx_pe = 0
        jerk_weight = 1.0

        if abs(jx) < comfort_jerk_limit:
            jx_pe = 0
        elif comfort_jerk_limit < abs(jx) < emergency_jerk_limit:
            jx_pe = ((abs(jx) - comfort_jerk_limit) / (emergency_jerk_limit - comfort_jerk_limit)) ** 2
        else:
            jx_pe = 1.0

        if ax < 0 and jx < 0:
            jerk_weight = 1.5
        final_jx = min(jerk_weight * jx_pe * 2.0 , 5)

        return final_jx

    def get_jy_penalty(self, jy):
        """Calc jy panalty."""

        comfort_jerk_limit = 0.3
        emergency_jerk_limit = 1.0
        jy_pe = 0
        jerk_weight = 1.0

        if abs(jy) < comfort_jerk_limit:
            jy_pe = 0
        elif comfort_jerk_limit < abs(jy) < emergency_jerk_limit:
            jy_pe = ((abs(jy) - comfort_jerk_limit) / (emergency_jerk_limit - comfort_jerk_limit)) ** 2
        else:
            jy_pe = 1.0

        final_jy = min(jerk_weight * jy_pe * 2.0 , 5)

        return final_jy


    def forward(self, dx_list, dy_list, dyaw_list, pred_ts, last_ax, ego_polygon, ts):

        ax = self._get_ax_from_polygon(ego_polygon[ts - 2 : ts + 1])

        ay = 0.0
        jx_penalty = 0.0
        jy_penalty = 0.0
        jx = 0
        jy = 0

        if pred_ts >= 1 and pred_ts < len(dx_list) - 1:
            dx = dx_list[pred_ts + 1]
            dy = dy_list[pred_ts + 1]
            dyaw = dyaw_list[pred_ts + 1]
            dx_prev = dx_list[pred_ts]
            dy_prev = dy_list[pred_ts]
            dy = dy_list[pred_ts + 1]
            vy = dy * self.fps
            vy_prev = dy_prev * self.fps
            ddy = (vy - vy_prev) * self.fps  # dot dot y
            ay = ddy + dx * self.fps * dyaw * self.fps

            # ---- 当前帧加速度 ----
            dyaw_prev = dyaw_list[pred_ts]
            dx_prev_prev = dx_list[pred_ts - 1]
            dy_prev_prev = dy_list[pred_ts - 1]
            dx_prev = dx_list[pred_ts]
            dy_prev = dy_list[pred_ts]
            ax_prev = (dx_prev - dx_prev_prev) * self.fps * self.fps
            ay_prev = ((dy_prev - dy_prev_prev) * self.fps * self.fps) + dx_prev * self.fps * dyaw_prev * self.fps

            # ---- jerk ----
            jx = (ax - ax_prev)
            jy = (ay - ay_prev)
            jx_penalty = self.get_jx_penalty(ax, jx)
            jy_penalty = self.get_jy_penalty(jy)

        return (jx_penalty, jy_penalty, jx, jy)

class EtcSpeedReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps

    def forward(self, etc_dis, speed, front_center, sod_pose, sod_status):
        # 1. calc speed reward
        max_speed_20 = 35.0
        max_speed_10 = 20.0
        max_gate_speed = 20.0
        max_sod_speed_front = 14.0
        max_sod_speed = 10.0
        front_etc_begin_dis = 20.0
        front_etc_end_dis = 10.0
        in_etc_begin_dis = 10.0
        in_etc_end_dis = -5.0
        front_sod_begin_dis = 14.0
        front_sod_end_dis = 10.0
        speed_tolerance = 5.0
        sod_speed_tolerance = 2.0

        reward = 0.0
        if etc_dis > 20.0:
            return reward

        # a. in front etc speed is 30
        if front_etc_begin_dis > etc_dis > front_etc_end_dis:
            distance_ratio = (etc_dis - front_etc_end_dis) / (front_etc_begin_dis - front_etc_end_dis)
            max_allow_speed = max_speed_10 + distance_ratio * (max_speed_20 - max_speed_10)
            speed_excess = speed - max_allow_speed
            if speed_excess > speed_tolerance:
                reward = (speed_excess) * 0.1

        # b. in gate etc speed is 20
        sod_dis = 1000.0
        if in_etc_begin_dis >= etc_dis > in_etc_end_dis:
            speed_excess = speed - max_gate_speed
            if speed_excess > 0:
                reward = (speed_excess) * 0.2

            #c. in front sod, speed is 10
            if sod_pose is not None and len(sod_pose)>=2 and sod_status is not None:
                sod_dis = np.linalg.norm(np.array(sod_pose) - np.array(front_center)) * np.sign(sod_pose[0] - front_center[0])

                # 1. 10-0米 超速10惩罚
                if sod_dis < front_sod_end_dis and sod_dis > 0.0 :
                    speed_excess = speed - max_sod_speed
                    if speed_excess > 0:
                        reward = (speed_excess) * 0.3
                # 2. 14-10米 平滑过度 14-10km/h
                if sod_dis > front_sod_end_dis and sod_dis < front_sod_begin_dis :
                    dis_ratio = (sod_dis - front_sod_end_dis) / (front_sod_begin_dis - front_sod_end_dis)
                    max_allow_sod_speed = max_sod_speed + dis_ratio * (max_sod_speed_front - max_sod_speed)
                    sod_speed_excess = speed - max_allow_sod_speed
                    if sod_speed_excess > sod_speed_tolerance:
                        reward = (sod_speed_excess) * 1.0

        # print(f"ETC dis: {etc_dis:.3}, sod_dis: {sod_dis:.3}, speed: {speed:.3}, s_reward: {reward:.3}")
        return reward


class EtcTakeOffReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps

    def forward(self, etc_dis, ego_polygon, toll_info, raw_env, history_size, pred_ts, speed, stop_count):
        if etc_dis > 5.0 or etc_dis < -50.0:
            return 0.0

        ts = history_size + pred_ts
        dobj, dobj_next = get_xyyaw_from_polygon(ego_polygon[history_size + pred_ts - 1 : history_size + pred_ts + 1])
        take_off_reward = 0.0

        toll_sod = toll_info["sod"]
        dis_to_sod = (
            1000.0 if toll_sod is None else np.linalg.norm(toll_sod - dobj[:2]) * np.sign(toll_sod[0] - dobj[0])
        )

        toll_infos = raw_env[0]["toll_infos"]
        if toll_infos is None or len(toll_infos) < 1 or toll_infos["od"] is None:
            return 0.0

        toll_od = toll_infos["od"]
        if toll_od is not None and len(toll_od) > ts:
            dis_to_od = 1000.0 if toll_od[ts] is None else get_xyyaw_from_polygon(toll_od[ts])[0][0]
            if dis_to_sod > 5 and dis_to_od > 5 or (dis_to_sod < 0.0 and dis_to_od > 5.0):
                if speed < 4.0:
                    stop_count += 1
                    # print(f"Stop t: {pred_ts}, vel: {speed:.3}, count: {self.stop_count}")
                    # stop time > 5s = 5*5
                    if stop_count > 20:
                        take_off_reward = 5.0
                        # print(f"Stop true, time: {self.stop_count}")

                if speed > 6.0:
                    stop_count = 0
                    take_off_reward = 0.0

        return take_off_reward


class EtcMindistReward(BaseLoss):
    def forward(self, min_distance_list, min_distance_reward):
        """Forward."""
        min_distance = np.array(min_distance_list)
        for ind, min_dis in enumerate(min_distance):
            dis_penalty = max(0, 5 - 1.35 * min_dis) * 0.5
            min_distance_reward[ind + 1] = -dis_penalty
        return min_distance_reward


class NaviReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def get_stopline_pos(self, ego_polygon, stop_lines):
        """Get stopline pos.

        Args:
            ego_traj: n x 4 x 2
            stop_lines: n x 2
        Returns:
            pos: (x, y)  base on ego cooridinate
        """
        ego_traj = ego_polygon.mean(axis=1)  # n x 2
        indeices, _ = get_lines_distance(stop_lines, ego_traj[0])
        # check intersection
        intersected_traj_index = None
        intersected_stop_line_index = None
        find_flag = False
        intersection_p = np.array([])
        for index in indeices:
            nearest_stop_line = stop_lines[index]
            ego_traj_segments = split_points_to_segments(ego_traj)
            for i, segment in enumerate(ego_traj_segments):
                if segment_intersect(segment[0], segment[1], nearest_stop_line[0], nearest_stop_line[1]):
                    find_flag = True
                    intersected_stop_line_index = index
                    intersected_traj_index = i
                    break
            if find_flag:
                intersection_p = line_intersection_point(
                    ego_traj_segments[intersected_traj_index][0],
                    ego_traj_segments[intersected_traj_index][1],
                    stop_lines[intersected_stop_line_index][0],
                    stop_lines[intersected_stop_line_index][1],
                )
                return intersection_p
        # no intersection, calculate projection
        projection_p = np.array([])
        for index in indeices:
            nearest_stop_line = stop_lines[index]
            base_pt = ego_traj[-1]
            base_polygon = ego_polygon[-1]
            temp_projection_p = get_projection_on_segment(
                base_pt[0],
                base_pt[1],
                nearest_stop_line[0, 0],
                nearest_stop_line[0, 1],
                nearest_stop_line[1, 0],
                nearest_stop_line[1, 1],
            )
            if len(temp_projection_p) < 1:
                continue
            else:
                distance = ((base_pt[0] - temp_projection_p[0]) ** 2 + (base_pt[1] - temp_projection_p[1]) ** 2) ** 0.5
                theta = get_vector_angle(base_polygon[1:3, :].mean(axis=0) - base_pt, temp_projection_p - base_pt)
                if distance > 30.0 or 0.52 < abs(theta) < 2.61:
                    continue
                else:
                    projection_p = temp_projection_p
                    break
        return projection_p

    def forward(self, ego_polygon, gt_polygon, raw_env, stopline_crossings):
        """Forward."""
        # 简化：只考虑第一个停止线
        navi_road_sign_reward = 0.0  # 自车实际开出的方向与roadsign是否匹配
        navi_link_reward = 0.0  # 自车是否发生车道级偏航

        ego_path = ego_polygon.mean(axis=1)

        # 如果没穿过停止线，计算下最终位置距离停止线的距离
        reward_flag = False
        dist_to_stopline = 0
        if stopline_crossings:
            cross_ts, _, _ = stopline_crossings[0]
            reward_flag = True
        else:
            cross_ts = len(raw_env) - 1
        stopline_pos = self.get_stopline_pos(ego_polygon, raw_env[cross_ts]["stop_line"])
        if reward_flag and len(stopline_pos) == 0:
            stopline_pos = stopline_crossings[0][2]
        if len(stopline_pos):
            dist = calc_path_length_from_point(ego_polygon, stopline_pos)
            if dist < 0:
                dist_to_stopline = abs(dist)
                if dist_to_stopline < 10:
                    reward_flag = True

        if reward_flag:
            # 自车开过的roadsign
            road_sign_direction = 0b11111  # 默认全部可行
            road_sign_list = match_road_sign_from_path(ego_polygon, stopline_pos, raw_env[cross_ts])
            if road_sign_list:
                # 只使用最接近stopline的roadsign
                road_sign_ptype, _ = road_sign_list[-1]
                if road_sign_ptype in [905, 909]:  # 非机动车道，步道
                    road_sign_direction = 0b00000
                elif road_sign_ptype in road_arrow_to_lane_direction_mapping:
                    road_sign_direction = road_arrow_to_lane_direction_mapping[road_sign_ptype]
            road_sign_set = get_lane_direction(road_sign_direction)

            # 自车实际开出的方向
            new_path = ego_path[cross_ts:]
            if len(new_path) > 30:
                path_turn_type = determine_path_turn_type(new_path)
            else:
                path_turn_type = {Turntype.UNKNOWN}
            if path_turn_type != {Turntype.UNKNOWN}:
                if all(turn_type not in road_sign_set for turn_type in path_turn_type):
                    navi_road_sign_reward = -10.0

        # navilink reward
        # navi_link_reward_weight = 10.0
        # navi_link_reward =navi_link_reward_weight * self.calc_navi_link_reward(ego_polygon, gt_polygon, raw_env)
        return navi_road_sign_reward, navi_link_reward

class DangerLaneChangeReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps

    def split_lc_intervals(self, pred_ego_polygons, pred_raw_env):
        """划分出换道区间（自车压车道线）及所压车道线（截取第一次压线时自车指定范围内的部分车道线,是否为实线）"""
        lc_intervals = []
        cross_lane_line_info = {}

        cut_off_dist = 50.0
        extend_len = 5
        for ts, polygon in enumerate(pred_ego_polygons):
            filter_lines = []
            ego_center_pt = polygon.mean(axis=0).reshape(1, 2)

            for p, a in pred_raw_env[ts]["lane_lines"]:
                mask = np.linalg.norm(p[:, :2] - ego_center_pt, axis=1) < 5.0
                mask_2 = np.linalg.norm(p[:, :2] - ego_center_pt, axis=1) < cut_off_dist
                filter_pt = p[mask, :]
                filter_lines.append([p[mask, :2], a[mask, :], p[mask_2, :2]])

            for filter_pt, filter_attr, cut_off_pt in filter_lines:
                if filter_pt.shape[0] < 2:
                    continue
                polyline = LineString(filter_pt)
                ego_polygon = Polygon(polygon)
                if polyline.intersects(ego_polygon):
                    mid_idx = filter_pt.shape[0] // 2 if (filter_pt.shape[0] // 2) < filter_pt.shape[0] - 1 else filter_pt.shape[0] - 2
                    ref_yaw =  np.arctan2(filter_pt[mid_idx+1, 1] - filter_pt[mid_idx, 1], filter_pt[mid_idx+1, 0] - filter_pt[mid_idx, 0])
                    ref_point = np.array([filter_pt[mid_idx, 0], filter_pt[mid_idx, 1], ref_yaw])
                    is_solid_line = filter_attr[mid_idx, :][0] in [1, 5, 7, 8]
                    ref_point = np.array([filter_pt[mid_idx, 0], filter_pt[mid_idx, 1], ref_yaw])
                    cross_lane_line_info[ts] = [cut_off_pt, is_solid_line, ref_point]
                    if len(lc_intervals) == 0:
                        # 前向信息补全
                        start_ts = max(0, ts - extend_len)
                        for i in range(start_ts, ts):
                            cross_lane_line_info[i] = [cut_off_pt, is_solid_line, ref_point]
                        lc_intervals.append([start_ts, ts])
                    else:
                        last_lc_ts = lc_intervals[-1][1]
                        last_ref_point = cross_lane_line_info[last_lc_ts][2]

                        # 判定是否为同一根车道线
                        lon_dist, lat_dist = get_lon_lat_distance(ref_point, last_ref_point)
                        # 同一条车道线
                        if abs(lat_dist) < 1.75 and ts - last_lc_ts < extend_len:
                            for i in range(last_lc_ts, ts):
                                if i not in cross_lane_line_info:
                                    cross_lane_line_info[i] = [cut_off_pt, is_solid_line, ref_point]
                            lc_intervals[-1][1] = ts  # 同一根车道线，更新换道区间长度
                        # 切分新的换道区间
                        else:
                            # 上个换道区间后向信息补全
                            last_end_ts = min(len(pred_ego_polygons)-1, min(ts - extend_len, last_lc_ts + extend_len))
                            last_end_ts = max(last_end_ts, last_lc_ts)
                            for i in range(last_lc_ts, last_end_ts + 1):
                                if i not in cross_lane_line_info:
                                    cross_lane_line_info[i] = cross_lane_line_info[last_lc_ts]
                            lc_intervals[-1][1] = last_end_ts

                            # 新换道区间
                            start_ts = max(0, max(ts - extend_len, last_end_ts + 1))
                            for i in range(start_ts, ts):
                                if i not in cross_lane_line_info:
                                    cross_lane_line_info[i] = [cut_off_pt, is_solid_line, ref_point]
                            lc_intervals.append([start_ts, ts])
                    break

        # 最后一个换道区间后向信息补全
        if len(lc_intervals) > 0:
            last_ts = lc_intervals[-1][1]
            if last_ts in cross_lane_line_info:
                end_ts= min(len(pred_ego_polygons)-1, last_ts + extend_len)
                for i in range(last_ts + 1, end_ts + 1):
                    if i not in cross_lane_line_info:
                        cross_lane_line_info[i] = cross_lane_line_info[last_ts]
                lc_intervals[-1][1] = end_ts


        return lc_intervals, cross_lane_line_info

    def obj_relation_with_lane_line(self, lane_line_seg, obj_polygon):
        obj_center = obj_polygon.mean(axis=0).reshape(1, 2)
        min_idx = np.argmin(np.linalg.norm(lane_line_seg - obj_center, axis=1))
        closest_seg = lane_line_seg[min_idx : min_idx+2, :] if min_idx+1 < len(lane_line_seg) else lane_line_seg[min_idx-1 : min_idx+1, :]
        seg_yaw = np.arctan2(closest_seg[1, 1] - closest_seg[0, 1], closest_seg[1, 0] - closest_seg[0, 0])

        dx = obj_center[0, 0] - closest_seg[0, 0]
        dy = obj_center[0, 1] - closest_seg[0, 1]
        lat_dist = dy * np.cos(seg_yaw) - dx * np.sin(seg_yaw)

        return lat_dist, np.array([closest_seg[0, 0], closest_seg[0, 1], seg_yaw])


    def forward(self, ego_polygons, ego_speeds, raw_env, history_size):
        pred_ego_polygons = ego_polygons[history_size:]
        pred_raw_env = raw_env[history_size:]

        # 依据自车与车道线是否相交，划分出自车换道行为区间
        lc_intervals, cross_lane_line_info = self.split_lc_intervals(pred_ego_polygons, pred_raw_env)

        target_lane_lat_thresh = 2.5

        # 返回值
        danger_lc_result = {}
        pred_num = len(pred_ego_polygons)
        danger_lc_penalty = [0.0] * pred_num
        lc_status = [False] * pred_num
        lc_back_obj_id = [-1] * pred_num
        lc_back_safe_dist = [-1.0] * pred_num
        lc_ahead_obj_id = [-1.0] * pred_num
        is_danger_lc = [False] * pred_num

        # 筛选出目标车道上距离自车最近的侧后车
        for start_idx, end_idx in lc_intervals:
            lane_line_seg, is_solid_line, _ = cross_lane_line_info[start_idx]
            ego_lat_dist, _ = self.obj_relation_with_lane_line(lane_line_seg, pred_ego_polygons[start_idx])
            is_ego_in_left_side = ego_lat_dist > 0.0

            for ts in range(start_idx, end_idx + 1):
                # update 返回值
                lc_status[ts] = True

                # ego相关信息计算
                cur_ego_polygon = pred_ego_polygons[ts]
                cur_ego_direction = cur_ego_polygon[1,:] - cur_ego_polygon[0,:]
                cur_ego_direction = cur_ego_direction / np.linalg.norm(cur_ego_direction)
                cur_ego_center = cur_ego_polygon.mean(axis=0)
                cur_ego_v = ego_speeds[ts] / 3.6 # m/s
                lane_line_seg, is_solid_line, _ = cross_lane_line_info[ts]
                _, ego_ref_point = self.obj_relation_with_lane_line(lane_line_seg, cur_ego_polygon)
                ego_lon_dist, _ = get_lon_lat_distance(cur_ego_center, ego_ref_point)

                if cur_ego_v < 1.0:
                    continue

                #当前帧的动态目标
                _, dynamic_obs_info  = raw_env[0]["dobjs_full"][ts + history_size]
                dynamic_obs_polygons = raw_env[0]["dobjs_polygon"][ts + history_size]
                dynamic_obs_speed = np.linalg.norm(dynamic_obs_info[:,-2:], axis=1)
                dynamic_obs_tid = dynamic_obs_info[:,-3]

                #当前帧 目标车道 侧后交互目标 & 侧前交互目标 筛选
                target_behind_obs_tid = -1
                traget_behind_obs_lon_dist = 999.9
                target_behind_obs_polygon = None
                target_behind_obs_v = None

                target_ahead_obs_tid = -1
                target_ahead_obs_lon_dist = 999.9
                target_ahead_obs_v = None


                for i in range(len(dynamic_obs_polygons)):
                    cur_obs_polygon = dynamic_obs_polygons[i]
                    cur_obs_center = cur_obs_polygon.mean(axis=0)
                    cur_obs_tid = int(dynamic_obs_tid[i])

                    cur_obs_direction = cur_obs_polygon[1,:] - cur_obs_polygon[0,:]
                    cur_obs_direction = cur_obs_direction / max(np.linalg.norm(cur_obs_direction), 1e-3)
                    cos_ang = np.clip(cur_ego_direction @ cur_obs_direction, -1.0, 1.0)
                    ang_deg = np.degrees(np.arccos(cos_ang))
                    if ang_deg > 90.0:  # 过滤对向车
                        continue

                    obs_ego_dist = np.linalg.norm(cur_obs_center - cur_ego_center)
                    if obs_ego_dist > 100.0: # 过滤远距离车
                        continue

                    obs_lon_dist, _ = get_lon_lat_distance(cur_obs_center, ego_ref_point)
                    obs_lat_dist, _ = self.obj_relation_with_lane_line(lane_line_seg, cur_obs_polygon)
                    if abs(obs_lat_dist) > target_lane_lat_thresh:  # 过滤非目标车道车
                        continue

                    is_obs_before_ego = False
                    if ego_lon_dist < obs_lon_dist:  # 判定目标是前车还是后车
                        is_obs_before_ego = True

                    if is_ego_in_left_side == (obs_lat_dist > 0.0):
                        continue  #  过滤同侧车辆 （前车也被过滤,前车先靠ttcReward）

                    if is_obs_before_ego:
                        if (obs_lon_dist - ego_lon_dist) < target_ahead_obs_lon_dist:  # 对于侧前车和前车只用center 纵向距离筛可能存在一定风险
                            target_ahead_obs_lon_dist = obs_lon_dist - ego_lon_dist
                            target_ahead_obs_tid  = cur_obs_tid
                            target_ahead_obs_v = dynamic_obs_speed[i]
                    else:
                        if (ego_lon_dist - obs_lon_dist) < traget_behind_obs_lon_dist:
                            traget_behind_obs_lon_dist = ego_lon_dist - obs_lon_dist
                            target_behind_obs_tid = cur_obs_tid
                            target_behind_obs_polygon = cur_obs_polygon
                            target_behind_obs_v = dynamic_obs_speed[i]

                danger_lc_base_penalty = -10.0
                # 后向危险变道惩罚
                if target_behind_obs_tid != -1:
                    # update 返回值
                    lc_back_obj_id[ts] = target_behind_obs_tid

                    need_back_check = True
                    # 自车速度较低且非后方快速来车，不判定危险变道惩罚，避免丢失加塞博弈能力
                    if cur_ego_v < 5.0 and target_behind_obs_v - cur_ego_v < 5.0:
                        need_back_check = False

                    # 静止车 or 低速车 不判定危险变道惩罚，避免影响绕行能力
                    if target_behind_obs_v < 3.0:
                        continue

                    if need_back_check:
                        ego_poly = Polygon(cur_ego_polygon)
                        obj_poly = Polygon(target_behind_obs_polygon)
                        ego_obj_min_dist = ego_poly.distance(obj_poly)
                        # 危险变道惩罚计算
                        base_dist = 2.5 + target_behind_obs_v * 0.45  # 20kph->5.0  60kph->10.0
                        ttc_adjust_dist = (target_behind_obs_v - cur_ego_v) * 1.5 # 相对速度1.5s时距, 可能为负值
                        safe_dist = max(5.0, base_dist + ttc_adjust_dist)
                        lc_back_safe_dist[ts] = safe_dist

                        if ego_obj_min_dist < safe_dist:
                            danger_lc_penalty[ts] += danger_lc_base_penalty * (1 - ego_obj_min_dist/safe_dist)**2
                            # if is_solid_line:
                            #     danger_lc_penalty[ts] += danger_lc_base_penalty * (1 - ego_obj_min_dist/safe_dist)**2
                            if ego_obj_min_dist/safe_dist < 0.60: # 当前距离不足安全距离 60%
                                is_danger_lc[ts] = True

                # 前向危险变道目标id记录，后续在collisionReward 和 minDistanceReward 中对该目标padding
                if target_ahead_obs_tid != -1:
                    # update返回值
                    if target_ahead_obs_v > 5.0:  # 对高速目标才padding 避免丧失加塞插空能力
                        lc_ahead_obj_id[ts] = target_ahead_obs_tid

        danger_lc_result.update(
            {
                "danger_lc_penalty": danger_lc_penalty,
                "lc_status": lc_status,
                "lc_back_obj_id": lc_back_obj_id,
                "lc_back_safe_dist": lc_back_safe_dist,
                "lc_ahead_obj_id": lc_ahead_obj_id,
                "is_danger_lc": is_danger_lc,
            }
        )

        return danger_lc_result

class NaviLaneReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps
        self.STABLE_THRESHOLD = 0.75

    def calculate_joint_reward(self, current_idx, critical_idx, lateral_dist, spacing=2.0):
        """
        单公式、两段权重，25 m 过渡区
        current_idx: 轨迹点在中心线的序号
        critical_idx: 最晚变道点序号
        lateral_dist: 横向距离（绝对值，米）
        spacing: 中心线点间距（米/点）
        return: 平滑奖惩值
        """
        # 1. 纵向距离（米）
        remain_m = max(0.0, (critical_idx - current_idx) * spacing)

        # 2. 过渡因子 α ∈ [0,1]，sigmoid 平滑
        TRANSITION_M = 100.0
        ratio = remain_m / TRANSITION_M
        alpha = 1.0 / (1.0 + math.exp(-5.0 * (ratio - 0.5)))  # 越靠近 0 m，α 越小

        # 3. 两段函数
        def reward_zone(d):
            if d <= 0.5:      return  2.0
            elif d <= 2.0:    return  2.0 - 1.33 * (d - 0.5)
            elif d <= 4.0:    return -1.0 * (d - 2.0)
            else:             return -2.0

        def penalty_zone(d):
            if d <= 0.5:      return  4.0
            elif d <= 1.0:    return  4.0 - 6.0 * (d - 0.5)
            elif d <= 2.0:    return  1.0 - 6.0 * (d - 1.0)
            else:             return -19.0

        # 4. 平滑融合
        r = reward_zone(lateral_dist)
        p = penalty_zone(lateral_dist)
        return alpha * r + (1.0 - alpha) * p

    def forward(self, pred_polygon,gt_polygon, navi_centerline_list, is_highway_scene=False, is_od_scene_list=None, is_slow_scene_all=False, is_dead_car_scene=False):
        """基于位置index和最晚变道点index联合设计的平滑奖惩函数."""
        ego_path_info = get_xyyaw_from_polygon(pred_polygon)
        pred_num = len(ego_path_info)
        navi_lane_reward = [0.0] * pred_num
        navi_lane_change_reward = [0.0] * pred_num
        closest_navilane_lat_info = {}

        navi_centerline_list = [navi_lane for navi_lane in navi_centerline_list if navi_lane[2] == 0 or navi_lane[2] == 2]

        if len(navi_centerline_list) < 1:
            return navi_lane_reward, navi_lane_change_reward

        # 预处理导航中心线
        points_x_y_yaw_list = []
        for index, navi_lane in enumerate(navi_centerline_list):
            points_x_y_yaw = calc_path_point_heading(navi_lane[0][:, :2])
            points_x_y_yaw_list.append(points_x_y_yaw)

        # ego轨迹下采样
        ego_path_downsample = []
        ego_path_downsample.append([0, ego_path_info[0]])
        sample_distance_th = 10.0 if is_highway_scene else 3.0
        for i in range(1, len(ego_path_info)):
            c_distance = np.linalg.norm(ego_path_info[i, :2] - ego_path_downsample[-1][1][:2])
            if c_distance > sample_distance_th:
                ego_path_downsample.append([i, ego_path_info[i]])

        # 计算投影信息
        last_search_result_cache = {}
        lat_dist_info_all_time = []

        for i in range(len(ego_path_downsample)):
            pos = np.squeeze(np.array([ego_path_downsample[i][1][[0, 1, 6]]]))
            lat_dist_info_all_lane = []

            for lane_idx, navi_centerline_points in enumerate(points_x_y_yaw_list):
                if lane_idx not in last_search_result_cache:
                    min_point_index, min_dist = find_nearest_point(pos, navi_centerline_points, 1.047)
                else:
                    last_min_point_index = last_search_result_cache[lane_idx]
                    min_point_index, min_dist = find_nearest_point(
                        pos, navi_centerline_points, 1.047, last_min_point_index
                    )

                if min_point_index < 0:
                    continue

                last_search_result_cache[lane_idx] = min_point_index
                min_projection_distance = get_lat_distance(pos, navi_centerline_points[min_point_index])
                is_begin_point = min_point_index < 2
                is_end_point = min_point_index > (len(navi_centerline_points) - 3)

                # 关键：计算最晚变道点信息
                solid_line_start_index = (
                    navi_centerline_list[lane_idx][1]
                    if navi_centerline_list[lane_idx][1] is not None
                    else len(navi_centerline_points)
                )

                lat_dist_info = {
                    "lane_index": lane_idx,
                    "current_index": min_point_index,
                    "critical_index": solid_line_start_index,
                    "min_l2_distance": min_dist,
                    "lateral_distance": min_projection_distance,
                    "is_begin": is_begin_point,
                    "is_end": is_end_point,
                }

                lat_dist_info_all_lane.append(lat_dist_info)

            lat_dist_info_all_time.append(lat_dist_info_all_lane)

        # 新的联合奖励计算
        lane_change_interval = [None, None, None]
        init_lat_distance = {}
        target_navi_lane_index = None

        # 记录当前稳定锁定的导航车道索引，None表示未锁定
        current_stable_lane_idx = None
        enable_lane_lock = not is_slow_scene_all and not is_dead_car_scene # OD场景下禁用锁定，慢车场景禁用，死车场景禁用

        for i in range(len(ego_path_downsample)):
            ego_traj_index = ego_path_downsample[i][0]
            lat_dist_infos = lat_dist_info_all_time[i]

            best_lane_info = None

            if is_od_scene_list[ego_traj_index]: # 前方有阻挡OD，取消锁定
                current_stable_lane_idx = None

            # 强制锁定
            if enable_lane_lock and not is_od_scene_list[ego_traj_index] and current_stable_lane_idx is not None:
                # 尝试在投影列表中找到锁定的车道
                for lat_info in lat_dist_infos:
                    if lat_info["lane_index"] == current_stable_lane_idx:
                        # 检查车道是否“完结” (到了导航线的尽头)
                        if lat_info["is_end"]:
                             # 车道结束，自然解锁，允许寻找下一条车道
                             current_stable_lane_idx = None
                        else:
                             # 即使距离很远 (比如 3.5m)，只要车道没结束，就强制使用
                             best_lane_info = lat_info
                        break

                # 如果遍历完发现锁定的车道id找不到了也解锁
                if best_lane_info is None and current_stable_lane_idx is not None:
                     current_stable_lane_idx = None

            # 全局搜索 (仅当未锁定或解锁时执行)
            if best_lane_info is None:
                min_total_distance = float("inf")
                for lat_info in lat_dist_infos:
                    # 过滤距离起止点过远的车道
                    if (lat_info["is_begin"] or lat_info["is_end"]) and lat_info["min_l2_distance"] > 2.0:
                        continue

                    if lat_info["min_l2_distance"] < min_total_distance:
                        min_total_distance = lat_info["min_l2_distance"]
                        best_lane_info = lat_info

            # 如果还是找不到，跳过
            if best_lane_info is None:
                continue

            # 状态更新：建立锁定
            current_lat_dist = abs(best_lane_info["lateral_distance"])

            # 只有在未锁定的情况下，且距离足够近，才建立新的锁定
            # 一旦锁定，就不会因为距离变大而解锁 (除非遇到 is_end)
            if not is_od_scene_list[ego_traj_index] and enable_lane_lock and current_stable_lane_idx is None and current_lat_dist < self.STABLE_THRESHOLD:
                current_stable_lane_idx = best_lane_info["lane_index"]

            reward = self.calculate_joint_reward(
                best_lane_info["current_index"],
                best_lane_info["critical_index"],
                current_lat_dist,
                spacing=2.0,
            )

            navi_lane_reward[ego_traj_index] = reward

            # 以下逻辑用于计算换道奖励和危险检查
            if best_lane_info['current_index'] < best_lane_info['critical_index']:
                closest_navilane_lat_info[ego_traj_index] = current_lat_dist
                cur_navi_lane_idx = best_lane_info['lane_index']
                # 记录初次横向位置
                if cur_navi_lane_idx not in init_lat_distance:
                    init_lat_distance[cur_navi_lane_idx] = current_lat_dist
                # 目标导航位置变更，清空导航变道奖励段信息
                if cur_navi_lane_idx != target_navi_lane_index and lane_change_interval[2] is None:
                    lane_change_interval = [None, None, None]
                    target_navi_lane_index = cur_navi_lane_idx
                if init_lat_distance[cur_navi_lane_idx] > 1.75 and lane_change_interval[0] is None:
                    lane_change_interval[0] = ego_traj_index
                if (current_lat_dist < 3.0 and
                    lane_change_interval[0] is not None and
                    lane_change_interval[1] is None and
                    init_lat_distance[cur_navi_lane_idx] - abs(best_lane_info['lateral_distance']) > 0.5):  # 相比于初始位置有明确横向靠近动作
                    lane_change_interval[1] = ego_traj_index
                if (current_lat_dist < 0.75 and
                    lane_change_interval[1] is not None and
                    lane_change_interval[2] is None):
                    lane_change_interval[2] = ego_traj_index

            # 严重偏航终止条件
            if (
                best_lane_info["current_index"] >= best_lane_info["critical_index"]
                and current_lat_dist > 2.5
            ):
                navi_lane_reward[ego_traj_index] = -19

        # navi_lane_change_reward
        lane_change_success_total_reward = 50.0 * 2
        if (lane_change_interval[1] is not None) and (lane_change_interval[2] is not None):
            sub_reward = lane_change_success_total_reward / (lane_change_interval[2] - lane_change_interval[1] + 1)
            for i in range(lane_change_interval[1], lane_change_interval[2] + 1):
                # if is_od_scene_list[i]: # OD场景下换道奖励下午换道奖励
                #     continue
                navi_lane_change_reward[i] = sub_reward

        return navi_lane_reward, navi_lane_change_reward


class WrongWayReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def get_pedestrian_crossing_pos(self, pred_polygon, raw_env, threshold=0.5):
        """判断车辆是否压过人行道，只记录首次压过同一块人行道的信息.

        Args:
            pred_polygon: np.ndarray [T, 4, 2] 车辆 polygon 轨迹
            raw_env: list of dicts，长度为 T，每个时刻的环境信息
            threshold: 距离阈值，用于判断是否是“同一块人行道”

        Returns:
            crossings: list of (frame_idx, pedestrian_crossing_id, intersection_point)
        """
        crossings = []
        seen_crossings = []

        def get_centroid(polygon):
            return np.mean(np.array(polygon), axis=0)

        def find_existing_crossing_id(center):
            for idx, existing_center in enumerate(seen_crossings):
                if np.linalg.norm(existing_center - center) < threshold:
                    return idx
            return None

        for t in range(len(raw_env)):
            ego_poly = pred_polygon[t]
            ego_segs = polygon_to_segments(ego_poly)

            road_signs = raw_env[t]["road_sign"]
            for rs in road_signs:
                if rs[1] != 300:
                    continue
                ped_polygon = np.array(rs[0])[:, :2]

                if ped_polygon.shape != (4, 2):
                    continue

                center = get_centroid(ped_polygon)
                crossing_id = find_existing_crossing_id(center)
                if crossing_id is None:
                    crossing_id = len(seen_crossings)
                    seen_crossings.append(center)

                # 如果这块人行道已经记录过，跳过
                if any(item[1] == crossing_id for item in crossings):
                    continue

                ped_segs = polygon_to_segments(ped_polygon)
                for seg1 in ego_segs:
                    for seg2 in ped_segs:
                        if segment_intersect(seg1[0], seg1[1], seg2[0], seg2[1]):
                            pt = line_intersection_point(
                                np.array(seg1[0]), np.array(seg1[1]), np.array(seg2[0]), np.array(seg2[1])
                            )
                            if pt is not None:
                                crossings.append((t, crossing_id, pt))
                                break
                    else:
                        continue
                    break
        return crossings

    def check_crossing_order(self, stopline_crossings, pedestrian_crossings, time_threshold=15, dist_threshold=10.0):
        """判断 ego 是否先压过 stopline 再压过人行道，且时间和位置接近.

        Args:
            stopline_crossings: list of (t_stopline, stopline_pos)
            pedestrian_crossings: list of (t_pc, idx, pc_pos)
            time_threshold: 最大允许的帧间隔
            dist_threshold: 最大允许的空间距离

        Returns:
            result: True if ego先压stopline再压人行道，且时间/空间接近
        """
        if not stopline_crossings or not pedestrian_crossings:
            return False, 0

        for t_pc, _, pos_pc in pedestrian_crossings:
            for t_stop, _, pos_stop in stopline_crossings:
                if t_stop > t_pc:
                    dt = t_stop - t_pc
                    dist = np.linalg.norm(pos_stop - pos_pc)
                    if dt <= time_threshold and dist <= dist_threshold:
                        return True, t_stop
        return False, 0

    def forward(self, pred_polygon, raw_env, stopline_crossings):
        """Forward."""
        pedestrian_crossings = self.get_pedestrian_crossing_pos(pred_polygon, raw_env)
        wrong_way, t_stop = self.check_crossing_order(stopline_crossings, pedestrian_crossings)
        if wrong_way:
            return 10, t_stop
        else:
            return 0, t_stop

class GateMachineReward(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        # 奖励参数
        self.reward_weight = 1.0

        # 闸机多边形膨胀缓冲（米）
        self.gate_buffer_m = 0.2

        # 距离筛选（米）
        self.fast_dist_thresh = float(8.0)

        # 类别参数
        self.GATE_CID = int(29)
        self.SID_CLOSED = int(10001)
        self.SID_OPENING = int(10002)
        self.SID_OPENED = int(10003)

        # 默认不把 opening 当作 open
        self.treat_opening_as_open = False

        self.output_debug = False

        # 进度奖励
        self.k_progress = 0.2
        self.progress_clip = 0.3
        self.not_cross_penalty = -0.05

        # 成功失败奖励
        self.r_success_event = 15
        self.r_illegal_event = -20
        self.r_stuck_event = -10

        # 人驾约束相关参数
        self.band_D = 18.0 # 距离闸机距离18m
        self.band_w_near = 0.3
        self.band_w_far = 1.6

        self.band_sigma = 0.5
        self.band_lam = 2.0

        # 碰撞/危险 -> 正奖励缩放
        self.collision_beta = 2.0        # 衰减强度：越大越狠
        self.collision_min_scale = 0.01   # 最小缩放（0=最严重时正奖励归零）
        self.collision_window = 1        # 取当前帧 or 最近窗口均值（1表示只看当前帧）

    def _cross2d(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculates the 2D cross product of two 2D vectors."""
        return v1[0] * v2[1] - v1[1] * v2[0]

    def _project_point_to_polyline_vectorized(
        self, c_gt: np.ndarray, s_gt: np.ndarray, p: np.ndarray
    ) -> Tuple[float, float]:
        # 1. 准备数据：A是所有线段起点，B是所有线段终点
        A = c_gt[:-1]
        B = c_gt[1:]

        # 2. 计算线段向量 AB 和 向量 AP
        AB = B - A
        AP = p - A  #利用广播机制，p(2,) 会被减去 A中的每一行

        # 3. 计算 AB 的长度平方 (N-1,)
        ab_len2 = np.einsum('ij,ij->i', AB, AB)

        valid_mask = ab_len2 > 1e-9
        if not np.any(valid_mask):
              return 0.0, 0.0

        # 只计算有效线段
        A = A[valid_mask]
        AB = AB[valid_mask]
        AP = AP[valid_mask]
        ab_len2 = ab_len2[valid_mask]
        s_gt_valid = s_gt[:-1][valid_mask]

        # 4. 计算投影比例 t (N-1,)
        t = np.einsum('ij,ij->i', AP, AB) / ab_len2

        # 5. 截断 t 到 [0, 1] 区间
        t = np.clip(t, 0.0, 1.0)

        # 6. 计算投影点 Q 和 距离向量 D
        Q = A + t[:, np.newaxis] * AB
        D = p - Q

        # 7. 计算距离平方并找到最小值
        dist2 = np.einsum('ij,ij->i', D, D)
        min_idx = np.argmin(dist2) # 找到最近线段的索引

        # 8. 提取最优结果
        best_t = t[min_idx]
        best_seg_len = np.sqrt(ab_len2[min_idx])

        # 计算 s_pred
        best_s = s_gt_valid[min_idx] + best_t * best_seg_len

        # 计算 d_lat (需要 best_D 和 best_tangent)
        best_D = D[min_idx]
        best_AB = AB[min_idx]

        # 计算切线方向
        tangent = best_AB / max(1e-6, best_seg_len)

        # 使用 cross2d 计算 signed distance
        best_d_lat = self._cross2d(tangent, best_D)

        return best_s, best_d_lat

    def _build_s_gt_vectorized(self, c_gt: np.ndarray) -> np.ndarray:
        """c_gt: (M,2)"""
        seg_vecs = c_gt[1:] - c_gt[:-1]
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        s_gt = np.zeros((c_gt.shape[0],), dtype=np.float64)
        s_gt[1:] = np.cumsum(seg_lengths)
        return s_gt

    def _band_width(self, dist_to_gate: float) -> float:
        r = float(np.clip(dist_to_gate / self.band_D, 0.0, 1.0))
        return float(self.band_w_near + (self.band_w_far - self.band_w_near) * r)

    def _band_penalty(self, d_lat: float, dist_to_gate: float) -> float:
        w = self._band_width(dist_to_gate)
        z = max(0.0, (abs(d_lat) - w) / max(1e-6, self.band_sigma))
        # 饱和二次：小超出近似二次，大超出饱和到 -band_lam
        return float(-self.band_lam * (z * z) / (1.0 + z * z))


    def _is_open(self, sid: int) -> bool:
        if sid == self.SID_OPENED:
            return True
        if self.treat_opening_as_open and sid == self.SID_OPENING:
            return True
        return False

    def _safe_poly(self, poly_xy: np.ndarray) -> Optional[ShpPolygon]:
        """poly_xy: (M,2)"""
        try:
            if poly_xy is None or len(poly_xy) < 3:
                return None
            shp = ShpPolygon(poly_xy)
            if shp.is_empty:
                return None
            if not shp.is_valid:
                shp = shp.buffer(0)  # 修复自交
                if shp.is_empty:
                    return None
            return shp
        except Exception:
            return None

    def _get_gates_in_frame(self, env_frame: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        只解析“当前帧”环境里的闸机:
        返回 gates: {gate_id: {sid, center(2,), poly(M,2), shp, shp_buf}}
        """
        gates: Dict[int, Dict[str, Any]] = {}
        if ("sobjs" not in env_frame) or ("sobjs_polygon" not in env_frame):
            return gates
        sobjs = env_frame["sobjs"]
        sobjs_polygon = env_frame["sobjs_polygon"]
        if sobjs is None or sobjs_polygon is None:
            return gates

        for i, obj in enumerate(sobjs):
            try:
                cid = int(obj[7])
                if cid != self.GATE_CID:
                    continue
                gid = int(obj[9])
                sid = int(obj[8])

                poly = np.asarray(sobjs_polygon[i], dtype=np.float64)
                center = np.mean(poly, axis=0)

                shp = self._safe_poly(poly)
                if shp is None:
                    continue

                # 预先 buffer 一次，避免 detect 时重复 buffer
                shp_buf = shp.buffer(self.gate_buffer_m)

                gates[gid] = dict(
                    sid=sid,
                    center=center,
                    poly=poly,
                    shp=shp,
                    shp_buf=shp_buf,
                )
            except Exception:
                continue
        return gates

    def _detect_cross_one_step(
        self,
        prev_poly: np.ndarray,
        curr_poly: np.ndarray,
        gates: Dict[int, Dict[str, Any]],
        is_pred: bool,
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """
        用“中心点线段与(闸机polygon buffer)相交”判定穿闸。
        注意: gates 必须来自同一帧（由调用方保证），不跨帧混用。
        """
        if not gates:
            return None, None

        best_gid = None
        best_info = None
        best_dist = 1e9

        # 后续计算相交可能使用polygon
        prev_c = np.mean(prev_poly, axis=0)
        curr_c = np.mean(curr_poly, axis=0)

        seg = LineString([prev_c.tolist(), curr_c.tolist()])
        if seg.length < 1e-6:
            seg = Point(curr_c.tolist()).buffer(1e-6)

        curr_pt = Point(curr_c.tolist())

        for gid, info in gates.items():
            gate_center = info["center"]
            # 快速距离筛选
            if np.linalg.norm(curr_c - gate_center) > self.fast_dist_thresh:
                continue

            gate_test = info.get("shp_buf", None)
            if gate_test is None:
                gate_shp: ShpPolygon = info["shp"]
                gate_test = gate_shp.buffer(self.gate_buffer_m)

            if is_pred:
                # 预测轨迹时候使用polygon相交计算
                ego_shp = Polygon(curr_poly)
                if not ego_shp.intersects(gate_test):
                    continue
                d = 0
            else:
                # 关键判定: 线段是否与闸机区域相交
                if not seg.intersects(gate_test):
                    continue

                # 多个 gate 都相交时，选“当前车位置”距离最近的 gate
                d = gate_test.distance(curr_pt)
            if d < best_dist:
                best_dist = d
                best_gid = gid
                best_info = info
            if is_pred:
                break

        return best_gid, best_info

    def _build_gt_gate_seq_by_id(
        self,
        ego_gt_polygon: np.ndarray,
        raw_env: List[Dict[str, Any]],
        pred_num: int,
    ):
        # 1) 预取每帧 gates, 和外面的保持一致
        gates_by_t = [
            self._get_gates_in_frame(env) for env in raw_env[:pred_num]
        ]

        # 遍历gates_by_t，打印每一帧的中心点，ID，和状态
        if self.output_debug:
            for t, gates_t in enumerate(gates_by_t):
                for gid, ginfo in gates_t.items():
                    logger.info(f"ori: frame {t}, gate_id {gid}, sid {ginfo['sid']}, center {ginfo['center']}")

        # 2) 用 GT 找到“穿过的 gate_id”（第一次穿闸的那个）
        target_gid: Optional[int] = None
        ego_traj_num = ego_gt_polygon.shape[0]

        if self.output_debug:
            for t in range(1, ego_traj_num):
                if t > len(gates_by_t) - 1:
                    break
                gates_t = gates_by_t[t]
                if not gates_t:
                    continue
                gid, _ = self._detect_cross_one_step(
                    ego_gt_polygon[t - 1], ego_gt_polygon[t], gates_t, False
                )
                if gid is not None:
                    target_gid = int(gid)
                    logger.info(f"GateMachineReward: detected GT gate crossing at t={t}, gid={target_gid}.")
                    # 打印选中的gate的位置和gid
                    logger.info(f"GateMachineReward: GT gate info: center={gates_t[gid]['center']}, poly={gates_t[gid]['poly']}.")

        # 记录碰撞时候的时间戳
        last_gate_t = None
        for t in range(1, ego_traj_num):
            gates_t = gates_by_t[t]
            if gates_t:
                last_gate_t = gates_t
                break
        cross_time = None
        for t in range(1, ego_traj_num):
            if t > len(gates_by_t) - 1:
                break
            gates_t = gates_by_t[t]
            if not gates_t:
                gates_t = last_gate_t
            gid, _ = self._detect_cross_one_step(
                ego_gt_polygon[t - 1], ego_gt_polygon[t], gates_t, False
            )
            last_gate_t = gates_t
            if gid is not None:
                target_gid = int(gid)
                cross_time = t
                break

        if self.output_debug:
            logger.info(f"GateMachineReward: final target_gid={target_gid}.")
        if target_gid is None:
            return [None] * pred_num, None

        # 3) 按 gate_id 逐帧索引，构造真值序列;
        # @todo， 这里本质上已经在同一个坐标系下了，可以做一个局部的融合，然后把所有的帧填补到每一帧，前处理做
        # 最前面增加last作为初始记录
        last_valid_gate_info = None
        for t in range(pred_num):
            if target_gid in gates_by_t[t]:
                last_valid_gate_info = gates_by_t[t][target_gid]
                break
        if last_valid_gate_info is None:
            return [None] * pred_num, None
        # 填充某些帧无闸机
        gt_gate_seq: List[Optional[Dict[str, Any]]] = [None] * pred_num
        for t in range(pred_num):
            gate_info = gates_by_t[t].get(target_gid, None)
            if gate_info is not None:
                last_valid_gate_info = gate_info
            else:
                gate_info = last_valid_gate_info
            if self.output_debug:
                logger.info(f"GateMachineReward: gt_gate_seq t={t}, gid={target_gid}, info_center={gate_info['center']}, poly={gate_info['poly']}, gate_is_none={gate_info is None}")
            gt_gate_seq[t] = {target_gid: gate_info}

        return gt_gate_seq, cross_time

    def _build_gt_gate_seq_by_fusion(self, fusion: Any, pred_num: int):
        """
        Use GateFusionOutput to build a per-frame gate sequence with the same
        structure as _build_gt_gate_seq_by_id.
        """
        if fusion is None or (not fusion.valid()):
            return [None] * pred_num, None

        fused_center = fusion.fused_center
        fused_poly = fusion.fused_poly
        fused_sid = fusion.fused_sid
        cross_time = fusion.cross_time_gt

        if fused_center is None or fused_poly is None or fused_sid is None:
            return [None] * pred_num, None

        T = int(
            min(
                pred_num,
                fused_center.shape[0],
                fused_poly.shape[0],
                fused_sid.shape[0],
            )
        )
        if T <= 0:
            return [None] * pred_num, None

        shp = self._safe_poly(fused_poly[0])
        if shp is None:
            return [None] * pred_num, None
        shp_buf = shp.buffer(self.gate_buffer_m)

        gid = 0  # synthetic id for fusion gate
        gt_gate_seq: List[Optional[Dict[int, Dict[str, Any]]]] = [None] * pred_num
        for t in range(pred_num):
            idx = min(t, T - 1)
            ginfo = dict(
                sid=int(fused_sid[idx]),
                center=fused_center[idx],
                poly=fused_poly[idx],
                shp=shp,
                shp_buf=shp_buf,
            )
            gt_gate_seq[t] = {gid: ginfo}

        if self.output_debug:
            logger.info(
                "GateMachineReward: fusion gate_id=%d, cross_time=%s, sid_first=%d.",
                gid,
                str(cross_time),
                int(fused_sid[0]),
            )

        if cross_time is None:
            return gt_gate_seq, None
        return gt_gate_seq, int(cross_time)

    def _build_safe_weight_from_penalty(
        self,
        pred_num: int,
        reward_results: Dict[str, Any],
        start_time: Optional[int],
        end_time: Optional[int],
    ) -> np.ndarray:
        safe_w = np.ones(pred_num, dtype=np.float64)

        collision_pen = reward_results.get("collision_reward", None)
        ttc = reward_results.get("ttc_reward", None)
        min_dist = reward_results.get("min_distance_reward", None)

        arrs = [x for x in (collision_pen, ttc, min_dist) if isinstance(x, np.ndarray)]
        pen_arr = np.minimum.reduce(arrs) if arrs else None
        if not isinstance(pen_arr, np.ndarray):
            return safe_w

        # 只在 [s, e) 内生效
        s = int(np.clip(start_time if start_time is not None else 0, 0, pred_num))
        e = int(np.clip(end_time if end_time is not None else pred_num, 0, pred_num))
        if e <= s:
            return safe_w

        max_t = min(e, pen_arr.shape[0] - 1)  # t 的最大合法值，使得 t+1 < pen_arr.shape[0]
        if max_t <= s:
            return safe_w

        pen_step = pen_arr[s + 1 : max_t + 1].astype(np.float64, copy=False)
        severity = np.maximum(0.0, -pen_step)
        safe_w[s:max_t] = np.exp(-float(self.collision_beta) * severity)

        safe_w = np.clip(safe_w, float(self.collision_min_scale), 1.0)
        return safe_w

    def forward(self, pred_polygon, ego_gt_polygon, raw_env, reward_results):
        """
        Args:
            pred_polygon: np.ndarray [T,4,2]
            ego_gt_polygon: np.ndarray [T,4,2]
            raw_env: list[dict] [T]
        Returns:
            np.ndarray [T] 的逐帧 reward
        """
        pred_num = pred_polygon.shape[0]

        reward = np.zeros(pred_num, dtype=np.float64)
        lat_gt_reward = np.zeros(pred_num, dtype=np.float64)
        progress_reward = np.zeros(pred_num, dtype=np.float64)

        # 增加warning信息
        # uuid = raw_env[0]['uuid'] if raw_env else 'unknown'
        # if pred_num != len(raw_env):
        #     logger.warning(
        #         f"GateMachineReward: pred_num ({pred_num}) != raw_env len ({len(raw_env)}), uuid ({uuid})."
        #     )

        # --- Step 1: 抽 GT 穿闸事件（真值 gate_id + 属性） ---
        # gt_events_ori, cross_time_ori = self._build_gt_gate_seq_by_id(ego_gt_polygon, raw_env, pred_num)
        # gt_events, cross_time = self._build_gt_gate_seq_by_id(ego_gt_polygon, raw_env, pred_num)
        if ego_gt_polygon.shape[0] + 1 == pred_num:
            # 兼容性处理，GT轨迹比预测轨迹短1帧的情况，简单复制最后一帧GT
            ego_gt_polygon = np.concatenate([ego_gt_polygon, ego_gt_polygon[-1:]], axis=0)
        gate_fusion = build_gate_gt_fusion(
            ego_gt_polygon,
            raw_env,
            pred_num=pred_num
        )
        gt_events, cross_time = self._build_gt_gate_seq_by_fusion(gate_fusion, pred_num)
        if all(e is None for e in gt_events) or cross_time is None:
            return reward, None

        # --- Step 2: 遍历预测轨迹，预计算ego-gt和ego-gate ---
        ego_front_center = (pred_polygon[:, 1, :] + pred_polygon[:, 2, :]) / 2.0  # (T,2)
        dist_to_gate = np.zeros(pred_num, dtype=np.float64) + 1e9  # 计算车头到闸机中心的距离
        dist_pred_to_gt_lat = np.zeros(pred_num, dtype=np.float64) # 计算预测轨迹偏离GT的横向距离

        s_ego_pred = np.zeros(pred_num, dtype=np.float64)  # 预测车头投影得到的弧长
        ds_to_gate = np.zeros(pred_num, dtype=np.float64)  # 沿路剩余距离：ds_to_gate = s_gate - s_ego_pred
        s_gate_path = gate_fusion.s_gate[:pred_num] # gate所在gt的距离s

        gt_center = ego_gt_polygon.mean(axis=1)  # (T,2)
        s_gt = self._build_s_gt_vectorized(gt_center)

        # 增加保护，不满足条件直接返回，在checker端也要过滤这种数据
        if s_gate_path.shape[0] < pred_num or gt_center.shape[0] < pred_num:
            return reward, gt_events

        for i in range(pred_num):
            _, d_lat = self._project_point_to_polyline_vectorized(gt_center, s_gt, ego_front_center[i])
            dist_pred_to_gt_lat[i] = d_lat

            s_ego, _ = self._project_point_to_polyline_vectorized(gt_center, s_gt, ego_front_center[i])
            s_ego_pred[i] = s_ego
            ds_to_gate[i] = float(s_gate_path[i] - s_ego)  # gate所在的s - 车头所在的s
            if gt_events[i] is not None:
                for gid, ginfo in gt_events[i].items():  # @todo, 考虑闸机的检测不稳定丢失；
                    gate_center = ginfo["center"]
                    dist = np.linalg.norm(ego_front_center[i] - gate_center)
                    if dist < dist_to_gate[i]:
                        dist_to_gate[i] = dist

        cross_t = None
        crossed = False
        cross_open = None
        start_t = None
        for t in range(1, pred_num):
            gates_t = gt_events[t]
            gid, ginfo = self._detect_cross_one_step(pred_polygon[t - 1], pred_polygon[t], gates_t, True)
            if gid is None or ginfo is None:
                continue

            crossed = True
            cross_t = t
            cross_open = self._is_open(int(ginfo["sid"]))
            break

        # --- Step 3: 遍历预测轨迹，稠密奖励，奖励向闸机方向移动的好的行为，惩罚偏离自车轨迹的坏的行为 ---
        end_t = cross_t if crossed else pred_num
        # 计算start_t
        start_t = next((i for i in range(end_t) if dist_to_gate[i] < self.band_D), 0)

        # 根据碰撞指标计算衰减权重
        safe_w = self._build_safe_weight_from_penalty(pred_num, reward_results, start_t, end_t)
        for t in range(end_t):
            # 在闸机一定范围内才处理跟随人驾和闸机引力
            if dist_to_gate[t] < self.band_D:
                band_penalty = self._band_penalty(dist_pred_to_gt_lat[t], dist_to_gate[t]) # 惩罚项，惩罚距离GT远的
                reward[t] += band_penalty
                lat_gt_reward[t] = band_penalty
                if t >= 1:
                    dp_ds = float(ds_to_gate[t - 1] - ds_to_gate[t])
                    # 防止异常大跳变
                    dp_ds = 0.0 if dp_ds > 200 else dp_ds
                    # 只奖励向闸机推进（不奖励后退/犹豫）
                    dp_ds = max(0.0, dp_ds)

                    r_progress = self.k_progress * dp_ds
                    r_progress = float(np.clip(r_progress, -self.progress_clip, self.progress_clip))
                    progress_reward[t] = r_progress

                    r_progress *= float(safe_w[t])
                    reward[t] += r_progress

        # 处理穿闸机的情况
        if crossed and cross_t is not None:
            if cross_open:
                W = 10
                w0 = max(0, int(cross_t) - W + 1)
                w_win = safe_w[w0:int(cross_t) + 1]

                eps = 1e-6
                reward_weight = float(np.exp(np.mean(np.log(np.clip(w_win, eps, 1.0)))))

                reward[cross_t] += self.r_success_event * reward_weight
                # @todo, 成功后分摊少量到窗口（改善 advantage 的时间分配）
                # logger.warning(f"GateMachineReward: SUCCESS cross_t={cross_t}, uuid={uuid}.")
            else:
                # reward[cross_t] += self.r_illegal_event
                # logger.warning(f"GateMachineReward: ILLEGAL cross_t={cross_t}, uuid={uuid}.")
                t_end = int(np.clip(int(cross_t), 0, pred_num - 1))
                t_start = max(0, t_end - 3)
                window_lengths = 10
                t_start = max(0, t_end - window_lengths)
                scan_end = min(len(lat_gt_reward), t_end + 1)
                for i in range(scan_end):
                    if abs(lat_gt_reward[i]) > 1e-6:
                        t_start = max(t_start, i)
                        break
                span = t_end - t_start + 1
                w = np.linspace(1.0, 4.0, span, dtype=np.float32)
                w = w / (w.sum() + 1e-6)
                reward[t_start:t_end + 1] += self.r_illegal_event * w

            return reward * self.reward_weight, gt_events

        # 处理不穿闸机的情况，应该如何增加卡住的惩罚
        if not crossed:
            t0 = None
            for i in range(len(lat_gt_reward)):
                if np.abs(lat_gt_reward[i]) > 1e-6:
                    t0 = min(i, pred_num - 1)
                    break

            if t0 is None:
                for i in range(len(progress_reward) - 3, -1, -1):
                    if progress_reward[i] > 1e-6 and progress_reward[i+1] < 1e-6 and progress_reward[i+2] < 1e-6:
                        t0 = i + 1
                        break

            if t0 is None:
                for i in range(pred_num):
                    if dist_to_gate[i] > self.band_D:
                        t0 = min(i, pred_num - 1)
                        break

            t0 = cross_time if t0 is None else t0

            # 3) 只对 [t0, end) 分摊一个固定总惩罚 B_fail（避免轨迹越长罚越多）
            L = pred_num - t0
            w = np.linspace(0.2, 1.0, L)
            w = w / (w.sum() + 1e-12)
            reward[t0:] += self.r_stuck_event * w

        return reward * self.reward_weight, gt_events

class TrafficLightReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def get_tld_direct_select(self, raw_env, tld, tldcolor):
        # TODO: 红绿灯不再绑定到到单个导航，需要自己确定自车所在的车道来获取红绿灯状态
        # 这里先暂时使用导航信息来获取红绿灯状态
        ##用来确定是否到达路口
        # 这里先暂时使用导航信息来获取红绿灯状态
        lane_infos = raw_env["navi_infos"]["lane_infos"]
        if len(lane_infos) > 0:
            highline_directions = [info[1] for info in lane_infos if info[1]]
            if highline_directions:
                highline_direction = highline_directions[0]
                direct_light = LANE_DIRECTION_TO_TLD_ID_MAPPING[highline_direction]
                tld_select = tld[direct_light]
                tld_color_select = tldcolor[tld_select]
            else:
                tld_color_select = "gray"
                direct_light = 4
        else:
            tld_color_select = "gray"  # 还没到路口
            direct_light = 4
        return tld_color_select, direct_light

    def get_scene_class(self, raw_env):
        ###scene_id 0表示无待转区， 1表示有待转区,且未进入待转区 2.表示有待转区，且已经进入待转区
        scene_id = 0
        ##如果有两条停止线，一条是待转区的，一条是normal的，且绿灯为直行，自车为左转，左转红灯亮了，直行绿灯亮了，自车速度>0,给正向reward 10
        wm_stoplines = raw_env["egopath_stoplines"]
        egopath_stoplines_ids = []
        egopath_stoplines_points = []
        for wm_stopline_junc in wm_stoplines:
            if len(wm_stopline_junc) == 0:
                continue
            t = wm_stopline_junc[2]
            p = wm_stopline_junc[0]
            egopath_stoplines_ids.append(t)
            egopath_stoplines_points.append(p)
        if len(egopath_stoplines_ids) > 0:
            if len(egopath_stoplines_ids) == 2 and 2 in egopath_stoplines_ids:
                ##这是一个待转区，且自车未进入待转区
                scene_id = 1
            if len(egopath_stoplines_ids) == 1 and 2 in egopath_stoplines_ids:
                scene_id = 2
        return scene_id, egopath_stoplines_points

    def judge_ego_has_front_obj(self, raw_env_seq, ego_polygon, ts):
        has_front_obj = False
        curr_ego_center = ego_polygon[1].mean(axis=0)
        if ts < len(raw_env_seq[0]["dobjs_full"]):
            _, dobjs = raw_env_seq[0]["dobjs_full"][ts]
            dobjs_polygon = raw_env_seq[0]["dobjs_polygon"][ts]
            for dobj, dobj_ploy in zip(dobjs, dobjs_polygon):
                dobj_x = dobj[0]
                dobj_y = dobj[1]
                dobj_l = dobj[3]
                dists = np.linalg.norm(dobj_ploy - curr_ego_center, axis=1)
                idx = np.argmin(dists)  # 最近点的索引
                min_dist = dists[idx]
                dx = dobj_x - curr_ego_center[0]
                dy = dobj_y - curr_ego_center[1]
                if dx > 2.5 and abs(dy) < 1.5 and min_dist < 20:
                    has_front_obj = True
                    break
        # sobjs = raw_env_seq[ts]["sobjs_polygon"]
        _, sobjs = filter_opened_gate_sobj(raw_env_seq[ts])
        for sobj in sobjs:
            sod_xy = sobj.mean(axis=0)
            sod_dists = np.linalg.norm(sobj - curr_ego_center, axis=1)
            idx = np.argmin(sod_dists)
            sod_min_dist = sod_dists[idx]
            sdx = sod_xy[0] - curr_ego_center[0]
            sdy = sod_xy[1] - curr_ego_center[1]
            if sdx > 5 and abs(sdy) < 1.2 and sod_min_dist < 15:
                has_front_obj = True
                break
        return has_front_obj

    def get_stopline_pos(self, ego_polygon, stop_lines):
        """
        Args:
            ego_traj: n x 4 x 2
            stop_lines: n x 2
        Returns:
            pos: (x, y)  base on ego cooridinate
        """
        ego_traj = ego_polygon.mean(axis=1)  # n x 2
        indeices, _ = get_lines_distance(stop_lines, ego_traj[0])
        # check intersection
        intersected_traj_index = None
        intersected_stop_line_index = None
        find_flag = False
        intersection_p = np.array([])
        for index in indeices:
            nearest_stop_line = stop_lines[index]
            ego_traj_segments = split_points_to_segments(ego_traj)
            for i, segment in enumerate(ego_traj_segments):
                if segment_intersect(segment[0], segment[1], nearest_stop_line[0], nearest_stop_line[1]):
                    find_flag = True
                    intersected_stop_line_index = index
                    intersected_traj_index = i
                    break
            if find_flag:
                intersection_p = line_intersection_point(
                    ego_traj_segments[intersected_traj_index][0],
                    ego_traj_segments[intersected_traj_index][1],
                    stop_lines[intersected_stop_line_index][0],
                    stop_lines[intersected_stop_line_index][1],
                )
                return intersection_p, find_flag
        # no intersection, calculate projection
        projection_p = np.array([])
        for index in indeices:
            nearest_stop_line = stop_lines[index]
            base_pt = ego_traj[-1]
            base_polygon = ego_polygon[-1]
            temp_projection_p = get_projection_on_segment(
                base_pt[0],
                base_pt[1],
                nearest_stop_line[0, 0],
                nearest_stop_line[0, 1],
                nearest_stop_line[1, 0],
                nearest_stop_line[1, 1],
            )
            if len(temp_projection_p) < 1:
                continue
            else:
                distance = ((base_pt[0] - temp_projection_p[0]) ** 2 + (base_pt[1] - temp_projection_p[1]) ** 2) ** 0.5
                theta = get_vector_angle(base_polygon[1:3, :].mean(axis=0) - base_pt, temp_projection_p - base_pt)
                if distance > 30.0 or 0.52 < abs(theta) < 2.61:
                    continue
                else:
                    projection_p = temp_projection_p
                    break
        return projection_p, False

    def get_red_cross_penalty(
        self,
        dist,
        cross_flag,
        cross_reward,
        pred_dist,
        red_reward,
        longdistance_reward,
        pred_dist_thresh,
        dist_to_stopline_thresh,
        action_time_thresh,
    ):
        red_relative_reward = 0.0
        ##压停止线
        if cross_flag:
            red_relative_reward -= cross_reward
            return red_relative_reward

        if dist < 0:
            if -dist < dist_to_stopline_thresh:  ##在停止线前，计算刹停距离
                ### v = dist/0.2
                ##反应时间2s
                stop_dist = pred_dist / 0.2 * action_time_thresh
                if stop_dist > -dist:  ##要闯红灯
                    red_relative_reward -= red_reward
            # 是否停车过远，前方必须没车？？现在没考虑 ###5-15米
            else:
                if pred_dist < pred_dist_thresh:
                    red_relative_reward -= longdistance_reward
        return red_relative_reward

    def speed_penalty(self, speed, green_reward=1.0):
        """
        计算速度惩罚值
        speed: 速度 (km/h)，范围 [0, 5]
        返回: 惩罚值，范围 [-1.0, 0]
        """
        if speed <= 0:
            return -green_reward
        elif speed >= 5:
            return 0.0
        else:
            return -green_reward + (float(speed) / 5.0) * green_reward

    def forward(self, ego_polygon, raw_env_seq, ts):
        """Forward."""
        # tld info
        traffic_light_reward = 0.0
        longdistance_reward = 0
        # green_reward = 1.0
        red_reward = 10.0
        cross_reward = 10.0
        pred_dist_thresh = 0.1  ##速度小于2km/h开始给惩罚
        action_time_thresh = 1.0
        dist_to_stopline_thresh = 3.0
        # tld info
        raw_env = raw_env_seq[ts]
        raw_env_pre = raw_env_seq[ts - 1]
        tld_pre = raw_env_pre["tld"]
        tld = raw_env["tld"]  # 4
        if np.array(tld).sum() == 16:  ##无红绿灯路口
            return traffic_light_reward
        tldcolor = ["k", "red", "yellow", "green", "gray"]
        tld_color_select, direct_light = self.get_tld_direct_select(raw_env, tld, tldcolor)
        tld_color_select_pre, direct_light_pre = self.get_tld_direct_select(raw_env_pre, tld_pre, tldcolor)
        stright_color = tldcolor[tld[2]]
        stright_color_pre = tldcolor[tld_pre[2]]
        c = tld_color_select
        c_pre = tld_color_select_pre
        scene_id, egopath_stoplines_points = self.get_scene_class(raw_env)
        ##计算是否在停止线附近
        ####这里如果有egopath_stoplines_ids,可以用这两条线去算，没有的再采用这种全局搜索计算距离
        ###直行灯绿灯时，这个距离不满足条件，为啥是视觉上看很近，数值上差很多45m
        ## 针对双路口情况的优化
        if len(egopath_stoplines_points) > 0:
            stopline_pos, cross_flag = self.get_stopline_pos(ego_polygon, egopath_stoplines_points)
        else:
            stopline_pos, cross_flag = self.get_stopline_pos(ego_polygon, raw_env["stop_line"])
        has_front_obj = self.judge_ego_has_front_obj(raw_env_seq, ego_polygon, ts)
        # if (len(stopline_pos) > 0 or cross_flag) and (not has_front_obj) and scene_id == 1:
        if len(stopline_pos) > 0 or cross_flag:
            dist = calc_path_length_from_point(ego_polygon, stopline_pos)  # 从 stopline 开始的路径长度
            pred_dist = calc_path_length(ego_polygon.mean(axis=1))  # 预测经过的路径的长度，这里可以进一步细化
            # positive dist means pred traj has crossed stopline
            half_length = 10.0
            ##如果自车距离停止满足一定距离，考虑红绿灯的reward,这里是27米，不满足距离停止线较近约束
            if dist > -half_length or cross_flag:  # and pred_dist - dist > half_length:
                # half vehcile length
                ##如果不需要进入待转区或者已经在待转区内
                if scene_id == 0 or scene_id == 2:
                    pass
                ##如果在待转区前，需要考虑目标方向
                else:
                    ##左右转
                    if direct_light == 1:
                        ##有没有左右转的信号灯，看直行灯
                        stright_red = stright_color == "red"
                        left_red = c == "red"
                        left_green = (c == "green") and (c_pre == "green")
                        stright_green = (stright_color == "green") and (stright_color_pre == "green")
                        speed = pred_dist / 0.2 * 3.6
                        if c == "gray":
                            if stright_green:
                                if pred_dist <= pred_dist_thresh and dist < 0 and speed <= 10:  ##速度>5km/h
                                    speed = min(5, max(0, speed))
                                    traffic_light_reward += self.speed_penalty(speed)
                        ##有左右转的信号灯，看直行灯和左右转的灯
                        else:
                            if (
                                (stright_green or left_green) and pred_dist <= pred_dist_thresh and dist < 0 and speed <= 10
                            ):  ##速度>5km/h:
                                ####需要进入待转区
                                speed = min(5, max(0, speed))
                                traffic_light_reward += self.speed_penalty(speed)
        return traffic_light_reward


class VirtualWallReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.penalty_score = -4.0
        self.bias = 0.1
        self.backtrace_dist = 36.0  # 回溯距离阈值，单位米
        self.overleap_thresh = 0.9  # 膨胀后的覆盖率阈值

    def trigger_list_by_scene(self, navi_info):
        """
        输出场景触发的距离阈值
        """
        trigger_list = []
        # 道路分岔场景
        trigger_list.extend(
            distance_ranges_to_target_scene(
                navi_info,
                main_actions=ROAD_SPLIT_MAIN_ACTION_MAPPING,
                assist_actions=ROAD_SPLIT_ASSIST_ACTION_MAPPING,
                action_distance_range=(-15, 50),
                penalty_dist_thresh=30,
                action_max_distance=300,
            )
        )

        return trigger_list

    def expanded_gt_vs_ego_area(
        self,
        gt_corners: np.ndarray,
        ego_corners: np.ndarray,
        eps = 1e-3
    ):
        # 处理道路级偏航定义的膨胀系数
        lon_range = 6.0  # 允许比较大的纵向移动
        lat_range = 0.6  # 允许较小的横向移动

        gt_rect = np.asarray(gt_corners, dtype=float)
        ego_rect = np.asarray(ego_corners, dtype=float)

        gt_poly = Polygon(gt_rect)
        ego_poly = Polygon(ego_rect)
        if not gt_poly.is_valid or not ego_poly.is_valid:
            gt_poly = gt_poly.buffer(0)
            ego_poly = ego_poly.buffer(0)

        # 计算 GT 的局部纵/横向（前向、左向）
        u_lon = gt_rect[1] - gt_rect[0]  # 右后 -> 右前（前向）
        u_lat = gt_rect[3] - gt_rect[0]  # 右后 -> 左后（左向）
        lon_len = float(np.linalg.norm(u_lon))
        lat_len = float(np.linalg.norm(u_lat))

        # 将前向旋到全局 x 轴对齐
        theta_deg = math.degrees(math.atan2(u_lon[1], u_lon[0]))
        center = tuple(gt_rect.mean(axis=0))
        gt_aligned = rotate(gt_poly, -theta_deg, origin=center, use_radians=False)

        # 计算按“定量余量”膨胀所需缩放比例（关于中心缩放）
        sx = (lon_len + 2.0 * lon_range) / lon_len
        sy = (lat_len + 2.0 * lat_range) / lat_len
        gt_expanded_aligned = scale(gt_aligned, xfact=sx, yfact=sy, origin=center)
        gt_expanded = rotate(gt_expanded_aligned, theta_deg, origin=center, use_radians=False)

        inter = gt_expanded.intersection(ego_poly)
        inter_area = float(inter.area) if not inter.is_empty else 0.0
        ego_area = float(ego_poly.area) if ego_poly.area > eps else 0.0

        cover_ratio = inter_area / ego_area if ego_area > eps else 0.0
        cover_ratio = max(0.0, min(1.0, cover_ratio))
        return cover_ratio, gt_expanded

    def find_backtrace_start_idx_by_distance(self, pred_polygon, cross_idx):
        if cross_idx <= 0:
            return 1

        centers = pred_polygon.mean(axis=1)  # (T, 2)

        cur_dist = 0.0
        k = cross_idx - 1

        while k > 0 and cur_dist < self.backtrace_dist:
            step_vec = centers[k + 1] - centers[k]
            step_dist = float(np.linalg.norm(step_vec))
            cur_dist += step_dist
            k -= 1

        start_k = max(0, k + 1)
        return start_k


    def forward(self, pred_polygon, ego_gt_polygon, raw_env):
        """根据轨迹是否穿过惩罚墙设计的惩罚函数"""
        ego_path_info = get_xyyaw_from_polygon(pred_polygon)
        pred_num = len(ego_path_info)
        virtual_wall_reward = [0.0] * pred_num

        if len(raw_env[0]["navi_infos"]["virtual_wall"]) < 1:
            return virtual_wall_reward

        for i in range(1, pred_num):
            ego_point1, ego_point2 = ego_path_info[i - 1, :2], ego_path_info[i, :2]
            virtual_wall = raw_env[i]["navi_infos"]["virtual_wall"]
            for wall_coords in virtual_wall:
                wall_point1, wall_point2 = wall_coords[0], wall_coords[1]
                if is_segments_intersection(ego_point1, ego_point2, wall_point1, wall_point2, self.bias):
                    # 增加穿墙后的惩罚
                    for j in range(i, pred_num):
                        virtual_wall_reward[j] = self.penalty_score

                    # 当前只从i开始处理， 回溯穿墙前的惩罚, 按照距离回溯（20m）
                    end_k = min(i, min(pred_polygon.shape[0], ego_gt_polygon.shape[0]))
                    start_k = self.find_backtrace_start_idx_by_distance(pred_polygon, i)
                    real_start_k = start_k
                    for k in range(start_k, end_k):
                        inter_ratio, _ = self.expanded_gt_vs_ego_area(ego_gt_polygon[k], pred_polygon[k])
                        if inter_ratio < self.overleap_thresh:
                            real_start_k = k
                            break

                    start_k = real_start_k
                    num_steps = max(1, end_k - start_k)
                    for k in range(start_k, end_k):
                        inter_ratio, _ = self.expanded_gt_vs_ego_area(ego_gt_polygon[k], pred_polygon[k])
                        # 考虑距离衰减
                        stage_ratio = (k - start_k + 1) / num_steps
                        # 计算几何比例
                        geo_ratio = abs(1 - inter_ratio)
                        # 综合比例，随时间增加，考虑几何偏差
                        w = 0.7 * stage_ratio + 0.3 * geo_ratio
                        w = max(0.0, min(1.0, w))
                        virtual_wall_reward[k] = self.penalty_score * w

                    return virtual_wall_reward

        return virtual_wall_reward
@dataclass
class RibbonModel:
    s: np.ndarray         # (N,)
    C: np.ndarray         # (N,2)
    T: np.ndarray         # (N,2)
    N: np.ndarray         # (N,2), 指向“右侧”（保证 n_right >= 0）
    n_right: np.ndarray   # (N,)
    n_left: np.ndarray    # (N,)

logging.getLogger("shapely.geos").setLevel(logging.ERROR)

class HumanoidReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps
        self.passageway = None
        self.expand_width = 0.2

    @staticmethod
    def unit(v, eps=1e-9):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        n = np.maximum(n, eps)
        return v / n

    @staticmethod
    def rot90(v):
        # 逆时针旋转 90°
        return np.stack([-v[..., 1], v[..., 0]], axis=-1)

    def half_plane_polygon(self, point, tangent, keep_front=True):
        """
        生成一个半平面多边形，用于与目标几何相交实现裁剪。
        - point: [2,]，裁剪线经过的点
        - tangent: [2,]，裁剪线的法向向量
        """
        t = self.unit(tangent)
        # 与 t 正交的方向
        n = np.array([-t[1], t[0]], dtype=float)

        p_mid = np.asarray(point, dtype=float)
        p1 = p_mid - n * 100
        p2 = p_mid + n * 100

        # 选择半平面：沿 n 的一侧构造；方向根据 keep_front 取正负
        sign = 1.0 if keep_front else -1.0
        q1 = p1 + sign * t * 100
        q2 = p2 + sign * t * 100

        # 这个四边形覆盖对应半平面
        return Polygon([p1, p2, q2, q1])

    def clip_quad_by_endcaps(self, quad, C):
        """
        - quad: (M,2) 四边形/多边形坐标
        - C: (N,2) 中心线离散点
        返回裁剪后的 shapely Polygon
        """
        quad = np.asarray(quad, dtype=float)
        poly_q = Polygon(quad)
        if poly_q.is_empty:
            return poly_q

        v0, C0 = C[1] - C[0], C[0]
        vn, Cn = C[-1] - C[-2], C[-1]

        # 判断是否需要裁起点侧：如果任一点在“后侧”（dot<0），就裁
        dots_start = (np.asarray(poly_q.exterior.coords)[:, :2] - C0) @ self.unit(v0)
        need_clip_start = np.any(dots_start < 0)

        # 判断是否需要裁终点侧：如果任一点在“前侧”（dot>0），就裁
        dots_end = (np.asarray(poly_q.exterior.coords)[:, :2] - Cn) @ self.unit(vn)
        need_clip_end = np.any(dots_end > 0)

        poly = poly_q
        if need_clip_start:
            hp_start = self.half_plane_polygon(C0, v0, keep_front=True)   # 保留起点线的前侧
            poly = poly.intersection(hp_start)
            if poly.is_empty:
                return poly

        if need_clip_end:
            hp_end = self.half_plane_polygon(Cn, vn, keep_front=False)    # 保留终点线的后侧
            poly = poly.intersection(hp_end)

        return poly

    def build_passageway(self, quads):
        """
        quads: (N,4,2)  顶点顺序: [右后(0), 右前(1), 左前(2), 左后(3)]
        返回 RibbonModel，其中 N 的正方向被强制指向右侧（0/1 这条边所在侧）
        """
        assert quads.ndim == 3 and quads.shape[1:] == (4, 2)

        # 过滤低速、倒车轨迹段
        quads_mask = (np.sqrt(np.diff(quads.mean(axis=-2)[:, 0]) ** 2 +
                           np.diff(quads.mean(axis=-2)[:, 1]) ** 2) * self.fps) > 0.1
        quads_mask = np.insert(quads_mask, 0, [True])
        quads = quads[quads_mask]
        q0, q1, q2, q3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]

        # 1) 基于长边构造中心 C 与切向 T
        M_right = 0.5 * (q0 + q1)      # 右侧长边中点（0->1）
        M_left  = 0.5 * (q3 + q2)      # 左侧长边中点（3->2）
        C = 0.5 * (M_right + M_left)   # 两条长边中点的中点，避免形心偏斜
        C[0]  = 0.5 * (q0[0] + q3[0])  # 后边（0,3）的中点
        C[-1] = 0.5 * (q1[-1] + q2[-1])# 前边（1,2）的中点

        vR = q1 - q0
        vL = q2 - q3
        T = self.unit(vR + vL)
        bad = (np.linalg.norm(vR + vL, axis=1) < 1e-8)
        if np.any(bad):
            mid_front = 0.5 * (q1 + q2)
            mid_rear  = 0.5 * (q0 + q3)
            T_fb = self.unit(mid_front - mid_rear)
            T[bad] = T_fb[bad]

        Nvec = self.rot90(T)

        # 2) 半宽：用端点投影“紧密包含”整条长边
        def endpoints_proj(edge2x2):
            a = np.sum((edge2x2[:, 0] - C) * Nvec, axis=1)
            b = np.sum((edge2x2[:, 1] - C) * Nvec, axis=1)
            return a, b

        ar, br = endpoints_proj(quads[:, [0, 1]])  # 右侧 0->1
        al, bl = endpoints_proj(quads[:, [3, 2]])  # 左侧 3->2

        # 初值（中点投影）+ 端点极值收紧
        n_right0 = np.sum((M_right - C) * Nvec, axis=1)
        n_left0  = np.sum((M_left  - C) * Nvec, axis=1)
        n_right = np.maximum.reduce([n_right0, ar, br])
        n_left  = np.minimum.reduce([n_left0,  al, bl])

        # 3) 全局朝向校正（右半宽应为非负）
        if np.median(n_right) < 0:
            Nvec = -Nvec
            T    = -T
            n_right, n_left = -n_right, -n_left

        # 4) 弧长参数 s（沿中心线 C）
        ds = np.linalg.norm(np.diff(C, axis=0), axis=1, keepdims=True)
        s = np.concatenate([np.zeros((1, 1)), np.cumsum(ds, axis=0)], axis=0).ravel()

        return RibbonModel(
            s=s, C=C, T=T, N=Nvec,
            n_right=n_right, n_left=n_left
        )

    def envelopes_to_world(self, model, expand=0.0):
        """
        返回两条外扩后的多段线：
        right: C + (n_right + expand)*N
        left : C + (n_left  - expand)*N
        """
        right = model.C + (model.n_right + expand)[:,None] * model.N
        left  = model.C + (model.n_left  - expand)[:,None] * model.N
        return right, left

    @staticmethod
    def sample_curve(curve, s_axis, s_query):
        if s_query <= s_axis[0]: return curve[0]
        if s_query >= s_axis[-1]: return curve[-1]
        j = np.searchsorted(s_axis, s_query)
        t = (s_query - s_axis[j-1]) / (s_axis[j] - s_axis[j-1] + 1e-12)
        return (1-t)*curve[j-1] + t*curve[j]

    def build_band_polygon_between_s(self, model, s_min, s_max, expand=0.0):
        right, left = self.envelopes_to_world(model, expand=expand)
        s = model.s
        i0 = np.searchsorted(s, s_min, side='left')
        i1 = np.searchsorted(s, s_max, side='right')
        r_seg = right[max(i0-1,0):min(i1+1,len(s))]
        l_seg = left [max(i0-1,0):min(i1+1,len(s))]
        r_start = self.sample_curve(right, s, s_min)
        r_end   = self.sample_curve(right, s, s_max)
        l_start = self.sample_curve(left , s, s_min)
        l_end   = self.sample_curve(left , s, s_max)
        poly = np.vstack([r_start, r_seg, r_end, l_end, l_seg[::-1], l_start])
        # 去除重复点
        _, idx = np.unique(np.round(poly, 6), axis=0, return_index=True)
        return poly[np.sort(idx)]

    def can_make_polygon(self, coords) -> bool:
        """最小判断：能否构成 polygon shell（至少 3 个不重复点）"""
        if coords is None:
            return False
        try:
            arr = np.asarray(coords, dtype=np.float64)
        except Exception:
            return False
        if arr.ndim != 2 or arr.shape[1] != 2:
            return False
        # 去掉 NaN/Inf
        m = np.isfinite(arr).all(axis=1)
        arr = arr[m]
        if arr.shape[0] < 3:
            return False
        # 统计不重复点
        uniq = np.unique(arr, axis=0)
        return uniq.shape[0] >= 3

    def quad_outside_area(self, quad):
        """
        返回： quad 在通行走廊范围外的面积
        """
        # --- 0) 基础几何 ---
        poly_q = Polygon(quad)
        if poly_q.is_empty or not poly_q.is_valid:
            poly_q = poly_q.buffer(0.0)
            if poly_q.is_empty or not poly_q.is_valid:
                return 0.0

        C, s = self.passageway.C, self.passageway.s  # C: (N,2), s: (N,)

        # --- 1) 构建通行走廊 ---
        cache_key = ("__band_full__", float(self.expand_width))
        band_full = getattr(self.passageway, "_band_full_cache", {}).get(cache_key, None)
        if band_full is None:
            band_full_coords = self.build_band_polygon_between_s(
                self.passageway, s[0], s[-1], expand=self.expand_width
            )

            if not self.can_make_polygon(band_full_coords):
                return 0.0
            try:
                band_full = Polygon(band_full_coords)
            except (ValueError, ShapelyError):
                return 0.0

            if not band_full.is_valid:
                band_full = band_full.buffer(0.0)
                if band_full.is_empty or not band_full.is_valid:
                    return 0.0

            if not hasattr(self.passageway, "_band_full_cache"):
                self.passageway._band_full_cache = {}
            self.passageway._band_full_cache[cache_key] = band_full

        # --- 2) 检查范围限定到通行走廊有效范围内 ---
        quad_clipped = self.clip_quad_by_endcaps(quad=quad, C=self.passageway.C)
        if quad_clipped.is_empty:
            return 0.0

        # --- 3) 用近邻 s 估计局部范围（两侧各扩 1）---
        dist = np.linalg.norm(C[None, :, :] - np.asarray(quad)[:, None, :], axis=2)  # (4,N)
        idx = np.argmin(dist, axis=1).astype(int)
        i_min = max(0, idx.min() - 1)
        i_max = min(len(s) - 1, idx.max() + 1)
        s_min, s_max = s[i_min], s[i_max]

        # --- 4) 计算惩罚面积 ---
        band_local_coords = self.build_band_polygon_between_s(
            self.passageway, s_min, s_max, expand=self.expand_width
        )

        if not self.can_make_polygon(band_local_coords):
            return 0.0
        try:
            band_local = Polygon(band_local_coords)
        except (ValueError, ShapelyError):
            return 0.0

        if not band_local.is_valid:
            band_local = band_local.buffer(0.0)
            if band_local.is_empty or not band_local.is_valid:
                return 0.0

        inter = quad_clipped.intersection(band_local)
        inside_area = inter.area if not inter.is_empty else 0.0
        outside_area = quad_clipped.area - inside_area
        return max(outside_area, 0.0)

    def first_index_after_distance(self, points, base_idx, min_dist=3.5):
        """
        从 base_idx 开始向后累计距离，返回首个累计距离>=min_dist的下标,
        若达不到返回 len(pts) - 1。
        """
        pts = np.asarray(np.mean(points, axis=-2), dtype=float)
        n = len(pts)
        if base_idx >= n - 1:
            return n - 1

        seg = np.diff(pts[:, :2], axis=0)        # (n-1, 2)
        seg_len = np.hypot(seg[:, 0], seg[:, 1]) # (n-1,)
        # 从 base_idx 开始的累积距离
        cum = np.cumsum(seg_len[base_idx:])

        k = np.searchsorted(cum, min_dist, side='left')
        if k >= cum.size:
            return n - 1
        return min(base_idx + k + 1, n - 1)

    def expanded_gt_vs_ego_area(
        self,
        gt_corners: np.ndarray,
        ego_corners: np.ndarray,
        eps: float = 1e-3
    ):
        """
        将 GT 四边形按其自身朝向做横向/纵向膨胀后，与 EGO 四边形之间的覆盖率:
        - cover_ratio: Expanded(GT) 对 EGO 的 coverage = area(intersection) / area(ego)
        """
        gt_rect = np.asarray(gt_corners, dtype=float)
        ego_rect = np.asarray(ego_corners, dtype=float)

        ego_poly = Polygon(ego_rect)
        if not ego_poly.is_valid:
            ego_poly = ego_poly.buffer(0)

        # 计算 GT 的局部纵/横向向量
        u_lon = gt_rect[1] - gt_rect[0]  # 右后 -> 右前（前向）
        u_lat = gt_rect[3] - gt_rect[0]  # 右后 -> 左后（左向）
        lon_len = float(np.linalg.norm(u_lon))
        lat_len = float(np.linalg.norm(u_lat))

        if lon_len < eps or lat_len < eps:
            return 0.0, Polygon(gt_rect)

        u_lon_unit = u_lon / lon_len
        u_lat_unit = u_lat / lat_len

        lon_front = 0.5
        lon_back  = 2.
        lat_left  = 0.3
        lat_right = 0.3

        RR_exp = gt_rect[0] - u_lon_unit * lon_back  - u_lat_unit * lat_right  # 右后
        FR_exp = gt_rect[1] + u_lon_unit * lon_front - u_lat_unit * lat_right  # 右前
        FL_exp = gt_rect[2] + u_lon_unit * lon_front + u_lat_unit * lat_left   # 左前
        RL_exp = gt_rect[3] - u_lon_unit * lon_back  + u_lat_unit * lat_left   # 左后

        gt_expanded_rect = np.stack([RR_exp, FR_exp, FL_exp, RL_exp], axis=0)
        gt_expanded = Polygon(gt_expanded_rect)
        if not gt_expanded.is_valid:
            gt_expanded = gt_expanded.buffer(0)

        inter = gt_expanded.intersection(ego_poly)
        inter_area = float(inter.area) if not inter.is_empty else 0.0
        ego_area = float(ego_poly.area)

        if ego_area <= eps:
            cover_ratio = 0.0
        else:
            cover_ratio = inter_area / ego_area
            if cover_ratio < 0.0:
                cover_ratio = 0.0
            elif cover_ratio > 1.0:
                cover_ratio = 1.0

        return cover_ratio, gt_expanded

    def forward(self, ego_polygon, gt_polygon, raw_env, his_num):
        """Forward."""
        # 构建gt通行走廊
        # 找出过路口的起始索引
        direct_light_list = np.empty(len(gt_polygon) - 2, dtype=object)
        for t in range(0, len(gt_polygon) - 3, 3):
            gt_val = judge_intersection_and_maneuver(gt_polygon[t:t+4], raw_env, t)
            pred_val = judge_intersection_and_maneuver(ego_polygon[t:t+4], raw_env, t)
            val = gt_val if gt_val else pred_val
            direct_light_list[t] = val
            if val is not np.nan:
                break

        mask = np.isin(direct_light_list, (1, 2, 3))
        indices = np.where(mask)[0]
        need_left = False
        need_right = False
        need_straight = False
        if indices.size != 0:
            first_idx = int(indices[0])

            # 初始时刻自车在gt附近
            inter_ratio, _ = self.expanded_gt_vs_ego_area(gt_polygon[first_idx], ego_polygon[first_idx])
            if inter_ratio < 0.2:
                return None, None

            # 检查gt轨迹与导航信息一致性
            check_end = self.first_index_after_distance(gt_polygon, first_idx, min_dist=50)
            path_points = gt_polygon[first_idx:check_end:3]
            path_turn_type = determine_path_turn_type(path_points.mean(axis=-2),
                                                        threshold_degree=60, fuzzy_threshold_degree=40)
            need_left = np.any(direct_light_list[first_idx] == 1)
            need_straight = np.any(direct_light_list[first_idx] == 2)
            need_right = np.any(direct_light_list[first_idx] == 3)
            if (need_left and (Turntype.LEFT not in path_turn_type)) or \
                (need_straight and (Turntype.STRAIGHT not in path_turn_type)) or \
                (need_right and (Turntype.RIGHT not in path_turn_type)):
                return None, None

        start_t = 0
        end_t = 0

        if need_left:
            start_t = max(first_idx, his_num)
            check_end = self.first_index_after_distance(gt_polygon, first_idx, min_dist=30)
            end_t = min(start_t + 40, check_end)
            # 检查gt合理性
            diff_poly = gt_polygon[1:] - gt_polygon[:-1]
            mean_step = np.mean(diff_poly, axis=-2)
            gt_vel = np.linalg.norm(mean_step, axis=-1) * self.fps
            gt_vel_intersect = gt_vel[start_t:end_t]
            # gt路口左转速度过高
            if np.any(gt_vel >= 60 / 3.6) or np.any(gt_vel_intersect >= 40 / 3.6):
                return None, None
            gt_check_end = self.first_index_after_distance(gt_polygon, first_idx, min_dist=5)
            path_points = gt_polygon[first_idx:gt_check_end]
            check_turn_type = determine_path_turn_type(path_points.mean(axis=-2),
                                                        threshold_degree=60, fuzzy_threshold_degree=40)
            # gt路口转小弯
            if Turntype.LEFT in check_turn_type:
                return None, None

        if need_right:
            start_t = max(first_idx, his_num)
            check_end = self.first_index_after_distance(gt_polygon, first_idx, min_dist=20)
            end_t = min(start_t + 40, check_end)
            # 检查gt合理性
            diff_poly = gt_polygon[1:] - gt_polygon[:-1]
            mean_step = np.mean(diff_poly, axis=-2)
            gt_vel = np.linalg.norm(mean_step, axis=-1) * self.fps
            gt_vel_intersect = gt_vel[start_t:end_t]
            # gt路口右转速度过高
            if np.any(gt_vel >= 60 / 3.6) or np.any(gt_vel_intersect >= 40 / 3.6):
                return None, None

        if start_t < end_t:
            self.passageway = self.build_passageway(gt_polygon[start_t : end_t])
        else:
            return None, None

        # 计算惩罚
        penalty_list = np.zeros(ego_polygon.shape[0], dtype=np.float64)
        T = len(ego_polygon)
        decay_t1 = int((end_t - start_t) * 0.2 + start_t)
        # decay_t2 = (T - end_t) * 0.2 + end_t
        for t in range(his_num, T):
            quad_area = Polygon(ego_polygon[t]).area
            outside_area = self.quad_outside_area(ego_polygon[t])
            decay_ratio = 1.
            if t <= decay_t1:
                decay_ratio = 1 - (decay_t1 - t) / decay_t1
            # if t >= decay_t2:
            #     decay_ratio = 1 - (t - decay_t2) / (T - decay_t2)
            penalty_list[t] = (outside_area / quad_area) * -4.0 * decay_ratio

        expand_passageway = None
        if self.passageway:
            expand_passageway = self.envelopes_to_world(self.passageway, self.expand_width)

        return penalty_list[his_num:], expand_passageway

logging.getLogger("shapely.geos").setLevel(logging.ERROR)

class HumanoidNudgeReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps
        self.buffer_start = -2
        self.buffer_lat_threshold = 2.0
        self.buffer_end = -40
        self.penaty = 1.5
    def ShouldToGo(self,reward_results,t,seq_len):
        #safe: ttc_reward,traffic_light_reward,min_distance_reward,speed_limit_reward
        is_safe = reward_results["ttc_reward"][t] > -0.1 and reward_results["traffic_light_reward"][t] > -0.1 and reward_results["min_distance_reward"][t] > -0.1 and reward_results["speed_limit_reward"][t] >-0.1
        #logic cross_solid_line_reward dangerous_lc_penalty  navi_lane_reward  navi_reward wrong_way_reward
        is_logic = reward_results["cross_solid_line_reward"][t] > -0.1 and reward_results["danger_lc_reward"][t] > -0.1 and reward_results["navi_lane_reward"][t] > -0.1 and reward_results["navi_reward"][t] >-0.1 and reward_results["wrong_way_reward"][t] >-0.1
        #progress_reward
        is_no_progress_reward = reward_results["progress_reward"][t] <0.02
        #is not x acc
        is_not_ax=reward_results["ax"][t] <0.2
        # no collision
        is_no_collision = ~np.any(reward_results["collision_reward"] <= -10)
        future_t = t+15 if t+15<seq_len else seq_len
        if_no_has_future_ttc = True
        if np.any(reward_results["ttc_reward"][t:future_t] <-0.1):
            if_no_has_future_ttc = False


        return (is_safe and is_logic and is_no_progress_reward and is_not_ax and is_no_collision and if_no_has_future_ttc)

    def forward(self, ego_polygon, gt_polygon, raw_env, his_num,reward_results):
        """
            Forward.
            拟人惩罚：
                1、进度与人驾驶进度比较，若预测结果慢于人驾结果，则惩罚1.5。
        """
        #计算roll-out进度
        ego_pos =  np.mean(ego_polygon,axis=-2)
        ego_pos_diff = np.diff(ego_pos,axis=0)
        ego_pos_diff_abs = np.linalg.norm(ego_pos_diff,axis=1)
        ego_pos_s = np.insert(np.cumsum(ego_pos_diff_abs,axis=0),0,0)
        #计算gt进度
        gt_pos =  np.mean(gt_polygon,axis=-2)
        gt_pos_diff = np.diff(gt_pos,axis=0)
        gt_pos_diff_abs = np.linalg.norm(gt_pos_diff,axis=1)
        gt_pos_s = np.insert(np.cumsum(gt_pos_diff_abs,axis=0),0,0)
        #判断轨迹左右转情况
        path_turn_type = determine_path_turn_type(gt_polygon[::5].mean(axis=-2),
                                                    threshold_degree=60, fuzzy_threshold_degree=40)


        reward_list = np.zeros(ego_polygon.shape[0] + 1, dtype=np.float64)
        progress_diff =  np.zeros(ego_polygon.shape[0] + 1, dtype=np.float64)
        if Turntype.STRAIGHT not in path_turn_type:
            return reward_list[his_num:], progress_diff[his_num:]

        seq_len = len(gt_polygon) - his_num+1-1
        for t in range(his_num, len(gt_polygon)):
            if self.ShouldToGo(reward_results,t-his_num+1,seq_len):
                penalize = False
                penalize_lat = False

                if ego_pos_s[t] < gt_pos_s[t] + self.buffer_start:
                    penalize = True

                    prev_idx = max(his_num, t - 1)
                    next_idx = min(len(ego_pos) - 1, t + 1)
                    tang = ego_pos[next_idx] - ego_pos[prev_idx]
                    tang_norm = np.linalg.norm(tang)
                    if tang_norm > 1e-6:
                        tang_u = tang / tang_norm
                        normal_u = np.array([-tang_u[1], tang_u[0]])
                        rel_gt = gt_pos[t] - ego_pos[t]
                        gt_lat = float(np.dot(rel_gt, normal_u))
                        if abs(gt_lat) > self.buffer_lat_threshold:
                            penalize_lat = True
                            # print("t:", t, "gt_lat:", gt_lat)

                if penalize:
                    reward_list[t] = -1.2
                    if penalize_lat:
                        reward_list[t] -= 1.2
                    progress_diff[t] = gt_pos_s[t] - ego_pos_s[t]
        return reward_list[his_num:], progress_diff[his_num:]


class ContinuousLaneChangeReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()
        self.penalty_duration = 8  # 连续换道惩罚持续帧数
        self.penalty_mag = 8.0  # 连续换道惩罚幅度
        self.angle_threshold = np.radians(30)  # 车道线夹角阈值

    def trigger_list_by_scene(self, navi_info):
        """
        输出场景触发的距离
        """
        trigger_list = []
        # 左转/右转/左转调头 + 无辅动作 + ≥100 m
        trigger_list.extend(
            distance_ranges_to_current_link_only(
                navi_info,
                main_actions=LEFT_RIGHT_TURN_MAIN_ACTION_MAPPING,
                assist_actions=NONE_ASSIST_ACTION_MAPPING,
                trigger_distance=100.0,
            )
        )

        return trigger_list

    def _calc_continuous_lc_reward(self, pre_ego_polygon, raw_env):
        """
        计算整段轨迹中，跨过车道线的ts_idx，用于计算横跨一个车道的停留时间，来惩罚连续换道
        """
        ego_path_info = get_xyyaw_from_polygon(pre_ego_polygon)
        is_cross_lanelines = []

        # 1. 检测所有压线事件
        for i, polygon in enumerate(pre_ego_polygon):
            ego_polygon = Polygon(polygon)
            ego_yaw = ego_path_info[i, 6]
            ego_point = polygon.mean(axis=0).reshape(1, 2)
            is_cross_lanelines.append([0, None, 0.0])
            lane_lines = raw_env[i]["lane_lines"]

            for j, lane_line in enumerate(lane_lines):
                polyline = LineString(lane_line[0][:, :2])
                if not polyline.intersects(ego_polygon):
                    continue

                # 方向夹角过滤
                pts = lane_line[0][:, :2]
                line_vec = pts[-1] - pts[0]
                line_yaw = np.arctan2(line_vec[1], line_vec[0])
                delta_yaw = abs(normalize_angle(ego_yaw - line_yaw))
                if delta_yaw > self.angle_threshold:
                    continue  # 非平行，跳过

                # 有一个车道线与ego交互
                x = ego_point[0][0]
                y = ego_point[0][1]
                distance_signed = signed_distance(Point(x, y), polyline)
                is_cross_lanelines[i][2] = distance_signed
                is_cross_lanelines[i][1] = polyline

                if abs(distance_signed) < 0.1:
                    is_cross_lanelines[i][0] = 1

                if i > 0 and abs(is_cross_lanelines[i-1][2]) > 1e-6:
                    # 防止自车偏一侧压线也被误判为cross
                    if (distance_signed * is_cross_lanelines[i-1][2]) < 0:
                        is_cross_lanelines[i][0] = 1

        continuous_lc_reward = [0] * len(pre_ego_polygon)

        # 2. 收集所有压线帧索引
        cross_idx = []
        for i, [is_cross, _, _] in enumerate(is_cross_lanelines):
            if is_cross == 1:
                cross_idx.append(i)

        if len(cross_idx) < 2:
            return continuous_lc_reward

        # 3. 检测连续换道事件，并在第二次压线时检查ODD
        for i in range(1, len(cross_idx)):
            # 检查两次压线是否跨越不同车道线（距离>2.5m）
            if abs(is_cross_lanelines[cross_idx[i]][1].distance(
                   is_cross_lanelines[cross_idx[i-1]][1])) < 2.5:
                continue

            # 检查时间间隔是否过短（<25帧）
            if cross_idx[i] - cross_idx[i-1] >= 25:
                continue

            # 在第一次和第二次压线时检查ODD
            first_cross_frame = cross_idx[i-1]
            second_cross_frame = cross_idx[i]
            is_first_in_odd = self.trigger_list_by_scene(raw_env[first_cross_frame]["navi_infos"])
            is_second_in_odd = self.trigger_list_by_scene(raw_env[second_cross_frame]["navi_infos"])

            if not (is_first_in_odd and is_second_in_odd):
                # 不在ODD范围内，跳过这次连续换道惩罚
                continue

            # 在ODD范围内，给予惩罚
            start = cross_idx[i-1]
            end = min(cross_idx[i], len(continuous_lc_reward))
            for j in range(start, end):
                continuous_lc_reward[j] -= self.penalty_mag

        return continuous_lc_reward

    def forward(self, pre_ego_polygon, raw_env):
        """Forward."""
        ego_path_info = get_xyyaw_from_polygon(pre_ego_polygon)
        pred_num = len(ego_path_info)
        continuous_lc_reward = [0.0] * pred_num

        continuous_lc_reward = self._calc_continuous_lc_reward(pre_ego_polygon, raw_env)

        return continuous_lc_reward

class JunctionLaneSelectReward(BaseLoss):
    def __init__(self) -> None:
        """Init."""
        super().__init__()

    def point_to_infinite_line_distance(self, p, a, b):
        """
        计算点 p 到由 a-b 两点确定的无限直线的距离
        """
        p = np.array(p, dtype=float)
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        ab = b - a
        ap = p - a

        # 如果 a == b，则处理为点的情况
        if np.allclose(ab, 0):
            return np.linalg.norm(ap)

        # 二维向量叉乘（结果是标量）
        cross_val = abs(ab[0] * ap[1] - ab[1] * ap[0])
        return cross_val / np.linalg.norm(ab)

    def reward_zone(self, d):
        center = 0.5  # 决定从正到负的转折位置
        slope = 0.5  # 决定曲线“平缓”
        high = 1.0  # d 很小时的奖励
        low = 0.0  # d 很大时的惩罚
        one_times = 1.0 / ((high + low) / 2 + (high - low) / 2 * np.tanh(-slope * (-center)))  # 为了让d=0时奖励为1

        return one_times * ((high + low) / 2 + (high - low) / 2 * np.tanh(-slope * (d - center)))

    def forward(self, cur_env, pred_traj, next_lanenr_idx, passway, is_human_reward_valid, gt_traj):
        junction_reward = np.zeros((pred_traj.shape[0]))
        if not cur_env["centerlines"] or cur_env["navi_path"] is None:
            return junction_reward
        DIRECTION_THRE = 10.0 / 180.0 * 3.14
        DISTANCE_THRE = 10.0
        ILLEGAL_CENTERLINE_TYPE = [
            3,  # 非机动车道
            5,  # 摩托车道
            6,  # 人行道
        ]
        END_DISTANCE_THRE = 10.0

        centerline_points_list = []
        valid_indices = []
        for i, centerline in enumerate(cur_env["centerlines"]):
            if centerline and len(centerline) > 0 and len(centerline[0]) >= 2:
                try:
                    centerline_points_list.append(LineString(centerline[0]))
                    valid_indices.append(i)
                except Exception:
                    continue

        if not centerline_points_list:
            return junction_reward

        centerline_types_list = [cur_env["centerlines"][i][1] for i in valid_indices]
        navi_path_subset = cur_env["navi_path"][cur_env["navi_path"].shape[0] // 2:]
        if len(navi_path_subset) >= 2:
            navpath = LineString(navi_path_subset)
        else:
            return junction_reward
        centerline_direction_mean = []
        for i in valid_indices:
            centerline = cur_env["centerlines"][i]
            centerline_diff = centerline[0][1:] - centerline[0][:-1]
            centerline_diff = centerline_diff / np.linalg.norm(centerline_diff, axis=1)[:, None]
            centerline_vec_sum = np.sum(centerline_diff, axis=0)
            centerline_angle = np.arctan2(centerline_vec_sum[1], centerline_vec_sum[0])
            centerline_direction_mean.append(centerline_angle)

        # 1. 找到离navpath最近的一条中心线
        min_dist = None
        min_idx = None

        for idx, centerline in enumerate(centerline_points_list):
            dist = abs(centerline.distance(navpath))
            if min_dist is None or dist < min_dist:
                min_dist = dist
                min_idx = idx
        # 2. 根据中心线方向 筛选出车道组
        dist_mask = np.array(
            [
                centerline.distance(centerline_points_list[min_idx]) < DISTANCE_THRE
                for centerline in centerline_points_list
            ]
        ).astype(bool)
        direction_mask = abs(centerline_direction_mean - centerline_direction_mean[min_idx]) < DIRECTION_THRE
        centerline_type_mask = np.array(
            [centerline_type not in ILLEGAL_CENTERLINE_TYPE for centerline_type in centerline_types_list]
        )
        centerline_group_mask = np.logical_and(direction_mask, dist_mask)
        centerline_group_mask = np.logical_and(centerline_group_mask, centerline_type_mask)
        valid_centerline_group = [
            cur_env["centerlines"][valid_indices[idx]][0] for idx, mask in enumerate(centerline_group_mask) if mask
        ]

        if not valid_centerline_group:
            return junction_reward
        # 3. 在车道组中找离自车最近的一条中心线
        valid_centerline_group_direction_mean = np.mean(
            [direction for idx, direction in enumerate(centerline_direction_mean) if centerline_group_mask[idx]]
        )
        min_dist = None
        selected_cl = None
        ego_pt = pred_traj[next_lanenr_idx]
        if is_human_reward_valid:
            for cl_idx, centerline in enumerate(valid_centerline_group):
                cl_first_pt_s_coord = centerline[0, 0] * np.cos(
                    valid_centerline_group_direction_mean
                ) + centerline[0, 1] * np.sin(valid_centerline_group_direction_mean)
                gt_traj_pt_s = gt_traj * np.cos(valid_centerline_group_direction_mean) + gt_traj * np.sin(
                    valid_centerline_group_direction_mean
                )
                gt_mask = np.logical_and(gt_traj_pt_s - cl_first_pt_s_coord < END_DISTANCE_THRE,
                                         gt_traj_pt_s - cl_first_pt_s_coord > 0.0)
                if not np.any(gt_mask):
                    continue
                gt_traj_near_cl = gt_traj[gt_mask]

                for gt_pt in gt_traj_near_cl:
                    single_line_min_dist = []
                    for i in range(centerline.shape[0] // 3 - 1):
                        pt2segment_dist = self.point_to_infinite_line_distance(
                            gt_pt, centerline[i, :2], centerline[i + 1, :2]
                        )
                        single_line_min_dist.append(pt2segment_dist)
                if min_dist is None or np.mean(single_line_min_dist) < min_dist:
                    min_dist = np.mean(single_line_min_dist)
                    selected_cl = centerline
        else:
            for centerline in valid_centerline_group:
                single_line_min_dist = []
                for i in range(centerline.shape[0] - 1):
                    pt2segment_dist = self.point_to_infinite_line_distance(ego_pt, centerline[i, :2], centerline[i + 1, :2])
                    single_line_min_dist.append(pt2segment_dist)
                if min_dist is None or np.mean(single_line_min_dist) < min_dist:
                    min_dist = np.mean(single_line_min_dist)
                    selected_cl = centerline
        if selected_cl is None:
            return junction_reward
        # 4. 计算生效范围
        ego_s_coord = pred_traj[:, 0] * np.cos(valid_centerline_group_direction_mean) + pred_traj[:, 1] * np.sin(
            valid_centerline_group_direction_mean
        )
        cl_s_coord = selected_cl[0][0] * np.cos(valid_centerline_group_direction_mean) + selected_cl[0][1] * np.sin(
            valid_centerline_group_direction_mean
        )
        valid_mask = np.logical_and(
            cl_s_coord - ego_s_coord < 1.5 * END_DISTANCE_THRE, ego_s_coord - cl_s_coord < END_DISTANCE_THRE
        )  # 前向15m 后向10m范围内生效
        if not np.any(valid_mask):
            return junction_reward
        # 5. 用最近距离计算奖励
        for idx in range(pred_traj.shape[0]):
            if valid_mask[idx]:
                min_dist = None
                for i in range(selected_cl.shape[0] - selected_cl.shape[0] // 3):
                    pt2segment_dist = self.point_to_infinite_line_distance(
                        pred_traj[idx], selected_cl[i, :2], selected_cl[i + selected_cl.shape[0] // 3, :2]
                    )
                    if min_dist is None or pt2segment_dist < min_dist:
                        min_dist = pt2segment_dist
                junction_reward[idx] = self.reward_zone(min_dist)
        # if passway is not None:        # 6. 除以生效点数
        junction_reward = junction_reward / (np.sum(valid_mask) + 1e-6)

        #     import matplotlib.pyplot as plt
        #     cl_plot = [centerline[0] for centerline in cur_env['centerlines']]
        #     navpath_plot = np.array(navpath.coords)

        #     for cl in cl_plot:
        #         plt.plot(cl[:,0], cl[:, 1], c='r')
        #     for cl in valid_centerline_group:
        #         plt.plot(cl[:,0], cl[:, 1],'x-', c='r')
        #     plt.plot(navpath_plot[:, 0], navpath_plot[:, 1], c="g")
        #     plt.plot(selected_cl[:, 0], selected_cl[:, 1], c="y")
        #     x = np.linspace(-10, 20, 100)
        #     y = cl_s_coord-x * np.cos(valid_centerline_group_direction_mean) / np.sin(valid_centerline_group_direction_mean)
        #     plt.plot(x, y, c='gray')
        #     # plt.plot(pred_traj[:,0], pred_traj[:,1], 'o-', c='b')
        #     # plt.plot(pred_traj[valid_mask][:,0], pred_traj[valid_mask][:,1], c='g')
        #     plt.plot(gt_traj[:,0], gt_traj[:,1], 'o--', c='black')

        #     # plt.plot(passway[0][:, 0], passway[0][:, 1], 'x--', c='orange')
        #     # plt.plot(passway[1][:, 0], passway[1][:, 1], 'x--', c='orange')
        #     import time
        #     plt.savefig(f"{time.time()}.png")
        return junction_reward

class ChooseEtcReward(BaseLoss):
    def __init__(self, fps=5) -> None:
        """Init."""
        super().__init__()
        self.fps = fps


    def check_invade_traffic_flow(self, ego_polygon_ts, speed_ts, raw_env, ts):
        """判断预测轨迹是否插入车流

        Args:
            ego_polygon_ts: ts时刻的预测polygon
            speed_ts: ts时刻的速度
            raw_env: 每个时刻的环境信息
            ts: 预测时间

        Returns:
        invade_traffic_flow: pred traj是否入侵车流
        """
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        dobjs_full = raw_env[0]["dobjs_full"]
        invade_traffic_flow = False

        # 过滤掉 后向60°（cos>0.5）或距离<0.8m的动态od
        dobjs_polygon_ts = dobjs_polygon[ts]
        if len(dobjs_polygon_ts) == 0:
            ego_rect = np.asarray(ego_polygon_ts, dtype=float)
            ego_poly = Polygon(ego_rect)
            return False, ego_poly

        ego_center = np.mean(ego_polygon_ts, axis=0, keepdims=True)
        rea_center = np.mean(ego_polygon_ts[[0, 3]], axis=0, keepdims=True)
        obj_center = np.mean(dobjs_polygon_ts, axis=1)

        ego_direct = rea_center - ego_center
        ego_direct = ego_direct / np.linalg.norm(ego_direct, axis=1, keepdims=True)

        obj_direct = obj_center - ego_center
        dst_center = np.linalg.norm(obj_direct, axis=1, keepdims=True)
        dst_center = np.clip(dst_center, 1e-6, None)
        obj_direct = obj_direct / dst_center

        cos_direct = (obj_direct @ ego_direct.T).flatten()
        dst_center = dst_center.flatten()

        # 后向30°（cos>0.86）或距离<0.8m 视为忽略
        ignore_mask = (cos_direct > 0.86) | (dst_center < 0.8)
        filtered_dobjs_polygon_at_t = dobjs_polygon_ts[~ignore_mask]

        # 计算 ego 的局部纵/横向（前向、左向）
        ego_rect = np.asarray(ego_polygon_ts, dtype=float)
        ego_poly = Polygon(ego_rect)
        if not ego_poly.is_valid:
            ego_poly = ego_poly.buffer(0)
        u_lon = ego_rect[1] - ego_rect[0]  # 右后 -> 右前（前向）
        u_lat = ego_rect[3] - ego_rect[0]  # 右后 -> 左后（左向）
        lon_len = float(np.linalg.norm(u_lon))
        lat_len = float(np.linalg.norm(u_lat))

        # 将前向旋到全局 x 轴对齐
        theta_deg = math.degrees(math.atan2(u_lon[1], u_lon[0]))
        center = tuple(ego_rect.mean(axis=0))
        ego_aligned = rotate(ego_poly, -theta_deg, origin=center, use_radians=False)

        # 计算ego_polygon的膨胀结果，中心缩放
        v_ratio = speed_ts / 20
        lon_range = max(0.6, v_ratio * 2.0)
        lat_range = 2.0
        sx = (lon_len + 2.0 * float(lon_range)) / lon_len
        sy = (lat_len + 2.0 * float(lat_range)) / lat_len
        ego_expanded_aligned = scale(ego_aligned, xfact=sx, yfact=sy, origin=center)
        ego_expanded = rotate(ego_expanded_aligned, theta_deg, origin=center, use_radians=False)

        # 构建STRtree，query与ego_expanded相交的dobjs
        filtered_dobjs_polys = [Polygon(np.asarray(rect, dtype=float)) for rect in filtered_dobjs_polygon_at_t]
        tree = STRtree(filtered_dobjs_polys)
        result_geoms = tree.query(ego_expanded)
        # 构建一个从WKT到索引的映射
        geom_to_index = {}
        for idx, geom in enumerate(filtered_dobjs_polys):
            geom_to_index[geom.wkt] = idx
        candidate_indices = [geom_to_index[geom.wkt] for geom in result_geoms]
        candidate_dobjs = [filtered_dobjs_polygon_at_t[idx] for idx in candidate_indices]

        left_obj = False
        right_obj = False
        ego_direct_front = [-ego_direct[0, 0], ego_direct[0, 1]]
        for i, poly in enumerate(result_geoms):
            if left_obj and right_obj:
                break

            if poly.intersects(ego_expanded):
                poly_center = np.mean(candidate_dobjs[i], axis=0)

                poly_direct = poly_center - ego_center[0]
                norm_center = np.linalg.norm(poly_direct)
                norm_center = np.clip(norm_center, 1e-6, None)
                poly_direct = poly_direct / norm_center

                cross = ego_direct_front[0] * poly_direct[1] - ego_direct_front[1] * poly_direct[0]
                dot = poly_direct[0] * ego_direct_front[0] + poly_direct[1] * ego_direct_front[1]
                # 大于15°
                if dot < 0.96:
                  if cross > 0:
                      left_obj = True
                  else:
                      right_obj = True
        invade_traffic_flow = left_obj and right_obj

        # temp vis
        # fig, ax = plt.subplots(figsize=(15, 5))
        # plt.tight_layout()
        # ax.set_aspect("equal")
        # ax.grid(linestyle="--")

        # if len(raw_env[0]["dobjs_full"]) > ts:
        #     # 假设pred_polygons_draw是针对第一条轨迹的
        #     x3, y3 = ego_polygon_ts.mean(axis=0)
        #     x4, y4 = ego_polygon_ts[1:3, :].mean(axis=0)
        #     pred_traj_draw = ego_polygon_ts.mean(axis=1)  # ndraw x 2
        #     ax.plot(
        #         ego_polygon_ts[:, 0],
        #         ego_polygon_ts[:, 1],
        #         color="blue",
        #         alpha=0.7,
        #         linewidth=2,
        #     )
        #     ax.plot([x3, x4], [y3, y4], color="blue", linewidth=0.5)

        #     cxcy = filtered_dobjs_polygon_at_t.mean(axis=1)
        #     pxpy = filtered_dobjs_polygon_at_t[:, 1:3, :].mean(axis=1)
        #     edge = "darkgreen"
        #     for d, (x1, y1), (x2, y2) in zip(filtered_dobjs_polygon_at_t, cxcy, pxpy):
        #             ax.fill(
        #                 d[:, 0],
        #                 d[:, 1],
        #                 color=np.array([10, 10, 10]) / 255.0,
        #                 alpha=0.3,
        #                 edgecolor="lightgray",
        #                 linewidth=0.2,
        #             )
        #     plt.savefig(f"/home/barry.xiao/melange/output/{ts}.png", dpi=300)

        return invade_traffic_flow, ego_expanded


    def get_expert_zone(self, gt_traj, raw_env, ts):
        """对gt通道的reward zone再多extend一些,与其他etc通道差异化

        Args:
            ego_polygon_ts: ts时刻的预测polygon
            speed_ts: ts时刻的速度
            raw_env: 每个时刻的环境信息
            ts: 预测时间

        Returns:
        invade_traffic_flow: pred traj是否入侵车流
        """

        expert_zone = []
        expert_s = []
        plaza_length = 0.0
        gate_centerlines = raw_env[ts]["toll_gate_center_line"]
        if len(gate_centerlines) < 1:
            return expert_zone, expert_s, plaza_length

        # 计算gt gate line起点在gt轨迹上的长度
        gate_start_proj = None
        expert_zone_s = 0.0
        for gate_line in gate_centerlines:
            if len(gate_line[1]) < 2 or not gate_line[-2] or gate_line[3] != 204:
                continue

            gate_start_proj = project_to_line(gate_line[1][0][:2], gt_traj)
            plaza_length = gate_line[4]
            if plaza_length < 40.0:
                expert_gt_s = 5.0
            elif plaza_length < 100:
                expert_gt_s = plaza_length * 0.15
            else:
                expert_gt_s = max(15, min(plaza_length * 0.12, 25))
            break

        if gate_start_proj == None:
            return expert_zone, expert_s, plaza_length
        # 计算gt轨迹的长度
        diffs = np.diff(gt_traj, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        gt_cum_lengths = np.concatenate([[0], np.cumsum(distances)])

        for point, length in zip(gt_traj, gt_cum_lengths):
            if max(plaza_length * 0.2, gate_start_proj[0] - expert_gt_s) < length < gate_start_proj[0]:
                expert_zone.append(point)
                expert_s.append(length)

        return expert_zone, expert_s, plaza_length


    def get_queue_zone(self, gt_traj, gt_polygon_ts, expert_s, plaza_length, raw_env, ts):
        """ego在站内低速时, 计算跟车排队延长的区域

        Args:
            ego_polygon_ts: ts时刻的预测polygon
            expert_s: gt通道延长的长度信息
            raw_env: 每个时刻的环境信息
            ts: 预测时间

        Returns:candidate_dobjs
        queue_zone: 排队区域
        """
        queue_zone = []
        if len(expert_s) == 0:
            return queue_zone

        # 计算gt轨迹的长度
        diffs = np.diff(gt_traj, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        gt_cum_lengths = np.concatenate([[0], np.cumsum(distances)])

        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        # 过滤掉 后向60°（cos>0.5）或距离<0.8m的动态od
        dobjs_polygon_ts = dobjs_polygon[ts]
        if len(dobjs_polygon_ts) == 0:
            return queue_zone

        ego_center = np.mean(gt_polygon_ts, axis=0, keepdims=True)
        rea_center = np.mean(gt_polygon_ts[[0, 3]], axis=0, keepdims=True)
        obj_center = np.mean(dobjs_polygon_ts, axis=1)

        ego_direct = rea_center - ego_center
        ego_direct = ego_direct / np.linalg.norm(ego_direct, axis=1, keepdims=True)

        obj_direct = obj_center - ego_center
        dst_center = np.linalg.norm(obj_direct, axis=1, keepdims=True)
        dst_center = np.clip(dst_center, 1e-6, None)
        obj_direct = obj_direct / dst_center

        cos_direct = (obj_direct @ ego_direct.T).flatten()
        dst_center = dst_center.flatten()

        # 后向30°（cos>0.86）或距离<0.8m 视为忽略
        ignore_mask = (cos_direct > 0.86) | (dst_center < 0.8)
        filtered_dobjs_polygon_at_t = dobjs_polygon_ts[~ignore_mask]

        # 构建STRtree，query与gate_line相交的dobjs
        filtered_dobjs_polys = [Polygon(np.asarray(rect, dtype=float)) for rect in filtered_dobjs_polygon_at_t]
        tree = STRtree(filtered_dobjs_polys)
        # 构建一个从WKT到索引的映射
        geom_to_index = {}
        for idx, geom in enumerate(filtered_dobjs_polys):
            geom_to_index[geom.wkt] = idx

        gt_polyline = LineString(gt_traj)
        result_geoms = tree.query(gt_polyline)
        candidate_indices = [geom_to_index[geom.wkt] for geom in result_geoms]
        candidate_dobjs = [filtered_dobjs_polygon_at_t[idx] for idx in candidate_indices]
        # 找到gt轨迹上的排队车辆
        gt_queue_info = []
        for dobj_idx, candidate_dobj in enumerate(candidate_dobjs):
            dobj_info = get_xyyaw_from_polygon(np.expand_dims(candidate_dobj, axis=0))
            gt_proj = project_to_line(dobj_info[0, :2], gt_traj, dobj_info[0, 6])
            if gt_proj == None:
                continue

            gt_queue_info.append([dobj_idx, gt_proj[0]])

        if len(gt_queue_info) > 0:
            gt_queue_info.sort(key=lambda x: x[1])
            start_queue_s = max(plaza_length * 0.2, gt_queue_info[0][1] - 2.0)
            if start_queue_s < expert_s[0]:
                for point, length in zip(gt_traj, gt_cum_lengths):
                    if start_queue_s < length < expert_s[0]:
                        queue_zone.append(point)

        return queue_zone

    def process_gt_reward_zone(self, gt_polygon, gt_path_info, raw_env, his_num):
        expert_zone = []
        queue_zone = []
        expert_s = []
        plaza_length = 0.0
        get_static_expert = False
        for ts in range(len(gt_polygon)):
            gate_centerlines = raw_env[ts]["toll_gate_center_line"]
            if len(gate_centerlines) < 1:
                continue

            if not gate_centerlines[0][6]:
                continue

            gate_proj = project_to_line(gt_path_info[ts, :2], gate_centerlines[0][1][:, :2])
            if gate_proj and gate_proj[0] > 0:
                break

            if not get_static_expert:
                expert_zone, expert_s ,plaza_length= self.get_expert_zone(gt_path_info[:,:2], raw_env, ts)
                get_static_expert = True

            # if get_static_expert and ts > 0:
            #     gt_delta_s = np.mean(gt_polygon[ts] - gt_polygon[ts - 1], axis=-2)
            #     gt_vel = np.linalg.norm(gt_delta_s) * self.fps
            #     if gt_vel < 1.5:
            #         queue_zone = self.get_queue_zone(gt_path_info[:,:2], gt_polygon[ts], expert_s, plaza_length, raw_env, ts + his_num)
            #         break

        # reward_zone = queue_zone + expert_zone
        return expert_zone

    def compute_choose_etc_progress_reward(self, choose_gt_gate, gate_index, gt_gate_index, dist_to_gate, speeds):
        """计算站前效率选口reward.

        Args:
            choose_gt_gate: pred traj与gt选择相同通道
            gate_index: pred traj 进入闸机通道时的index
            dist_to_gate: pred起始时刻距离闸机通道的距离
            raw_env: 每个时刻的环境信息

        Returns:
            choose_etc_progress_reward: 效率选口reward平摊在站前轨迹
        """
        choose_etc_progress_reward = [0.0] * (gate_index + 1)
        if choose_gt_gate:
            base_progress_reward = 40
        else:
            base_progress_reward = 5

        if gt_gate_index > 0:
            # 计算理论最小步数(优先使用真值进入gate的步数，考虑蓝圈限速的偏差）
            min_theoretical_steps = gt_gate_index + 10

            # 计算步数效率（理论最小步数/实际步数）
            efficiency = min_theoretical_steps / (gate_index + 1) if gate_index > 0 else 0

            # 归一化处理，考虑初始距离的影响
            distance_factor = 1 - np.exp(-dist_to_gate / 10)  # 距离越远，基础奖励越高

            # 综合奖励，基础奖励为20
            select_etc_progress_reward = efficiency * distance_factor * base_progress_reward
        else:
            select_etc_progress_reward = 0

        speeds_sum = np.sum(speeds[:gate_index])
        for i in range(gate_index):
            r = select_etc_progress_reward * speeds[i] / speeds_sum if speeds_sum > 0 else 0
            choose_etc_progress_reward[i] = r

        return choose_etc_progress_reward

    def forward(self, pred_polygon, gt_polygon, speeds, raw_env, his_num):
        """Forward."""
        ego_path_info = get_xyyaw_from_polygon(pred_polygon)
        gt_path_info = get_xyyaw_from_polygon(gt_polygon)
        pred_num = len(ego_path_info)
        choose_etc_reward = [0.0] * pred_num
        expert_zone_np = np.zeros((0, 12), dtype=np.float32)

        # ego轨迹下采样
        ego_path_downsample = []
        ego_path_downsample.append([0, ego_path_info[0]])
        for i in range(1, len(ego_path_info)):
            c_distance = np.linalg.norm(ego_path_info[i, :2] - ego_path_downsample[-1][1][:2])
            if c_distance > 2.0:
                ego_path_downsample.append([i, ego_path_info[i]])

        # 对gt通道进行额外reward奖励
        expert_zone_length = 0
        expert_reward_zone = self.process_gt_reward_zone(gt_polygon, gt_path_info, raw_env, his_num)
        expert_zone_np = np.array(expert_reward_zone)
        if len(expert_reward_zone) > 0:
            expert_diffs = np.diff(expert_zone_np, axis=0)
            segment_lengths = np.linalg.norm(expert_diffs, axis=1)
            expert_zone_length = np.sum(segment_lengths)

        # 计算进入闸机段区域时的投影距离
        choose_etc_gate = False
        choose_gt_gate = False
        no_entry_toll_scene = all_unknown_gate = True
        expert_dis = {}
        force_dis = {}
        recommend_dis = {}
        near_gate = False
        entry_plaza_index = -1
        entry_gate_index = 0
        gt_gate_index = -1
        dist_to_gate = 0.0
        for ori_idx, path_point in ego_path_downsample:
            gate_centerlines = raw_env[ori_idx]["toll_gate_center_line"]
            if len(gate_centerlines) < 1:
                continue

            no_entry_toll_scene = False
            # 通过在通道横截线上的投影来排序
            centerline_points = gate_centerlines[0][1][:,:2]
            if centerline_points.shape[0] < 3: continue
            gate_start_points = [center_line[2][:2] for center_line in gate_centerlines if len(center_line[2]) > 0]
            start_proj_res = []
            for start_point in gate_start_points:
                start_proj = project_to_line(start_point, centerline_points)
                start_proj_res.append(start_proj[1])

            gate_start_points = [gate_start_points[i] for i in np.argsort(start_proj_res)[::-1]]
            gate_start_points = np.array(gate_start_points)

            gate_types = [center_line[3] for center_line in gate_centerlines]
            if all(x != 204 for x in gate_types):
                continue
            target_gate_type = 204
            # if all(x == 209 for x in gate_types):
            #     continue
            # else:
            #     target_gate_type = min(gate_types)



            if entry_plaza_index < 0 and gate_centerlines[0][6]:
                entry_plaza_index = ori_idx
            all_unknown_gate = False
            entrance_proj_res = project_to_line(path_point[:2], gate_start_points)
            if entrance_proj_res is None:
                continue
            elif dist_to_gate == 0:
                dist_to_gate = abs(entrance_proj_res[1])

            if len(expert_reward_zone) > 0:
                expert_proj = project_to_line(path_point[:2], expert_zone_np)
                if expert_proj and expert_proj[0] > 0 and expert_proj[0] < expert_zone_length:
                    expert_dis[ori_idx] = abs(expert_proj[1])

            recommend_extend_length = np.linalg.norm(centerline_points[0] - centerline_points[1])
            force_extend_length = np.linalg.norm(centerline_points[1] - centerline_points[2])
            gate_lengths = []
            dist_map_entrance = []
            dist_map_exit = []
            # 计算外延最长的gate_line在gate横截线上的投影距离
            for gate in gate_centerlines:
                total_length = 0
                extend_points = gate[1]
                for i in range(len(extend_points) - 1):
                    x1, y1, _ = extend_points[i]
                    x2, y2, _ = extend_points[i + 1]
                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    total_length += distance
                gate_lengths.append(total_length)
                # start point dist to map entrance line
                gate_start_proj = project_to_line(extend_points[0][:2], gate_start_points)
                dist_map_entrance.append(gate_start_proj[1])
                gate_end_proj = project_to_line(extend_points[-1][:2], gate_start_points)
                dist_map_exit.append(gate_end_proj[1])
                # 判断GT是否进入闸机段
                if (gt_gate_index < 0
                    and gate[5]
                    and ori_idx < len(gt_path_info)
                    and gt_path_info[ori_idx, 0] >= gate[0][0, 0]):
                    gt_gate_index = ori_idx

            min_dist_map_exit = min(dist_map_exit)
            min_dist_map_entrance = min(dist_map_entrance)

            s, l = entrance_proj_res
            if l > min_dist_map_entrance:
                entry_gate_index = ori_idx

                min_l = [float("inf"), float("inf")]  # 0-all gates,1-etc gates
                min_s = [float("-inf"), float("-inf")]
                min_index = [-1, -1]
                for index, center_line in enumerate(gate_centerlines):
                    points = center_line[1][:, :2]
                    gate_proj_res = project_to_line(path_point[:2], points, path_point[6])
                    if gate_proj_res is None:
                        continue

                    near_gate = True
                    if abs(gate_proj_res[1]) < min_l[0]:
                        min_l[0] = abs(gate_proj_res[1])
                        min_s[0] = gate_proj_res[0]
                        min_index[0] = index

                    if center_line[3] in [target_gate_type] and abs(gate_proj_res[1]) < min_l[1]:
                        min_l[1] = abs(gate_proj_res[1])
                        min_s[1] = gate_proj_res[0]
                        min_index[1] = index

                if min_index[0] >= 0 and min_l[0] < 2.0:
                    choose_etc_gate = gate_types[min_index[0]] == target_gate_type
                    choose_gt_gate = gate_centerlines[min_index[0]][5]
                else:
                    choose_etc_gate = False

                if min_index[1] >= 0:
                    if min_s[1] <= recommend_extend_length:
                        # 横向距离及是否为gt通道
                        recommend_dis[ori_idx] = [min_l[1], gate_centerlines[min_index[1]][5]]
                    else:
                        force_dis[ori_idx] = [min_l[1], gate_centerlines[min_index[1]][5]]

                if min_s[0] > recommend_extend_length + force_extend_length + 2.0:
                    near_gate = True
                    break

        # if not near_gate:
        #     choose_etc_gate = True

        # 计算静态选口reward
        for ego_traj_index, _ in ego_path_downsample:
            if ego_traj_index in recommend_dis:
                dist = abs(recommend_dis[ego_traj_index][0])
                if recommend_dis[ego_traj_index][1]:
                    min_dist_reward = [0.2, 4.0]
                    max_dist_reward = [1.0, 0.0]
                else:
                    min_dist_reward = [0.2, 2.0]
                    max_dist_reward = [1.0, 0.0]

                if dist < min_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] = min_dist_reward[1]
                elif dist > max_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] = max_dist_reward[1]
                else:
                    choose_etc_reward[ego_traj_index] = np.interp(
                        dist,
                        np.array([min_dist_reward[0], max_dist_reward[0]]),
                        np.array([min_dist_reward[1], max_dist_reward[1]]),
                    )
            elif ego_traj_index in force_dis:
                dist = abs(force_dis[ego_traj_index][0])
                if force_dis[ego_traj_index][1]:
                    min_dist_reward = [0.2, 8.0]
                    max_dist_reward = [1.2, -5.0]
                else:
                    min_dist_reward = [0.2, 4.0]
                    max_dist_reward = [1.2, -5.0]

                if dist < min_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] = min_dist_reward[1]
                elif dist > max_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] = max_dist_reward[1]
                else:
                    choose_etc_reward[ego_traj_index] = np.interp(
                        dist,
                        np.array([min_dist_reward[0], max_dist_reward[0]]),
                        np.array([min_dist_reward[1], max_dist_reward[1]]),
                    )

            if ego_traj_index in expert_dis:
                dist = abs(expert_dis[ego_traj_index])
                min_dist_reward = [0.2, 2.0]
                max_dist_reward = [1.0, 0.0]
                if dist < min_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] += min_dist_reward[1]
                elif dist > max_dist_reward[0]:
                    choose_etc_reward[ego_traj_index] += max_dist_reward[1]
                else:
                    choose_etc_reward[ego_traj_index] += np.interp(
                        dist,
                        np.array([min_dist_reward[0], max_dist_reward[0]]),
                        np.array([min_dist_reward[1], max_dist_reward[1]]),
                    )
        # if entry_plaza_index >= 0:
        #     for ts in range(entry_plaza_index, entry_gate_index):
        #         gate_centerlines = raw_env[ts]["toll_gate_center_line"]
        #         if len(gate_centerlines) < 1:
        #             continue

        #         invade_traffic_flow[ts], ego_expand[ts] = self.check_invade_traffic_flow(pred_polygon[ts], speeds[ts], raw_env, ts + his_num)

        # 考虑选口progress reward
        # if entry_gate_index > 0 and choose_etc_gate:
        #     choose_progress_reward = self.compute_choose_etc_progress_reward(
        #         choose_gt_gate, entry_gate_index, gt_gate_index, dist_to_gate, speeds
        #     )
        #     for i in range(len(choose_progress_reward)):
        #         choose_etc_reward[i] += choose_progress_reward[i]

        return choose_etc_reward, choose_etc_gate, all_unknown_gate, no_entry_toll_scene, entry_gate_index, expert_zone_np
