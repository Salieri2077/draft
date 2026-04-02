# -*- coding: utf-8 -*-
"""Astra planner head."""
import multiprocessing as mp
from typing import Any
from typing import Dict

import numpy as np
import torch
import statistics
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from torchpilot.model.module_interface import BaseModule
from torchpilot.utils.registries import HEAD
from torchpilot.utils.registries import LOSSES
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point
from copy import deepcopy
from collections import defaultdict

from tpp_onemodel.utils.reward_utils import compute_center_distance, trim_nonzero_runs
from tpp_onemodel.data.dataset.plannn2_dataset_utils.common import ScenarioEnum
from tpp_onemodel.data.dataset.plannn2_dataset_utils.common import get_map_cls_to_3cls
from tpp_onemodel.utils.deadcar_bypass_detector_v3 import DeadCarBypassDetectorV3
from tpp_onemodel.utils.reward_utils import Turntype
from tpp_onemodel.utils.reward_utils import calculate_signed_lateral_distance
from tpp_onemodel.utils.reward_utils import compute_center_distance
from tpp_onemodel.utils.reward_utils import get_distance
from tpp_onemodel.utils.reward_utils import cut_sub_path
from tpp_onemodel.utils.reward_utils import get_relative_pose_from_obj
from tpp_onemodel.utils.reward_utils import get_xyyaw_from_polygon
from tpp_onemodel.utils.reward_utils import interpolate_points
from tpp_onemodel.utils.reward_utils import line_intersection_point
from tpp_onemodel.utils.reward_utils import polygon_to_segments
from tpp_onemodel.utils.reward_utils import segment_intersect
from tpp_onemodel.utils.reward_utils import target_progress_scenario_exclude_condition
from tpp_onemodel.utils.reward_utils import calculate_signed_lateral_distance
from tpp_onemodel.data.dataset.plannn2_dataset_utils.common import get_map_cls_to_3cls
from tpp_onemodel.utils.reward_utils import judge_intersection_and_maneuver
from tpp_onemodel.utils.reward_utils import is_slow_scene
from tpp_onemodel.utils.reference_line_provider import ReferenceLine
from tpp_onemodel.utils.reward_utils import is_campus_scene
from tpp_onemodel.utils.reward_utils import is_campus_scene_v2
from tpp_onemodel.utils.rl_utils import get_default_reward_summary_config
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import transform_centerline_matrix
from tpp_onemodel.data.dataset.plannn2_dataset_utils.utils import point_to_centerline_dis
from tpp_onemodel.data.dataset.plannn2_dataset_utils.common import ScenarioEnum
from tpp_onemodel.utils.car_queue_bypass import CarQueueBypassDetector
from tpp_onemodel.utils.slow_follow_detector_v3_1 import SlowFollowDetectorV3_1
from tpp_onemodel.utils.slow_follow_detector_v3_2 import SlowFollowSegment, SlowFollowDetectorV3_2
# from tpp_onemodel.utils.slow_follow_detector_v3 import SlowFollowSegment
from tpp_onemodel.utils.reward_utils import Turntype
from tpp_onemodel.utils.reward_utils import determine_path_turn_type
from tpp_onemodel.utils.deadcar_bypass_detector_v3 import DeadCarBypassDetectorV3
from tpp_onemodel.utils.reference_line import ReferenceLine
from torchpilot.utils.stopwatch import stopwatch
import time


@HEAD.register_module()
class RLPlannerRewardModel(BaseModule):
    """RL planner reward model for trajectory planning.

    Args:
        collision_weight_decay (float, optional): Decay factor for collision penalties
            over time. Defaults to 0.92.
        history_t_num (int, optional): Number of history timesteps to consider.
            Defaults to 15.
        pred_t_num (int, optional): Number of prediction timesteps.
            Defaults to 35.

    Returns:
        torch.Tensor: Total reward combining all factors. Shape: [B, N]
            where B is batch size and N is number of trajectories.
    """

    def __init__(
        self,
        history_t_num=15,
        pred_t_num=25,
        summary_cfg=None,
        multi_process=False,
        gdpo_reward_weight=None,
        gdpo_reward_decay=None,
        reward_extra_cfg: dict = {},
    ):
        """Init."""
        super().__init__()
        self._stopwatch = None
        self._history_t_num = history_t_num
        self._pred_t_num = pred_t_num
        self.summary_cfg = summary_cfg
        if self.summary_cfg is None:
            summary_cfg = get_default_reward_summary_config()

        self.reward_funcs = {}
        for key, cfg in summary_cfg["rewards_types_cfg"].items():
            self.reward_funcs[key] = LOSSES.build(cfg["func"])

        self.rewards_weights = summary_cfg["rewards_weights_cfg"]

        self.reward_extra_cfg = reward_extra_cfg

        self.dt = summary_cfg.get("dt", 0.2)
        self.multi_process = multi_process
        self._reward_norm = summary_cfg.get("reward_norm", False)
        self.fine_tune = summary_cfg.get("fine_tune", False)
        self.fps = summary_cfg.get("fps", 5)
        self.gdpo_reward_weight = gdpo_reward_weight
        self.gdpo_reward_decay = gdpo_reward_decay
        # 初始化多进程池
        self.proc_pool = None
        if self.multi_process:
            self.proc_pool = mp.Pool(processes=32)

    def get_stopline_cross_points(self, ego_polygons, stop_lines, threshold=0.5):
        """Get stopline cross points.

        Args:
            ego_polygons: np.ndarray [T, 4, 2]
            stop_lines: list of dict with key "stop_line", value: list of np.ndarray [2, 2]
            threshold: float, distance threshold to consider two stoplines the same
        Returns:
            A list of (frame_index, stopline_id, intersection_point)
        """
        intersections = []
        seen_stoplines = []

        def get_midpoint(line):
            return (np.array(line[0]) + np.array(line[1])) / 2

        def find_existing_stopline_id(mid):
            for idx, existing_mid in enumerate(seen_stoplines):
                if np.linalg.norm(existing_mid - mid) < threshold:
                    return idx
            return None

        for t in range(ego_polygons.shape[0]):
            polygon = ego_polygons[t]
            poly_segs = polygon_to_segments(polygon)
            for stopline in stop_lines[t]["stop_line"]:
                stop_seg = (stopline[0], stopline[1])
                mid = get_midpoint(stop_seg)
                stopline_id = find_existing_stopline_id(mid)
                if stopline_id is None:
                    # New stopline
                    stopline_id = len(seen_stoplines)
                    seen_stoplines.append(mid)
                # Check if already recorded
                if any(item[1] == stopline_id for item in intersections):
                    continue
                for seg in poly_segs:
                    if segment_intersect(seg[0], seg[1], stop_seg[0], stop_seg[1]):
                        pt = line_intersection_point(
                            np.array(seg[0]), np.array(seg[1]), np.array(stop_seg[0]), np.array(stop_seg[1])
                        )
                        if pt is not None:
                            intersections.append((t, stopline_id, pt))
                            break  # 只记录一次交点
        return intersections

    def compute_collision_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, Any]:
        """计算碰撞reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]

        rear_collision_tids = set()
        sta_min_distance_list = []
        sta_min_distance_start_edge_flag_list = []
        dyn_min_distance_list = []
        reward_results.update(
            {
                "collision_reward": np.zeros_like(reward_results["reward"]),
                "collision": 0,
                "collision_static": 0,
                "collision_except_rear": 0,
                "dis": np.zeros(self._pred_t_num + 1) + 1.5,
                "rear_collision_tids": {},
                "final_result": 0,
            }
        )
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            is_ego_touch_solid_line_now = None
            if "is_ego_intersect_solid_line" in reward_results:
                is_ego_touch_solid_line_now = reward_results["is_ego_intersect_solid_line"][pred_ts + 1]

            if "cross_solid_line_reward" in reward_results:
                lane_lines_cur = reward_results["cross_solid_line_reward"][: pred_ts + 1]
            else:
                lane_lines_cur = None

            lc_ahead_obj_id = None
            if "lc_ahead_obj_id" in reward_results:
                lc_ahead_obj_id = reward_results["lc_ahead_obj_id"][1:]
            (
                collision_with_static,
                collision_with_dyn_normal,
                collision_with_dyn_rear,
                collision_with_sta_dyn,
                collision_penalty,
                static_distance,
                static_edge_start_mindist,
                normal_dynamic_distance,
                rear_collision_tids,
            ) = self.reward_funcs["collision_reward"](
                pred_polygon,
                raw_env,
                pred_ts + self._history_t_num,
                self._history_t_num,
                rear_collision_tids,
                lane_lines_cur,
                is_ego_touch_solid_line_now=is_ego_touch_solid_line_now,
                lc_ahead_obj_id = lc_ahead_obj_id,
            )
            reward_results["rear_collision_tids"][pred_ts] = rear_collision_tids.copy()
            # reward_results["dis"][pred_ts + 1] = min_distance
            # min_distance_list.append(min_distance)
            reward_results["dis"][pred_ts + 1] = min(static_distance, normal_dynamic_distance)
            sta_min_distance_list.append(static_distance)
            sta_min_distance_start_edge_flag_list.append(static_edge_start_mindist)
            dyn_min_distance_list.append(normal_dynamic_distance)

            if collision_with_static or collision_with_dyn_normal or collision_with_dyn_rear:
                reward_results["collision"] = 1

            collision = collision_with_static or collision_with_dyn_normal

            if collision:
                ## 训练后期1/4，去掉非静止障碍物Done的策略，且限制在路口生效
                iter_num = policy_update_output["iter_num"]
                if iter_num > 2100 and not collision_with_sta_dyn:
                    direct_light = -1
                    if pred_ts < 15:
                        direct_light = judge_intersection_and_maneuver(pred_polygon[: pred_ts + 1], raw_env, pred_ts)
                    else:
                        direct_light = judge_intersection_and_maneuver(
                            pred_polygon[pred_ts - 15 : pred_ts], raw_env, pred_ts
                        )

                    if not np.isnan(direct_light):
                        reward_results["final_result"] = 2
                        reward_results["collision_reward"][pred_ts + 1] = -collision_penalty
                        reward_results["collision_reward"][pred_ts + 1 :] = -collision_penalty
                        # reward_results["dones"][pred_ts + 1] = 1
                        # if pred_ts < self._pred_t_num - 1:
                        #     reward_results["dones"][pred_ts + 2 :] = -1
                        # hard code for alignment with plannn2
                        # reward_results["reward"][pred_ts + 1 :] = 0  ## 注释这一行，撞了之后的轨迹点，仍然计算reward
                        # reward_results["result"][pred_ts + 1 :] = 2
                        reward_results["collision_except_rear"] = collision_with_static or collision_with_dyn_normal
                        reward_results["collision_static"] = collision_with_static
                        # break
                    else:
                        reward_results["final_result"] = 2
                        reward_results["collision_reward"][pred_ts + 1] = -collision_penalty
                        reward_results["dones"][pred_ts + 1] = 1
                        if pred_ts < self._pred_t_num - 1:
                            reward_results["dones"][pred_ts + 2 :] = -1
                        # hard code for alignment with plannn2
                        reward_results["reward"][pred_ts + 1 :] = 0
                        reward_results["result"][pred_ts + 1 :] = 2
                        reward_results["collision_except_rear"] = collision_with_static or collision_with_dyn_normal
                        reward_results["collision_static"] = collision_with_static
                        break
                else:
                    reward_results["final_result"] = 2
                    reward_results["collision_reward"][pred_ts + 1] = -collision_penalty
                    reward_results["dones"][pred_ts + 1] = 1
                    if pred_ts < self._pred_t_num - 1:
                        reward_results["dones"][pred_ts + 2 :] = -1
                    # hard code for alignment with plannn2
                    reward_results["reward"][pred_ts + 1 :] = 0
                    reward_results["result"][pred_ts + 1 :] = 2
                    reward_results["collision_except_rear"] = collision_with_static or collision_with_dyn_normal
                    reward_results["collision_static"] = collision_with_static
                    break

        # reward_results["rear_collision_tids"] = rear_collision_tids
        reward_results["sta_min_distance_list"] = sta_min_distance_list
        reward_results["sta_min_distance_start_edge_flag_list"] = sta_min_distance_start_edge_flag_list
        reward_results["dyn_min_distance_list"] = dyn_min_distance_list

        return reward_results

    def compute_ttc_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, Any]:
        """计算TTC reward."""
        ego_polygon = policy_update_output["ego_polygon"]
        gt_polygon = policy_update_output["gt_polygon"]

        bypass_deadcar_flag = ScenarioEnum.kDeadCarBypass == policy_update_output['scenario_id']
        bypass_slow_flag = ScenarioEnum.kFollowSlow == policy_update_output['scenario_id']
        bypass_flag = bypass_deadcar_flag or bypass_slow_flag


        reward_results.update(
            {
                "ttc_reward": np.zeros_like(reward_results["reward"]),
                "velocity_scale": np.zeros_like(reward_results["reward"]),
                "traj_ttc_saturated": np.full_like(reward_results["reward"], False, dtype=bool),
                "encounter_ttc_reward": np.zeros_like(reward_results["reward"]),
                "sta_ttc_max_distance_list": np.zeros_like(reward_results["reward"]),
                "turn_left_cross_flag": np.full_like(reward_results["reward"], False, dtype=bool),
                "straight_cross_flag": np.full_like(reward_results["reward"], False, dtype=bool),
                "vru_caution": np.zeros_like(reward_results["reward"]),
            }
        )
        assert "rear_collision_tids" in reward_results

        vru_preds = {}
        vru_preds_plg = {}

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break

            ttc_penalty, velocity_scale, traj_ttc_saturated, max_encounter_penalty, max_distance_mindist_sta, cross_flag, vru_caution = self.reward_funcs["ttc_reward"](
                gt_polygon,
                ego_polygon,
                raw_env,
                pred_ts + self._history_t_num,
                reward_results["rear_collision_tids"][pred_ts],
                ignore_rear_obj=True,
                bypass_flag=bypass_flag,
                vru_preds=vru_preds,
                vru_preds_plg=vru_preds_plg,
            )
            reward_results["ttc_reward"][pred_ts + 1] = -ttc_penalty
            reward_results["velocity_scale"][pred_ts + 1] = velocity_scale
            reward_results["traj_ttc_saturated"][pred_ts + 1] = traj_ttc_saturated
            reward_results["encounter_ttc_reward"][pred_ts + 1] = -max_encounter_penalty
            reward_results["sta_ttc_max_distance_list"][pred_ts + 1] = max_distance_mindist_sta
            reward_results["vru_caution"][pred_ts + 1] = vru_caution
            if cross_flag:
                reward_results["turn_left_cross_flag"][pred_ts + 1] = cross_flag[0]
                reward_results["straight_cross_flag"][pred_ts + 1] = cross_flag[1]
        if np.any(reward_results["ttc_reward"] < 0):
            reward_results["traffic_light_reward"][reward_results["traffic_light_reward"] > -10] = 0
        if np.any(reward_results["collision_reward"] < 0):
            reward_results["traffic_light_reward"][reward_results["traffic_light_reward"] > -10] = 0

        return reward_results

    def compute_min_distance_reward(self, policy_update_output, reward_results) -> Dict[str, Any]:
        """计算最小距离惩罚reward."""
        # assert "min_distance_list" in reward_results
        # min_distance_list = reward_results["min_distance_list"]
        assert "sta_min_distance_list" in reward_results
        assert "sta_min_distance_start_edge_flag_list" in reward_results
        assert "dyn_min_distance_list" in reward_results
        sta_min_distance_list = reward_results["sta_min_distance_list"]
        sta_min_distance_start_edge_flag_list = reward_results["sta_min_distance_start_edge_flag_list"]
        dyn_min_distance_list = reward_results["dyn_min_distance_list"]
        sta_ttc_max_distance_list = reward_results["sta_ttc_max_distance_list"]
        speeds = policy_update_output["speeds"]
        min_distance_reward = np.zeros(self._pred_t_num + 1, dtype=np.float64)
        velocity_scale_list = reward_results["velocity_scale"]

        # min_distance_reward = self.reward_funcs["min_distance_reward"](min_distance_list, velocity_scale_list, min_distance_reward)
        sta_min_distance_reward = np.zeros(self._pred_t_num + 1, dtype=np.float64)
        dyn_min_distance_reward = np.zeros(self._pred_t_num + 1, dtype=np.float64)

        min_distance_reward, sta_min_distance_reward, dyn_min_distance_reward = self.reward_funcs[
            "min_distance_reward"
        ](
            sta_min_distance_list,
            sta_min_distance_start_edge_flag_list,
            dyn_min_distance_list,
            sta_ttc_max_distance_list,
            min_distance_reward,
            sta_min_distance_reward,
            dyn_min_distance_reward,
            velocity_scale_list,
        )
        logic_1 = np.logical_and(np.array([0.0] + speeds) <= 5, reward_results["dis"] < 0.2)
        logic_2 =np.logical_and(np.array([0.0] + speeds) <= 10, reward_results["dis"] < 0.3)
        logic_3 =np.logical_and(np.array([0.0] + speeds) <= 15, reward_results["dis"] < 0.4)
        logic_4 =np.logical_and(np.array([0.0] + speeds) <= 20, reward_results["dis"] < 0.6)
        min_distance_reward[logic_1 | logic_2 | logic_3 | logic_4] = 0
        reward_results["min_distance_reward"] = min_distance_reward
        return reward_results

    def compute_cross_solid_line_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, Any]:
        """计算跨实线reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        cross_solid_line_reward = np.zeros_like(reward_results["reward"])
        is_ego_intersect_solid_line = np.zeros_like(reward_results["reward"])

        continue_intersection_with_lines_frames = 0

        scenario_id=policy_update_output['scenario_id']
        is_campus = is_campus_scene(scenario_id)

        if not is_campus:
            for pred_ts in range(0, self._pred_t_num):
                if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                    break
                pred_polygon_at_t = pred_polygon[pred_ts]
                cur_env = raw_env[pred_ts + self._history_t_num]
                lane_nr_remain_distance = cur_env["navi_infos"]["lane_nr_remain_distance"]
                # 50m 内压线惩罚
                # 历史连续10帧 压线 则惩罚：实线+1 else +0.2
                last_continue_intersection_with_lines_frames = continue_intersection_with_lines_frames
                line_reward, continue_intersection_with_lines_frames = self.reward_funcs["cross_solid_line_reward"](
                    pred_polygon_at_t, cur_env, pred_ts, lane_nr_remain_distance, continue_intersection_with_lines_frames
                )
                cross_solid_line_reward[pred_ts + 1] = line_reward
                is_ego_intersect_solid_line[pred_ts + 1] = (
                    continue_intersection_with_lines_frames > last_continue_intersection_with_lines_frames
                )

        reward_results.update(
            {
                "cross_solid_line_reward": cross_solid_line_reward,
                "is_ego_intersect_solid_line": is_ego_intersect_solid_line,
            }
        )
        return reward_results

    def _is_in_junction(self, raw_env, pred_polygons):
        raw_env = raw_env[self._history_t_num :]
        # 1. lanenr 距离非单调递减
        lanenr_dist_list = [env["navi_infos"]["lane_nr_remain_distance"] for env in raw_env]
        lane_num_list = [len(env["navi_infos"]["lane_infos"]) for env in raw_env]
        is_cross_junction = False
        pre_dist = None
        next_lanenr_idx = -1
        for frame_idx, dist in enumerate(lanenr_dist_list):
            if pre_dist is not None and dist > pre_dist:
                is_cross_junction = True
                next_lanenr_idx = frame_idx
                break
            pre_dist = dist
        is_lanenr_num_larger_equal_2 = is_cross_junction and lane_num_list[next_lanenr_idx] >= 2 and lane_num_list[next_lanenr_idx - 1] >= 2
        # 2. 节点处存在停止线且clip轨迹穿过停止线
        is_cross_stopline = False
        if is_cross_junction:
            is_cross_stopline = len(self.get_stopline_cross_points(pred_polygons, raw_env, 2.0)) > 0

        return is_cross_stopline and is_cross_junction and is_lanenr_num_larger_equal_2, next_lanenr_idx

    def _is_near_junction(self, ego_point, line_points, start_thresh, end_thresh):
        """路口判断：计算ego_point到车道线段的投影，并判断投影点是否在线段范围内"""
        try:
            line_string = LineString(line_points)
            if line_string.is_empty:
                return True
            # 计算投影点
            projected_point = line_string.interpolate(line_string.project(ego_point))
            # 计算投影点距离线段起点的距离
            dist_to_start = LineString([line_points[0], projected_point]).length
            dist_to_end = LineString([projected_point, line_points[-1]]).length
            # 如果投影点在线段范围内，且距离起点和终点都大于阈值，则认为不在路口
            if dist_to_start > start_thresh and dist_to_end > end_thresh:
                return False
            return True
        except Exception as e:
            return True

    def _create_center_line(self, line1, line2, num_points=100):
        """
        通过插值计算两条平行线段的中心线

        Args:
            line1: LineString, 第一条线段
            line2: LineString, 第二条线段
            num_points: 插值点数（越多越精确）
        """

        # 在两条线上等间距采样
        center_points = []
        for i in range(num_points):
            # 标准化插值位置 [0, 1]
            t = i / (num_points - 1) if num_points > 1 else 0.5

            # 在两条线上分别插值
            point1 = line1.interpolate(t, normalized=True)
            point2 = line2.interpolate(t, normalized=True)

            # 计算中点
            mid_x = (point1.x + point2.x) / 2
            mid_y = (point1.y + point2.y) / 2
            center_points.append((mid_x, mid_y))

        return True, LineString(center_points)

    def _check_junction_scenario(self, cur_env, ego_point):
        JUNCTION_START_THRESH = 5
        JUNCTION_END_THRESH = 10
        lane_lines = cur_env["lane_lines"]  # list
        distances = []
        line_datas = []
        line_data1 = []
        line_data2 = []
        for points, _ in lane_lines:
            try:
                if len(points) < 2:
                    continue
                points = [x[:2] for x in points]
                line_polyline = LineString(points)
                distance = abs(line_polyline.distance(ego_point))
                if distance > 4.0:  # 距离自车中心超过4m的车道线不考虑，考虑了5m的超宽车道
                    continue
                distances.append(abs(distance))
                line_datas.append(points)
            except:
                continue
        if len(distances) < 2:  # 车道线不标准，则处于路口内，不居中
            return True, line_data1, line_data2
        else:
            sorted_indices = np.argsort(distances)
            closet_line_datas = [line_datas[i] for i in sorted_indices[:2]]  # 自车最近的两条车道线
            line_data1, line_data2 = closet_line_datas[0], closet_line_datas[1]
            if self._is_near_junction(
                ego_point, line_data1, JUNCTION_START_THRESH, JUNCTION_END_THRESH
            ) or self._is_near_junction(ego_point, line_data2, JUNCTION_START_THRESH, JUNCTION_END_THRESH):
                return True, line_data1, line_data2
        return False, line_data1, line_data2
    def _check_junction_scenario_step(self, cur_env, ego_point):
        JUNCTION_START_THRESH = 5
        JUNCTION_END_THRESH = 10
        lane_lines = cur_env["lane_lines"]  # list
        distances = []
        line_datas = []
        line_data1 = []
        line_data2 = []
        for points, _ in lane_lines:
            try:
                if len(points) < 2:
                    continue
                points = [x[:2] for x in points]
                line_polyline = LineString(points)
                distance = abs(line_polyline.distance(ego_point))
                if distance > 4.0:  # 距离自车中心超过4m的车道线不考虑，考虑了5m的超宽车道
                    continue
                distances.append(abs(distance))
                line_datas.append(points)
            except:
                continue
        if len(distances) < 2:  # 车道线不标准，则处于路口内，不居中
            return True, line_data1, line_data2
        else:
            sorted_indices = np.argsort(distances)
            closet_line_datas = [line_datas[i] for i in sorted_indices[:2]]  # 自车最近的两条车道线
            line_data1, line_data2 = closet_line_datas[0], closet_line_datas[1]

            if self._is_near_junction(
                ego_point, line_data1, JUNCTION_START_THRESH, JUNCTION_END_THRESH
            ) or self._is_near_junction(ego_point, line_data2, JUNCTION_START_THRESH, JUNCTION_END_THRESH):
                # 判断是不是小路
                link_info = cur_env.get("link_info", None)
                is_narrow_road_scenario = False
                if link_info is not None and len(link_info)>0 and link_info[0] is not None:
                    is_narrow_road_scenario = link_info[0].get("road_class", 0) >= 7
                # 不是小路维持原来逻辑
                if is_narrow_road_scenario is False:
                    return True, line_data1, line_data2
                # 是小路则double check, 若两个都是路口才给True
                elif self._check_narrow_road_junction(cur_env):
                    return True, line_data1, line_data2

        return False, line_data1, line_data2

    def _check_narrow_road_junction(self, cur_env):
        # 在拥堵情况下小路感知线会有遮挡，使用_is_near_junction会导致因为线出的不够长导致错判为路口
        # 错判为路口会导致od_scenenario的判断失效，存在navi_lane_reward和navi_lane_change_reward（直接给了100的奖励，很危险）
        manner_scene_list = cur_env.get("manner_scene_info", None)
        is_junction_scenario = False
        is_main_to_aux_scenario = False
        if manner_scene_list is not None and len(manner_scene_list)>0:
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
        is_junction = is_junction_scenario or is_main_to_aux_scenario
        return is_junction

    def _is_ego_front_dobj(self,ego_x,ego_y,ego_yaw,obj_x,obj_y,start,end):
        dx_world = obj_x - ego_x
        dy_world = obj_y - ego_y
        cos_yaw = np.cos(ego_yaw)
        sin_yaw = np.sin(ego_yaw)

        obj_x_local = dx_world *cos_yaw +dy_world *sin_yaw
        obj_y_local = -dx_world * sin_yaw + dy_world* cos_yaw
        if obj_x_local > start and obj_x_local < end :
            return True
        return False

    def _is_ego_overtake_dobj(self, reference_navi_centerline, pred_polygon, raw_env_t0, obj_id, pre_ts):
        if reference_navi_centerline == None:
            return False
        is_obj_at_front = True
        obj_current_s, obj_current_d = 0, 0
        for future_ts in range(pre_ts, self._pred_t_num - 1, 5):
            ego_future_point = np.mean(pred_polygon[future_ts], axis=0)
            _, dobjs = raw_env_t0["dobjs_full"][self._history_t_num + future_ts]
            trackids = dobjs[:, 9].astype(np.int32)
            cxcy = raw_env_t0["dobjs_polygon"][self._history_t_num + future_ts].mean(axis=1)
            if future_ts == pre_ts:
                for (obj_x, obj_y), tid in zip(cxcy, trackids):
                    if tid == obj_id:
                        obj_current_s, obj_current_d = reference_navi_centerline.xytosl(obj_x, obj_y)
                        ego_current_s, ego_current_d = reference_navi_centerline.xytosl(ego_future_point[0], ego_future_point[1])
                        if abs(obj_current_d - ego_current_d) > 2.0:
                            is_obj_at_front = False
            if not is_obj_at_front:
                break
            ego_future_s, ego_future_d = reference_navi_centerline.xytosl(ego_future_point[0], ego_future_point[1])
            for (obj_x, obj_y), tid in zip(cxcy, trackids):
                if tid == obj_id:
                    obj_future_s, obj_future_d = reference_navi_centerline.xytosl(obj_x, obj_y)
                    if ego_future_s is None or obj_future_s is None:
                        continue
                    obj_current_s = obj_future_s
                    if ego_future_s - obj_future_s > 0.0:
                        return True
                    if ego_future_s > reference_navi_centerline.get_length() or obj_future_s > reference_navi_centerline.get_length():
                        break
            if ego_future_s - obj_current_s > 0.0:
                return True
        return False

    def _check_od_scenario_step(self, line_data1, line_data2, raw_env_t0, cur_env, pre_ts, pred_polygon, spd_limit, ego_vel, yaw, overtake_obj_ids, navi_centerline_reference_lists):
        # 存在可能需要绕行的障碍物，则返回True，不居中；否则返回False，居中
        if len(line_data1) < 2 or len(line_data2) < 2:
            return True  # 车道线不存在，则不居中
        line1 = LineString(line_data1)
        line2 = LineString(line_data2)
        is_success, centerline = self._create_center_line(line1, line2)
        if not is_success:
            return True  # 中心线创建不成功，则不居中

        half_lane_width = line1.distance(line2) / 2.0
        _, dobjs = raw_env_t0["dobjs_full"][self._history_t_num + pre_ts]
        dobjs_polygon = raw_env_t0["dobjs_polygon"][self._history_t_num + pre_ts]

        if len(dobjs) == 0 or len(dobjs_polygon) == 0:
            return False  # 无障碍物，可居中
        dobjs_center = [np.mean(item, axis=0) for item in dobjs_polygon]
        ego_center = np.mean(pred_polygon[pre_ts], axis=0)
        # 只保留自车前方的障碍物，包含车后的部分障碍物。 buffer为0.5
        dobjs_mask = [self._is_ego_front_dobj(ego_center[0], ego_center[1], yaw, dobj[0], dobj[1], start=-1.0, end=100.0) for dobj in dobjs_center]

        dobjs = dobjs[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False

        dobjs_polygon = dobjs_polygon[dobjs_mask]
        # 横向过滤
        dobjs_shapely_polygon = np.array([Polygon(item) for item in dobjs_polygon])
        dobjs_mask = [centerline.distance(dobj) < half_lane_width + 0.5 for dobj in dobjs_shapely_polygon]
        dobjs = dobjs[dobjs_mask]
        dobjs_shapely_polygon = dobjs_shapely_polygon[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False
        # 速度过滤,
        need_avoidance = False
        reference_navi_centerline = None
        max_d = 10
        for reference_line in navi_centerline_reference_lists:
            s, d = reference_line.xytosl(ego_center[0], ego_center[1])
            if d is None:
                continue
            if abs(d) < max_d and s > 0 and s < reference_line.get_length():
                reference_navi_centerline = reference_line
                max_d = abs(d)

        for dobj, dobj_polygon in zip(dobjs, dobjs_shapely_polygon):
            dobj_speed = np.sqrt(dobj[10] ** 2 + dobj[11] ** 2)
            dx = dobj[0] - ego_center[0]
            dy = dobj[1] - ego_center[1]
            dobj_dist_ego = np.sqrt(dx**2 + dy**2)

            # 使用TTC
            relative_speed = ego_vel - dobj_speed
            if relative_speed > 0 and dobj_dist_ego > 0.0:
                ttc = dobj_dist_ego / relative_speed
            else:
                ttc = float("inf")  # 相对速度非正，理论上无碰撞风险
            # 即使TTC很大，但如果距离过近，仍然认为有风险（尤其应对前车急刹）
            is_critically_close = dobj_dist_ego < 5.0
            dobj_type = get_map_cls_to_3cls(dobj[7].astype(int))

            # 针对VRU（弱势道路使用者）的判断
            if dobj_type in ["pedestrian", "bicycle"]:
                if ttc < 5.0 or is_critically_close:  # 对行人/自行车使用更严格的TTC阈值
                    if dobj[9] in overtake_obj_ids:
                        need_avoidance = True
                        break  # VRU威胁高，发现一个即可终止
                    overtake_dobj = self._is_ego_overtake_dobj(reference_navi_centerline, pred_polygon, raw_env_t0, dobj[9], pre_ts)
                    if overtake_dobj:
                        overtake_obj_ids.append(dobj[9])
                        need_avoidance = True
                        break  # VRU威胁高，发现一个即可终止
                    # else:
                    #     need_avoidance = True
                    #     break  # VRU威胁高，发现一个即可终止

            # 针对机动车的判断
            else:
                # 非路口场景下的低速或危险接近判断
                is_low_speed_or_dangerous = (dobj_speed * 3.6 / spd_limit < 0.7) or (ttc < 4.0) or is_critically_close
                is_at_junction = self._is_near_junction(ego_center, line_data1, 5, 40) or self._is_near_junction(
                    ego_center, line_data2, 5, 40
                )
                is_crossing_line = (dobj_polygon.distance(line1) < 0.01) or (dobj_polygon.distance(line2) < 0.01)

                if is_low_speed_or_dangerous:
                    # 在非路口，直接认为需要避让
                    if not is_at_junction:
                        if dobj[9] in overtake_obj_ids:
                            need_avoidance = True
                            break
                        overtake_dobj = self._is_ego_overtake_dobj(reference_navi_centerline, pred_polygon, raw_env_t0, dobj[9], pre_ts)
                        if overtake_dobj:
                            overtake_obj_ids.append(dobj[9])
                            need_avoidance = True
                            break
                        # need_avoidance = True
                        # break
                    # 在路口，额外考虑压线因素：如果压线，则认为风险更高，需要避让
                    elif is_at_junction and is_crossing_line:
                        if dobj[9] in overtake_obj_ids:
                            need_avoidance = True
                            break
                        overtake_dobj = self._is_ego_overtake_dobj(reference_navi_centerline, pred_polygon, raw_env_t0, dobj[9], pre_ts)
                        if overtake_dobj:
                            overtake_obj_ids.append(dobj[9])
                            need_avoidance = True
                            break
                        # need_avoidance = True
                        # break

        return need_avoidance
    def _check_od_scenario(self, line_data1, line_data2, raw_env, ego_polygon, spd_limit, ego_vel):
        # 存在可能需要绕行的障碍物，则返回True，不居中；否则返回False，居中
        if len(line_data1) < 2 or len(line_data2) < 2:
            return True  # 车道线不存在，则不居中

        line1 = LineString(line_data1)
        line2 = LineString(line_data2)
        is_success, centerline = self._create_center_line(line1, line2)
        if not is_success:
            return True  # 中心线创建不成功，则不居中

        half_lane_width = line1.distance(line2) / 2.0
        _, dobjs = raw_env[0]["dobjs_full"][self._history_t_num]
        dobjs_polygon = raw_env[0]["dobjs_polygon"][self._history_t_num]
        if len(dobjs) == 0 or len(dobjs_polygon) == 0:
            return False  # 无障碍物，可居中
        dobjs_center = [np.mean(item, axis=0) for item in dobjs_polygon]
        ego_center = np.mean(ego_polygon, axis=0)
        # 只保留自车前方的障碍物，包含车后的部分障碍物。 buffer为0.5
        dobjs_mask = [dobj[0] - ego_center[0] > -1.0 and dobj[0] - ego_center[0] < 100.0 for dobj in dobjs_center]
        dobjs = dobjs[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False

        dobjs_polygon = dobjs_polygon[dobjs_mask]
        # 横向过滤
        dobjs_shapely_polygon = np.array([Polygon(item) for item in dobjs_polygon])
        dobjs_mask = [centerline.distance(dobj) < half_lane_width for dobj in dobjs_shapely_polygon]
        dobjs = dobjs[dobjs_mask]
        dobjs_shapely_polygon = dobjs_shapely_polygon[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False
        # 速度过滤,
        need_avoidance = False
        for dobj, dobj_polygon in zip(dobjs, dobjs_shapely_polygon):
            dobj_speed = np.sqrt(dobj[10] ** 2 + dobj[11] ** 2)
            dx = dobj[0] - ego_center[0]
            dy = dobj[1] - ego_center[1]
            dobj_dist_ego = np.sqrt(dx**2 + dy**2)

            # 使用TTC
            relative_speed = ego_vel - dobj_speed
            if relative_speed > 0 and dobj_dist_ego > 0.0:
                ttc = dobj_dist_ego / relative_speed
            else:
                ttc = float("inf")  # 相对速度非正，理论上无碰撞风险
            # 即使TTC很大，但如果距离过近，仍然认为有风险（尤其应对前车急刹）
            is_critically_close = dobj_dist_ego < 5.0
            dobj_type = get_map_cls_to_3cls(dobj[7].astype(int))

            # 针对VRU（弱势道路使用者）的判断
            if dobj_type in ["pedestrian", "bicycle"]:
                if ttc < 5.0 or is_critically_close:  # 对行人/自行车使用更严格的TTC阈值
                    need_avoidance = True
                    break  # VRU威胁高，发现一个即可终止

            # 针对机动车的判断
            else:
                # 非路口场景下的低速或危险接近判断
                is_low_speed_or_dangerous = (dobj_speed / spd_limit < 0.7) or (ttc < 4.0) or is_critically_close
                is_at_junction = self._is_near_junction(ego_center, line_data1, 5, 40) or self._is_near_junction(
                    ego_center, line_data2, 5, 40
                )
                is_crossing_line = (dobj_polygon.distance(line1) < 0.01) or (dobj_polygon.distance(line2) < 0.01)

                if is_low_speed_or_dangerous:
                    # 在非路口，直接认为需要避让
                    if not is_at_junction:
                        need_avoidance = True
                        break
                    # 在路口，额外考虑压线因素：如果压线，则认为风险更高，需要避让
                    elif is_at_junction and is_crossing_line:
                        need_avoidance = True
                        break

        return need_avoidance

    def _centralization_get_lines(self, cur_env, ego_point):
        lane_line_search_radius = 10.0      # 车道线搜索半径
        road_edge_search_radius = 8.0      # 路沿搜索半径
        max_distance_threshold = 4.0
        is_park_odd = False

        lane_lines = cur_env["lane_lines"]
        road_edge = cur_env["road_edge"]

        left_width = float('inf')
        right_width = float('inf')
        left_line = None
        right_line = None
        ego_pos = np.array([ego_point.x, ego_point.y])

        # 1. 遍历车道线一次，同时找左右（必须遍历所有，找最近）
        for points, _ in lane_lines:
            if len(points) < 2:
                continue

            # 快速过滤：检查是否有任何点在搜索半径内
            xy_points = np.asarray(points[:, :2])
            distances_to_ego = np.linalg.norm(xy_points - ego_pos, axis=1)
            mask = distances_to_ego < lane_line_search_radius
            xy_points = xy_points[mask]
            if len(xy_points) < 2:
                continue

            signed_dist = calculate_signed_lateral_distance(ego_point, xy_points)
            if signed_dist == float('inf'):
                continue

            if abs(signed_dist) < max_distance_threshold:
                if signed_dist < 0 and abs(signed_dist) < left_width:  # 左侧
                    left_width = abs(signed_dist)
                    left_line = points[:, :2]

                elif signed_dist > 0 and abs(signed_dist) < right_width:  # 右侧
                    right_width = abs(signed_dist)
                    right_line = points[:, :2]

        # 2. 按需遍历路沿（必须遍历所有，找最近）
        need_left = left_width > max_distance_threshold
        need_right = right_width > max_distance_threshold

        if (need_left or need_right) and is_park_odd:
            for edge_points in road_edge:
                if len(edge_points) < 2:
                    continue

                # 快速过滤：检查是否有任何点在搜索半径内
                xy_points = np.asarray(edge_points[:, :2])
                distances_to_ego = np.linalg.norm(xy_points - ego_pos, axis=1)
                mask = distances_to_ego < road_edge_search_radius
                xy_points = xy_points[mask]
                if len(xy_points) < 2:
                    continue

                signed_dist = calculate_signed_lateral_distance(ego_point, xy_points)
                if signed_dist == float('inf'):
                    continue

                # 只在缺失侧更新，但必须遍历所有找最小值
                if abs(signed_dist) < max_distance_threshold:
                    if signed_dist < 0 and need_left and abs(signed_dist) < left_width:  # 左侧缺失
                        left_width = abs(signed_dist)
                        left_line = edge_points[:, :2]
                    elif signed_dist > 0 and need_right and abs(signed_dist) < right_width:  # 右侧缺失
                        right_width = abs(signed_dist)
                        right_line = edge_points[:, :2]

        return left_line, right_line

    def _centralization_check_od_scenario(self, raw_env, pred_polygon, pred_traj, ego_vel, spd_limit):
        cur_env = raw_env[self._history_t_num - 1]
        ego_polygon = pred_polygon[0]
        ego_center = np.mean(ego_polygon, axis=0)
        ego_point = Point(ego_center)
        long_dist_thresh = max(ego_vel * 5.0, 30.0)

        # 计算左右车道
        left_line, right_line = self._centralization_get_lines(cur_env, ego_point)
        if left_line is None or right_line is None:  # 使用预测轨迹来构建一个通道
            centerline = LineString(pred_traj[:, :2])
            half_lane_width = 1.9  # 假设车道宽度为3.8m
        else:
            line1 = LineString(left_line)
            line2 = LineString(right_line)
            half_lane_width = line1.distance(line2) / 2.0
            _, centerline = self._create_center_line(line1, line2)

        _, dobjs = raw_env[0]["dobjs_full"][self._history_t_num]
        dobjs_polygon = raw_env[0]["dobjs_polygon"][self._history_t_num]
        if len(dobjs) == 0 or len(dobjs_polygon) == 0:
            return False  # 无障碍物，可居中
        dobjs_center = [np.mean(item, axis=0) for item in dobjs_polygon]
        # 只保留自车前方的障碍物，包含车后的部分障碍物。 buffer为0.5
        dobjs_mask = [dobj[0] - ego_center[0] > -1.0 and dobj[0] - ego_center[0] < long_dist_thresh for dobj in dobjs_center]
        dobjs = dobjs[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False

        dobjs_polygon = dobjs_polygon[dobjs_mask]
        # 横向过滤
        dobjs_shapely_polygon = np.array([Polygon(item) for item in dobjs_polygon])
        dobjs_mask: list[Any] = [centerline.distance(dobj) < half_lane_width for dobj in dobjs_shapely_polygon]
        dobjs = dobjs[dobjs_mask]
        dobjs_shapely_polygon = dobjs_shapely_polygon[dobjs_mask]
        if len(dobjs) < 1:  # 未筛选到障碍物，居中
            return False
        # 速度过滤,
        need_avoidance = False
        for dobj, dobj_polygon in zip(dobjs, dobjs_shapely_polygon):
            dobj_vx, dobj_vy = dobj[10], dobj[11]
            dobj_speed = np.sqrt(dobj_vx**2 + dobj_vy**2)
            dx = dobj[0] - ego_center[0]
            dy = dobj[1] - ego_center[1]
            dobj_dist_ego = np.sqrt(dx**2 + dy**2)
            # 识别逆向来车
            # 1. 明显的逆向行驶（最危险）
            if dobj_vx < -1.5:
                need_avoidance = True
                break
            # 2. 横向接近自车且距离较近
            if (abs(dobj_vy) > 2.5 and
                dy * dobj_vy < 0 and    # 横向运动指向自车
                dobj_dist_ego < 25.0 and     # 25米内
                dx > -5.0):             # 在前方区域
                need_avoidance = True
                break

            # 使用TTC
            relative_speed = ego_vel - dobj_speed
            if relative_speed > 0 and dobj_dist_ego > 0.0:
                ttc = dobj_dist_ego / relative_speed
            else:
                ttc = float("inf")  # 相对速度非正，理论上无碰撞风险
            # 即使TTC很大，但如果距离过近，仍然认为有风险（尤其应对前车急刹）
            is_critically_close = dobj_dist_ego < 5.0
            dobj_type = get_map_cls_to_3cls(dobj[7].astype(int))

            # 针对VRU的判断
            if dobj_type in ["pedestrian", "bicycle"]:
                if ttc < 5.0 or is_critically_close:  # 对行人/自行车使用更严格的TTC阈值
                    need_avoidance = True
                    break  # VRU威胁高，发现一个即可终止

            # 针对机动车的判断
            else:
                if (dobj_speed / spd_limit < 0.7) or (ttc < 4.0) or is_critically_close:
                    need_avoidance = True

        return need_avoidance

    def compute_centralization_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, Any]:
        """ "计算居中reward"""
        centralization_reward = np.zeros_like(reward_results["reward"])
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        centralization_reward_weight = 1.0
        spd_limit_list = policy_update_output["spd_limit_list"]
        spd_limit = spd_limit_list[0] / 3.6
        pred_traj = get_xyyaw_from_polygon(pred_polygon)
        pred_dxdy = pred_traj[1:][:, :2] - pred_traj[:-1][:, :2]
        pred_vxvy = pred_dxdy * self.fps  # m/s
        pred_vels = np.linalg.norm(pred_vxvy, axis=1)
        ego_vel = pred_vels[0]
        spd_limit_ratio = max(min(ego_vel / spd_limit, 1.0), 0.0)
        centralization_reward_weight = centralization_reward_weight * spd_limit_ratio

        # OD场景隔离,自车所在车道前方有低速障碍物时，不居中
        is_od_scene = self._centralization_check_od_scenario(raw_env, pred_polygon, pred_traj, ego_vel, spd_limit)

        # 待转区行为隔离和偏航隔离
        is_yawing = np.any(reward_results["navi_lane_reward"] < 0)
        is_traffic_light = np.any(reward_results["traffic_light_reward"] != 0)
        scenario_id=policy_update_output['scenario_id']
        is_slow_scene_all= is_slow_scene(scenario_id)
        is_campus = is_campus_scene(scenario_id)
        cur_env = raw_env[self._history_t_num - 1]
        turn_signal = cur_env.get("lane_change_signal", 0)
        is_turn_signal = turn_signal != 0
        # 计算居中reward
        if not is_od_scene and not is_yawing and not is_traffic_light and not is_slow_scene_all and not is_campus and not is_turn_signal:
            for pred_ts in range(0, self._pred_t_num):
                if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                    break
                # 逆向车道隔离
                if reward_results["wrong_way_reward"][pred_ts + 1] < 0.0:
                    continue
                pred_polygon_at_t = pred_polygon[pred_ts]
                cur_env = raw_env[pred_ts + self._history_t_num]
                central_reward = self.reward_funcs["centralization_reward"](
                    pred_polygon_at_t, cur_env, centralization_reward_weight
                )
                centralization_reward[pred_ts + 1] = central_reward

        reward_results.update(
            {
                "centralization_reward": centralization_reward,
            }
        )

        return reward_results

    def compute_toggle_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, Any]:
        """计算拨杆reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        toggle_reward = np.zeros_like(reward_results["reward"])

        last_lateral_distance = 10
        last_ego_center_pt = policy_update_output["ego_polygon"][self._history_t_num - 1].mean(axis=0).reshape(1, 2)
        prev_intersect_status = False

        cur_env = raw_env[self._history_t_num - 1]
        # 获取np_plus_lcc_status (0: 未激活, 1: 激活)
        np_plus_lcc_status = cur_env.get("np_plus_lcc_status", 0)
        navi_lane_reward_penalty_deviation = None
        navi_reward_penalty_deviation = None
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            pred_polygon_at_t = pred_polygon[pred_ts]
            cur_env = raw_env[pred_ts + self._history_t_num]
            # 获取拨杆信号 (0: 无信号, 1: 左转, 2: 右转)
            turn_signal = cur_env.get("lane_change_signal", 0)
            (
                np_lcc_penalty,
                turn_signal_reward,
                last_lateral_distance,
                last_ego_center_pt,
                new_intersect_status,
            ) = self.reward_funcs["toggle_reward"](
                pred_polygon_at_t,
                cur_env,
                pred_ts,
                turn_signal,
                last_lateral_distance,
                np_plus_lcc_status,
                last_ego_center_pt,
            )

            if "navi_lane_reward" in reward_results:
                navi_lane_reward = reward_results["navi_lane_reward"][pred_ts + 1]
                if navi_lane_reward < -5.0:
                    navi_lane_reward_penalty_deviation = True

            if "navi_reward" in reward_results:
                navi_reward = reward_results["navi_reward"][pred_ts + 1]
                if navi_reward < 0.0:
                    navi_reward_penalty_deviation = True
            if navi_lane_reward_penalty_deviation or navi_reward_penalty_deviation:
                np_lcc_penalty = 0

            prev_intersect_status = new_intersect_status
            toggle_reward[pred_ts + 1] = np_lcc_penalty + turn_signal_reward

        reward_results.update(
            {
                "toggle_reward": toggle_reward,
            }
        )
        return reward_results

    def compute_progress_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算通行效率reward."""
        reward_results.update(
            {
                "progress_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "progress_reward_spd": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "progress_reward_sub_path": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        ego_polygon = policy_update_output["ego_polygon"]
        pred_polygon = ego_polygon[self._history_t_num :]
        gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        spd_limit_list = policy_update_output["spd_limit_list"]
        speeds = policy_update_output["speeds"]
        assert "traffic_light_reward" in reward_results

        is_highway_scene = raw_env[0]["priority_road_class"] == 0 or raw_env[0]["priority_road_class"] == 1

        if is_highway_scene:
            max_speed = 130.0
            if (
                2 not in reward_results["result"]
                # and raw_env[self._history_t_num]["navi_infos"]["sub_path_main_path_points"] is not None
                and np.all(reward_results["traffic_light_reward"] == 0)
            ):
                # 最高速度相关
                for i in range(min(len(speeds), len(reward_results["progress_reward"]))):
                    weight = max_speed / 100.0
                    vel = speeds[i]
                    if vel > max_speed:
                        vel = max_speed
                    # spd_limit = spd_limit_list[i] if spd_limit_list[i] > 0 else gt_max_speed
                    # r1 = self.reward_funcs["progress_reward"](vel, spd_limit, highway_scenario_flag)
                    reward_results["progress_reward"][i + 1] = (vel / 100.0) * weight
                    reward_results["progress_reward_spd"][i + 1] = vel / 100.0
            return reward_results

        spd_limit_mode = spd_limit_mode = statistics.mode(spd_limit_list)
        is_use= ((reward_results["danger_lc_reward"] > -0.1)
                & (reward_results["navi_reward"] > -0.1)
                &(reward_results["ttc_reward"] > -0.1))
        city_use_progress  = spd_limit_mode > 60.0 and np.all(is_use)

        if city_use_progress: #限速大于60km/h 且 无危险变道，导航偏离，碰撞风险
            max_speed = 130.0
            if (
                2 not in reward_results["result"]
                and np.all(reward_results["traffic_light_reward"] == 0)
            ):
                for i in range(min(len(speeds), len(reward_results["progress_reward"]))):
                    weight = max_speed / 100.0 * 3.0
                    vel = speeds[i]
                    if vel > max_speed:
                        vel = max_speed
                    reward_results["progress_reward"][i + 1] = (vel / 100.0) * weight
                    reward_results["progress_reward_spd"][i + 1] = vel / 100.0
        else:
            if (
                2 not in reward_results["result"]
                and raw_env[0]["navi_infos"]["sub_path_main_path_points"] is not None
                and np.all(reward_results["traffic_light_reward"] > -4)
            ):
                pred_traj = get_xyyaw_from_polygon(pred_polygon)
                pred_dxdy = pred_traj[1:][:, :2] - pred_traj[:-1][:, :2]
                pred_vxvy = pred_dxdy * self.fps * 3.6  # m/s -> km/h
                pred_vels = np.linalg.norm(pred_vxvy, axis=1)


                # 速度相关
                progress_reward_max = 0.1
                if spd_limit_mode >= 60:
                    progress_reward_max = 0.3
                for i in range(min(len(pred_vels), len(reward_results["progress_reward"]))):
                    vel = pred_vels[i]
                    spd_limit = spd_limit_list[i]
                    r1 = self.reward_funcs["progress_reward"](vel, spd_limit, reward_max=progress_reward_max)
                    reward_results["progress_reward"][i + 1] = r1
                    reward_results["progress_reward_spd"][i + 1] = r1

                # 计算最终停下的位置在navi_path中的位置.
                subpath = raw_env[0]["navi_infos"]["sub_path_main_path_points"][:, :2]
                # 截断subpath, 自车车身后的subpath扔掉, 超过150m的扔掉
                subpath = interpolate_points(subpath, 5)
                subpath = cut_sub_path(subpath, 0, 800)  # n x 2
                subpath_cumdistance = [
                    0,
                ] + list(np.cumsum(np.linalg.norm(subpath[1:] - subpath[:-1], axis=1)))

                pred_polygon_last = pred_polygon[-1]
                pred_polygon_last_center = pred_polygon_last.mean(axis=0)
                sub_path_distance = np.linalg.norm(subpath - pred_polygon_last_center, axis=1)
                sub_path_min_index = np.argmin(sub_path_distance)
                sub_path_min_distance = subpath_cumdistance[sub_path_min_index]

                spd_limit_limit_valid = [s for s in spd_limit_list if s > 0]
                if spd_limit_limit_valid:
                    spd_limit_limit_mean = np.mean(spd_limit_limit_valid)
                else:
                    spd_limit_limit_mean = 80
                max_possible_distance = spd_limit_limit_mean / 3.6 * len(pred_polygon) / self.fps

                progress = min(1.0, sub_path_min_distance / max_possible_distance) * 50
                if spd_limit_mode >= 60:
                    progress *= 3.0
                mask = reward_results["dones"][:-1] == 0
                speeds_sum = np.sum(speeds * mask)
                for i in range(len(speeds)):
                    if "dones" in reward_results and reward_results["dones"][i] != 0:
                        break
                    r = progress * speeds[i] / speeds_sum if speeds_sum > 0 else 0
                    reward_results["progress_reward"][i + 1] += r
                    reward_results["progress_reward_sub_path"][i + 1] = r

                # 统计里程 速度
                gt_traj = get_xyyaw_from_polygon(gt_polygon)
                gt_dxdy = gt_traj[1:][:, :2] - gt_traj[:-1][:, :2]
                gt_vxvy = gt_dxdy * self.fps * 3.6  # m/s -> km/h
                gt_vels = np.linalg.norm(gt_vxvy, axis=1)

                reward_results["pred_mileage"] = np.linalg.norm(pred_traj[1:] - pred_traj[:-1], axis=1).sum()
                reward_results["gt_mileage"] = np.linalg.norm(gt_traj[1:] - gt_traj[:-1], axis=1).sum()
                reward_results["pred_gt_mileage_ratio"] = (
                    (reward_results["pred_mileage"] / reward_results["gt_mileage"])
                    if reward_results["gt_mileage"] > 1e-4
                    else 0
                )
                reward_results["pred_gt_speed_ratio"] = (pred_vels.mean() / gt_vels.mean()) if gt_vels.mean() > 1e-4 else 0

        indices_encounter = np.where(reward_results["encounter_ttc_reward"] < 0)[0]
        if len(indices_encounter) > 0:
            first_idx_encounter = indices_encounter[0]
            reward_results["progress_reward"][first_idx_encounter:] = 0
            reward_results["progress_reward_sub_path"][first_idx_encounter:] = 0
            reward_results["progress_reward_spd"][first_idx_encounter:] = 0

        turn_left_cross_idx = np.where(reward_results["turn_left_cross_flag"])[0]
        if len(turn_left_cross_idx) > 0:
            # 1. first_idx 之后的三类 reward 除以10
            first_idx = turn_left_cross_idx[0]
            reward_results["progress_reward"][first_idx:] /= 10
            reward_results["progress_reward_sub_path"][first_idx:] /= 10
            reward_results["progress_reward_spd"][first_idx:] /= 10
            # 2. 将每个 turn_left_cross_idx 前后延续5个元素置 0
            extend = 5
            n = len(reward_results["progress_reward"])
            for idx in turn_left_cross_idx:
                start = max(idx - extend, 0)
                end = min(idx + extend + 1, n)
                reward_results["progress_reward"][start:end] = 0
                reward_results["progress_reward_sub_path"][start:end] = 0
                reward_results["progress_reward_spd"][start:end] = 0

        vru_caution_idx = np.where(reward_results["vru_caution"])[0]
        if len(vru_caution_idx) > 0:
            first_idx = vru_caution_idx[0]
            begin_idx = max(0, first_idx)
            reward_results["progress_reward"][begin_idx:] /= 10
            reward_results["progress_reward_sub_path"][begin_idx:] /= 10
            reward_results["progress_reward_spd"][begin_idx:] /= 10
            vru_cross_checksize = 5
            for idx in vru_caution_idx:
                end_inx = min(idx + vru_cross_checksize + 1, len(reward_results["progress_reward"]))
                begin_idx = max(0, idx - vru_cross_checksize)
                reward_results["progress_reward"][begin_idx:end_inx] = 0
                reward_results["progress_reward_sub_path"][begin_idx:end_inx] = 0
                reward_results["progress_reward_spd"][begin_idx:end_inx] = 0

        logic_1 = np.logical_and(np.array([0.0] + speeds) > 5, reward_results["dis"] < 0.2)
        logic_2 =np.logical_and(np.array([0.0] + speeds) > 10, reward_results["dis"] < 0.3)
        logic_3 =np.logical_and(np.array([0.0] + speeds) > 15, reward_results["dis"] < 0.4)
        logic_4 =np.logical_and(np.array([0.0] + speeds) > 20, reward_results["dis"] < 0.6)
        indices = np.where(logic_1 | logic_2 | logic_3 | logic_4)[0]
        if len(indices) > 0:
            first_idx = indices[0]
            reward_results["progress_reward"][first_idx:] = 0
            reward_results["progress_reward_sub_path"][first_idx:] = 0
            reward_results["progress_reward_spd"][first_idx:] = 0

        # 根据距离增加安全补偿
        distance_new = deepcopy(reward_results["dis"])
        distance_new[distance_new > 1.0] = 1.0
        distance_new[distance_new < 0.1] = 1.0
        reward_results["progress_reward_sub_path"] /= distance_new
        reward_results["progress_reward"] = reward_results["progress_reward_sub_path"] + reward_results["progress_reward_spd"]

        return reward_results

    def compute_scenario_exclusion_flags(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        reward_results.update({"scene_exclusion_flag": np.zeros(self._pred_t_num + 1, dtype=np.bool_)})
        gt_egomotion_velocity = np.array(raw_env[self._history_t_num]["gt_vehspeeds"])
        gt_max_speed = min(90, gt_egomotion_velocity.max())
        ego_polygon = policy_update_output["ego_polygon"]
        pred_polygon = ego_polygon[self._history_t_num :]
        spd_limit_list = policy_update_output["spd_limit_list"]
        speeds = policy_update_output["speeds"]

        for pred_ts in range(0, self._pred_t_num):
            reward_results["scene_exclusion_flag"][pred_ts + 1]= target_progress_scenario_exclude_condition(
                    pred_polygon, raw_env, pred_ts, pred_ts + self._history_t_num, speeds, spd_limit_list, gt_max_speed
            )
        return reward_results

    def compute_speed_limit_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算超速惩罚."""
        reward_results.update(
            {
                "speed_limit_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "overspeed": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "overspeed_penalty": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        spd_limit_list = policy_update_output["spd_limit_list"]
        speeds = policy_update_output["speeds"]

        gt_egomotion_velocity = np.array(raw_env[self._history_t_num]["gt_vehspeeds"])
        gt_max_speed = min(90, gt_egomotion_velocity.max())

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            speed = speeds[pred_ts]
            spd_limit = spd_limit_list[pred_ts]
            if spd_limit > 0:
                overspeed_flg, penalty = self.reward_funcs["speed_limit_reward"](speed, spd_limit, self.fine_tune)

                reward_results["overspeed"][pred_ts + 1] = overspeed_flg
                reward_results["overspeed_penalty"][pred_ts + 1] = penalty

        reward_results["speed_limit_reward"] = -reward_results["overspeed_penalty"]

        return reward_results

    def compute_velocity_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算速度奖励."""
        reward_results.update(
            {
                "velocity_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "velocity_effective_flg": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        ego_polygon = policy_update_output["ego_polygon"]
        pred_polygon = ego_polygon[self._history_t_num :]
        spd_limit_list = policy_update_output["spd_limit_list"]
        speeds = policy_update_output["speeds"]
        assert "traffic_light_reward" in reward_results
        assert "ttc_reward" in reward_results
        assert "collision_reward" in reward_results
        assert "is_ego_intersect_solid_line" in reward_results

        gt_egomotion_velocity = np.array(raw_env[self._history_t_num]["gt_vehspeeds"])
        gt_max_speed = min(90, gt_egomotion_velocity.max())

        if (
            2 not in reward_results["result"]
            and np.all(reward_results["traffic_light_reward"] > -4)
            and np.all(reward_results["collision_reward"] > -10)  # no collision penalty
            and np.all(reward_results["virtual_wall_reward"] > -0.01)
            and np.all(reward_results["wrong_way_reward"] > -0.01)
            and np.all(reward_results["cross_solid_line_reward"] > -0.01)
            and np.all(np.abs(reward_results["navi_lane_reward"]) < 0.01)
        ):
            # 速度相关
            for pred_ts in range(self._pred_t_num):
                reward_check_frame_num = 25
                reward_check_min_idx = max(0, pred_ts + 1)
                reward_check_max_idx = min(self._pred_t_num - 1, pred_ts + 1 + reward_check_frame_num)
                if (
                    np.any(reward_results["ttc_reward"][reward_check_min_idx:reward_check_max_idx] < -0.01)
                    or reward_results["min_distance_reward"][pred_ts + 1] < -0.01
                ):
                    continue

                reward_check_frame_num = 10
                reward_check_min_idx = max(0, pred_ts + 1)
                reward_check_max_idx = min(self._pred_t_num - 1, pred_ts + 1 + reward_check_frame_num)
                if np.any(reward_results["is_ego_intersect_solid_line"][reward_check_min_idx:reward_check_max_idx]):
                    continue

                if target_progress_scenario_exclude_condition(
                    pred_polygon, raw_env, pred_ts, pred_ts + self._history_t_num, speeds, spd_limit_list, gt_max_speed
                ):
                    continue

                speed = speeds[pred_ts]
                spd_limit = 1.1 * (spd_limit_list[pred_ts] if spd_limit_list[pred_ts] > 0 else gt_max_speed)
                vel = min(speed, 90) - min(policy_update_output["gt_mean_speed_batch"], 90)
                velocity_reward, speed_limit_penalty = self.reward_funcs["velocity_reward"](vel, spd_limit)

                reward_results["velocity_reward"][pred_ts + 1] = velocity_reward
                if speed_limit_penalty > 1e-4 and "speed_limit_reward" in reward_results:
                    reward_results["speed_limit_reward"][pred_ts + 1] = -speed_limit_penalty

                reward_results["velocity_effective_flg"][pred_ts + 1] = 1.0

        return reward_results

    def compute_slow_follow_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算超速惩罚."""
        reward_results.update( { "slow_follow_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64)})

        is_checker_follow_slow = (ScenarioEnum.kFollowSlow == policy_update_output["scenario_id"]
                                  and 'scenario_info' in policy_update_output
                                  and 'follow_slow_segments' in policy_update_output['scenario_info'])

        is_human_follow_slow = ScenarioEnum.kFollowSlowByHuman == policy_update_output["scenario_id"]

        if not is_human_follow_slow and not is_checker_follow_slow:
            return reward_results

        spd_limit_list = np.array(policy_update_output["spd_limit_list"],dtype=float)
        speeds_kph = np.array(policy_update_output["speeds"], dtype=float)


        follow_slow_segs=policy_update_output['scenario_info']['follow_slow_segments'] if is_checker_follow_slow else []
        safe_drive_mask = policy_update_output['scenario_info'].get('safe_drive_mask',None)
        reward_results["slow_follow_reward"][1:] = -self.reward_funcs["slow_follow_reward"](
            speeds_kph=speeds_kph,
            speed_limits_kph=spd_limit_list,
            scenario_id=policy_update_output['scenario_id'],
            reward_results=reward_results,
            is_checker_follow_slow=is_checker_follow_slow,
            is_human_follow_slow=is_human_follow_slow,
            follow_slow_segs=follow_slow_segs,
            safe_drive_mask=safe_drive_mask,
            raw_env_list=raw_env,
        )


        # def should_to_go(t:int, seq_len:int):
        #     #safe: ttc_reward,traffic_light_reward,min_distance_reward,speed_limit_reward
        #     is_safe = reward_results["ttc_reward"][t] > -0.1 and reward_results["traffic_light_reward"][t] > -0.1 and reward_results["min_distance_reward"][t] > -0.1 and reward_results["speed_limit_reward"][t] >-0.1
        #     #logic cross_solid_line_reward dangerous_lc_penalty  navi_lane_reward  navi_reward wrong_way_reward
        #     is_logic = reward_results["cross_solid_line_reward"][t] > -0.1 and reward_results["dangerous_lc_penalty"][t] > -0.1 and reward_results["navi_lane_reward"][t] > -0.1 and reward_results["navi_reward"][t] >-0.1 and reward_results["wrong_way_reward"][t] >-0.1
        #     #progress_reward
        #     is_no_progress_reward = reward_results["progress_reward"][t] <0.02
        #     #is not x acc
        #     is_not_ax=reward_results["ax"][t] <0.2
        #     # no collision
        #     is_no_collision = ~np.any(reward_results["collision_reward"] <= -10)
        #     future_t = t+15 if t+15<seq_len else seq_len
        #     if_no_has_future_ttc = True
        #     if np.any(reward_results["ttc_reward"][t:future_t] <-0.1):
        #         if_no_has_future_ttc = False
        #
        #
        #     return (is_safe and is_logic and is_no_progress_reward and is_not_ax and is_no_collision and if_no_has_future_ttc)

        break_idx = -1
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break_idx=pred_ts+1
                break
        if break_idx>=0:
            reward_results["slow_follow_reward"][break_idx:]=0
        return reward_results

    def compute_acc_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算加速度reward."""
        reward_results.update(
            {
                "acc_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "ax": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "ax_penalty": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "ay_penalty": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "ay": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "dec_upbound": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                # "jx": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                # "jy": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        dx_list = policy_update_output["dx_list"]
        dy_list = policy_update_output["dy_list"]
        dyaw_list = policy_update_output["dyaw_list"]
        ego_polygon = policy_update_output["ego_polygon"]
        eval_type = policy_update_output["eval_type"] if "eval_type" in policy_update_output.keys() else None
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            ax, ax_penalty, ay, ay_penalty, comfort_dec_upbound, jx, jy = self.reward_funcs["acc_reward"](
                dx_list,
                dy_list,
                dyaw_list,
                pred_ts,
                reward_results["ax"][pred_ts],
                ego_polygon,
                self._history_t_num + pred_ts,
            )
            # 可视化聚类轨迹时，可能和场景速度不匹配，所以忽略掉前4个轨迹点
            if eval_type == "eval_gt_pred" and pred_ts <= 4:
                ax_penalty = 0
                ay_penalty = 0

            reward_results["ax"][pred_ts + 1] = ax
            reward_results["ay"][pred_ts + 1] = ay
            # reward_results["jx"][pred_ts + 1] = jx
            # reward_results["jy"][pred_ts + 1] = jy
            reward_results["ax_penalty"][pred_ts + 1] = ax_penalty
            reward_results["ay_penalty"][pred_ts + 1] = ay_penalty
            reward_results["dec_upbound"][pred_ts + 1] = -comfort_dec_upbound

        reward_results["acc_reward"] = -reward_results["ax_penalty"] - reward_results["ay_penalty"]

        return reward_results

    def compute_jerk_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算jerk reward"""
        reward_results.update(
            {
                "jerk_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "jx_penalty": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "jy_penalty": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "jx": np.zeros(self._pred_t_num + 1, dtype=np.float64),
                "jy": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        dx_list = policy_update_output["dx_list"]
        dy_list = policy_update_output["dy_list"]
        dyaw_list = policy_update_output["dyaw_list"]
        ego_polygon = policy_update_output["ego_polygon"]
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            jx_penalty, jy_penalty, jx, jy = self.reward_funcs["jerk_reward"](
                dx_list,
                dy_list,
                dyaw_list,
                pred_ts,
                reward_results["ax"][pred_ts],
                ego_polygon,
                self._history_t_num + pred_ts,
            )
            reward_results["jx"][pred_ts + 1] = jx
            reward_results["jy"][pred_ts + 1] = jy
            reward_results["jx_penalty"][pred_ts + 1] = jx_penalty
            reward_results["jy_penalty"][pred_ts + 1] = jy_penalty

        reward_results["jerk_reward"] = -reward_results["jx_penalty"] - reward_results["jy_penalty"]

        return reward_results

    def compyte_etc_take_reward(self, etc_dis, ego_polygon, toll_info, raw_env, history_size, pred_ts, speed, stop_count):
        if etc_dis > 5.0 or etc_dis < -50.0 :
            return 0.0, stop_count

        # 1. 找到最近的SOD,开则默认远,关则取距离
        dobj, dobj_next = get_xyyaw_from_polygon(ego_polygon[history_size + pred_ts-1 : history_size + pred_ts + 1])
        toll_sods = toll_info['sods']
        dis_to_sod = 1000.0
        if toll_sods is not None:
            for id, sod in enumerate(toll_sods) :
                sod_pose = sod["pos"]
                sod_status = sod["status"]
                sod_id = sod["id"]
                # 如果是关,则取正常距离
                if sod_pose is not None and sod_status in [10001, 10002] :
                    # TODO: use local sod pose calc distance
                    dis = np.linalg.norm(sod_pose - dobj[:2]) * np.sign(sod_pose[0] - dobj[0])
                    if dis >= 0.0 :
                        dis_to_sod = min(dis_to_sod, dis)

        #2. find nearest centerline od
        dis_to_od = 1000.0
        dobjs_polygon = raw_env[0]["dobjs_polygon"]
        toll_cenline = toll_info["cenline"]
        cur_ts = history_size + pred_ts - 1
        if len(dobjs_polygon) > cur_ts and toll_cenline is not None and len(toll_cenline)>2 :
            cur_ods = dobjs_polygon[cur_ts]
            state_polygon_at_t = ego_polygon[cur_ts]
            front_center = np.mean(state_polygon_at_t[[1, 2]], axis=0, keepdims=True)
            front_center = np.array(front_center).squeeze().tolist()
            min_dis_od_id = -1
            min_dis_od_dis = 10e6
            for id, obj in enumerate(cur_ods):
                if obj is not None and len(obj) > 3:
                    cxcy = obj.mean(axis=0)
                    if cxcy[0] > 0 :
                        od_cen_dis = point_to_centerline_dis(cxcy, toll_cenline)
                        ego_to_car_dis = cxcy[0] - front_center[0]
                        polygon_ego = Polygon(state_polygon_at_t)
                        polygon_front = Polygon(obj)
                        distance = polygon_ego.distance(polygon_front)
                        if od_cen_dis < 1.5 and ego_to_car_dis > 0 :
                            if min_dis_od_dis > distance :
                                min_dis_od_id = id
                                min_dis_od_dis = distance
            if min_dis_od_id >=0 :
                dis_to_od = min_dis_od_dis

        # 3. 距离过远,判断距离加惩罚
        take_off_reward = 0.0
        if dis_to_sod > 5 and dis_to_od > 5 or (dis_to_sod < 0.0 and dis_to_od > 5.0):
            if (speed < 4.0) :
                stop_count += 1
                # print(f"Stop t: {pred_ts}, vel: {speed:.3}, count: {stop_count}")
                # stop time > 5s = 4*5
                if stop_count > 16:
                    take_off_reward = 5.0
                    # print(f"Stop true, time: {stop_count}")

            if (speed > 5.0) :
                stop_count = 0
                take_off_reward = 0.0

        # print(f"t: {pred_ts}, dis_sod: {dis_to_sod:.3}, od: {dis_to_od:.3}, speed: {speed:.3}, t_r: {take_off_reward:.3}")
        return take_off_reward, stop_count

    def compute_etc_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算ETC ec reward."""
        reward_results.update(
            {
                "etc_take_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        ego_polygon = policy_update_output["ego_polygon"]
        speeds = policy_update_output["speeds"]
        stop_count = 0

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            cur_ts = pred_ts + self._history_t_num
            toll_info = raw_env[cur_ts]["toll_infos"]
            speed_reward = 0.0
            take_off_reward = 0.0
            if toll_info is None or len(toll_info) < 1:
                reward_results["etc_take_reward"][pred_ts + 1] = -take_off_reward
                continue

            etc_dis = toll_info["etc_dis"]
            if etc_dis is None or etc_dis > 40.0 or etc_dis < -50.0:
                reward_results["etc_take_reward"][pred_ts + 1] = -take_off_reward
                continue

            dobj, dobj_next = get_xyyaw_from_polygon(
                ego_polygon[self._history_t_num + pred_ts - 1 : self._history_t_num + pred_ts + 1]
            )
            speed = speeds[pred_ts]

            # calc center dis
            center_line = toll_info["cenline"]
            if center_line is not None and len(center_line) > 2:
                ego_cenline = transform_centerline_matrix(dobj[:3], center_line)
                front_dis = ego_cenline[0][0]
                back_dis = ego_cenline[-1][0]
                if front_dis > 0 and back_dis > 0:
                    etc_dis = front_dis
                elif front_dis <= 0.0 and back_dis >= 0.0:
                    etc_dis = 0.0
                else:
                    etc_dis = back_dis
                # print(f"t: {pred_ts}, front dis: {front_dis}, back dis: {back_dis}, etc_dis: {etc_dis}")

            # 2. calc acc reward
            take_off_reward, stop_count = self.compyte_etc_take_reward(
                etc_dis, ego_polygon, toll_info, raw_env, self._history_t_num, pred_ts, speed, stop_count
            )
            reward_results["etc_take_reward"][pred_ts + 1] = -take_off_reward
        return reward_results

    def compute_etc_speed_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算ETC etc speed reward."""
        reward_results.update(
            {
                "etc_speed_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        ego_polygon = policy_update_output["ego_polygon"]
        speeds = policy_update_output["speeds"]

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            cur_ts = pred_ts + self._history_t_num
            toll_info = raw_env[cur_ts]["toll_infos"]
            speed_reward = 0.0
            if toll_info is None or len(toll_info) < 1:
                reward_results["etc_speed_reward"][pred_ts + 1] = -speed_reward
                continue

            etc_dis = toll_info["etc_dis"]
            if etc_dis is None or etc_dis > 40.0 or etc_dis < -50.0:
                reward_results["etc_speed_reward"][pred_ts + 1] = -speed_reward
                continue

            dobj, dobj_next = get_xyyaw_from_polygon(
                ego_polygon[self._history_t_num + pred_ts - 1 : self._history_t_num + pred_ts + 1]
            )
            state_polygon_at_t = ego_polygon[self._history_t_num + pred_ts - 1]
            front_center = np.mean(state_polygon_at_t[[1, 2]], axis=0, keepdims=True)
            front_center = np.array(front_center).squeeze().tolist()
            speed = speeds[pred_ts]

            # calc center dis
            center_line = toll_info["cenline"]
            if center_line is not None and len(center_line) > 2:
                ego_cenline = transform_centerline_matrix(dobj[:3], center_line)
                front_dis = ego_cenline[0][0]
                back_dis = ego_cenline[-1][0]
                # print(f"front dis: {front_dis}, back dis: {back_dis}")
                if front_dis > 0 and back_dis > 0:
                    etc_dis = front_dis
                elif front_dis <= 0.0 and back_dis >= 0.0:
                    etc_dis = 0.0
                else:
                    etc_dis = back_dis

            # if in_etc_scene
            nearest_sod_id = -1
            dis_to_sod = 1000
            toll_sods = toll_info["sods"]
            near_sod = {"pos": None, "status": -1, "id":-1}
            if toll_sods is not None:
                for id, sod in enumerate(toll_sods) :
                    sod_pose = sod["pos"]
                    sod_status = sod["status"]
                    sod_id = sod["id"]
                    # 如果是关,则取正常距离
                    if sod_pose is not None and sod_status in [10001, 10002] :
                        # TODO: use local sod pose calc distance
                        dis = np.linalg.norm(sod_pose - dobj[:2]) * np.sign(sod_pose[0] - dobj[0])
                        if dis >= 0.0 and dis_to_sod > dis :
                            dis_to_sod = dis
                            nearest_sod_id = id
                if nearest_sod_id >= 0:
                    near_sod = toll_sods[nearest_sod_id]
            speed_reward = self.reward_funcs["etc_speed_reward"](etc_dis, speed, front_center, near_sod["pos"], near_sod["status"])
            reward_results["etc_speed_reward"][pred_ts + 1] = -speed_reward
        return reward_results

    def compute_etc_dis_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算ETC etc speed reward."""
        reward_results.update(
            {
                "etc_mindistance_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        min_car_dis_list = []
        ego_polygon = policy_update_output["ego_polygon"]

        # 1. collect min_car_distance list
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                min_car_dis_list.append(5.0)
                break
            cur_ts = pred_ts + self._history_t_num - 1
            ego_pose = ego_polygon[cur_ts].mean(axis=0)
            toll_info = raw_env[cur_ts]["toll_infos"]
            if toll_info is None or len(toll_info) < 1:
                min_car_dis_list.append(5.0)
                continue

            etc_dis = toll_info["etc_dis"]
            if etc_dis is None or etc_dis > 40.0 or etc_dis < -20.0:
                min_car_dis_list.append(5.0)
                continue

            # if in_etc_scene
            centerline = toll_info.get("cenline")
            if centerline is None or len(centerline) < 2:
                min_car_dis_list.append(5.0)
                continue

            # find nearest centerline od
            dobjs_polygon = raw_env[0]["dobjs_polygon"]
            if len(dobjs_polygon) < cur_ts :
                min_car_dis_list.append(5.0)
                continue
            cur_ods = dobjs_polygon[cur_ts]

            center_ods = []
            for id, obj in enumerate(cur_ods):
                if obj is not None and len(obj) > 3:
                    cxcy = obj.mean(axis=0)
                    if cxcy[0] > 0:
                        od_cen_dis = point_to_centerline_dis(cxcy, centerline)
                        if od_cen_dis < 1.5 and (cxcy[0] - ego_pose[0]) > 0 :
                            center_ods.append(obj)

            # polygons = np.concatenate(center_ods, axis=0)
            shape_polygons = MultiPolygon([Polygon(p) for p in center_ods])
            ego_poly = Polygon(ego_polygon[cur_ts])
            line_distance = get_distance(ego_poly, shape_polygons)
            min_car_dis_list.append(line_distance)
            # print(f"t: {pred_ts}, dis: {line_distance:.3}")

        # 2. calc distance reward
        reward_results["etc_min_dis"] = min_car_dis_list
        etc_distance_reward = np.zeros(self._pred_t_num + 1, dtype=np.float64)
        etc_distance_reward = self.reward_funcs["etc_mindistance_reward"](min_car_dis_list, etc_distance_reward)
        reward_results["etc_mindistance_reward"] = etc_distance_reward
        return reward_results


    def compute_traffic_light_reward(self, policy_update_output, raw_env, reward_results) -> Dict[str, torch.Tensor]:
        """计算闯红灯reward."""
        reward_results.update(
            {
                "traffic_light_reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            }
        )

        ego_polygon = policy_update_output["ego_polygon"]

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            reward_results["traffic_light_reward"][pred_ts + 1] = self.reward_funcs["traffic_light_reward"](
                ego_polygon[self._history_t_num + pred_ts - 1 : self._history_t_num + pred_ts + 1],
                raw_env,
                pred_ts + self._history_t_num,
            )

        return reward_results

    def revise_navi_centerline(self, raw_env):
        pre_navi_centerlines = raw_env[0]["navi_infos"]["navi_centerlines"]
        dobjs_full = raw_env[0]["dobjs_full"]
        dobjs_full_polygon = raw_env[0]["dobjs_polygon"]
        if not pre_navi_centerlines:
            return pre_navi_centerlines
        raw_env_length = len(raw_env)
        revised_navi_centerline = False
        for index in range(raw_env_length - 1, 0, -1):
            if revised_navi_centerline:
                break
            if len(raw_env[index]["parked_vehicle"]) > 0:
                _, dobjs = dobjs_full[index + self._history_t_num]
                dobjs_polygon = dobjs_full_polygon[index + self._history_t_num]
                trackids = dobjs[:, 9].astype(np.int32)
                cxcy = dobjs_polygon.mean(axis=1)
                for parked_vehicle_info in raw_env[index]["parked_vehicle"]:
                    if parked_vehicle_info["object_stop_info"] == 1:
                        for d, (x1, y1), tid in zip(dobjs_polygon, cxcy, trackids):
                            if tid == parked_vehicle_info["object_id"]:
                                for idx, navi_centerline in enumerate(pre_navi_centerlines):
                                    navi_points = navi_centerline[0]
                                    reference_line = ReferenceLine(navi_points, 0)
                                    s, d = reference_line.xytosl(x1, y1)
                                    if d is None:
                                        continue
                                    if abs(d) < 1.0:
                                        lane_line_seg = navi_points[:, :2]
                                        projection_index = np.argmin(np.linalg.norm(lane_line_seg - [(x1, y1)], axis=1))
                                        # 确保切片后至少保留2个点，否则calc_path_point_heading会报错
                                        # 如果projection_index太接近末尾，保留至少2个点
                                        max_projection_index = len(navi_centerline[0]) - 2
                                        projection_index = min(projection_index, max_projection_index)
                                        if projection_index < 0:
                                            # 如果原始导航中心线少于2个点，跳过修改
                                            continue
                                        navi_centerline[0] = navi_centerline[0][projection_index:]
                                        # 确保修改后至少还有2个点
                                        if len(navi_centerline[0]) < 2:
                                            continue
                                        if navi_centerline[1]:
                                            navi_centerline[1] = max(0, navi_centerline[1] - projection_index)
                                        raw_env[0]["navi_infos"]["navi_centerlines"][idx] = navi_centerline
                                        revised_navi_centerline = True



    def compute_navi_lane_reward(self, policy_update_output, raw_env, reward_results):
        """计算navi lane reward."""
        cur_env = raw_env[self._history_t_num - 1]
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        raw_env_clone = deepcopy(raw_env)
        gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]

        junc_scene_list = np.zeros_like(reward_results["reward"])
        od_scene_list = np.zeros_like(reward_results["reward"])
        spd_limit_list = policy_update_output["spd_limit_list"]
        spd_limit = spd_limit_list[0] / 3.6
        pred_traj = get_xyyaw_from_polygon(pred_polygon)
        pred_dxdy = pred_traj[1:][:, :2] - pred_traj[:-1][:, :2]
        pred_vxvy = pred_dxdy * self.fps * 3.6  # m/s -> km/h
        pred_vels = np.linalg.norm(pred_vxvy, axis=1)
        # vel = pred_vels[0]
        # ego_polygon = pred_polygon[0]
        # ego_pos = np.mean(ego_polygon, axis=0)
        # ego_point = Point(ego_pos)
        # hw场景标识
        is_highway_scene = raw_env[0]["priority_road_class"] == 0 or raw_env[0]["priority_road_class"] == 1
        #计算每一个时刻的路口标签和od场景标签
        # TBC
        overtake_obj_ids = []
        navi_centerline_reference_lists = []
        for navi_centerline in raw_env[0]["navi_infos"]["navi_centerlines"]:
            navi_centerline_reference_lists.append(ReferenceLine(navi_centerline[0], 0))
        for pred_ts in range(0, self._pred_t_num - 1):
            # 路口隔离
            vel = pred_vels[pred_ts]
            ego_polygon = pred_polygon[pred_ts]
            ego_pos = np.mean(ego_polygon, axis=0)
            ego_point = Point(ego_pos)
            is_junc_scene, line_data1, line_data2 = self._check_junction_scenario_step(raw_env[pred_ts], ego_point)

            # OD场景隔离,自车所在车道前方有低速障碍物时，不居中，低于限速 0.5倍，低于自车速度0.7倍
            is_od_scene = False
            if not is_junc_scene:
                is_od_scene = self._check_od_scenario_step(line_data1, line_data2, raw_env[0], raw_env[pred_ts], pred_ts, pred_polygon, spd_limit_list[pred_ts], vel, pred_traj[pred_ts][-1], overtake_obj_ids, navi_centerline_reference_lists)
            junc_scene_list[pred_ts] = is_junc_scene
            od_scene_list[pred_ts] = is_od_scene

        # od_scene_list = self._filter_consecutive_true(od_scene_list, min_length=5) # 至少存在1秒

        # 计算跟慢车场景隔离
        scenario_id = policy_update_output['scenario_id']
        is_slow_scene_all = is_slow_scene(scenario_id)
        is_dead_car_scene = ScenarioEnum.kDeadCarBypass == scenario_id
        # 根据死车信号修改Navi centerline
        self.revise_navi_centerline(raw_env)

        navi_centerline_reward, navi_lane_change_reward = self.reward_funcs["navi_lane_reward"](
            pred_polygon,gt_polygon, raw_env[0]["navi_infos"]["navi_centerlines"], is_highway_scene, od_scene_list, is_slow_scene_all, is_dead_car_scene
        )

        reward_results.update(
            {
                "navi_lane_reward": np.zeros_like(reward_results["reward"]),
                "navi_lane_change_reward": np.zeros_like(reward_results["reward"]),
                "junc_scene_list": junc_scene_list,
                "od_scene_list": od_scene_list,
            }
        )

        scenario_id = policy_update_output['scenario_id']
        is_campus = is_campus_scene(scenario_id)
        if is_campus:
            return reward_results

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            if len(navi_centerline_reward):
                reward_results["navi_lane_reward"][pred_ts + 1] = navi_centerline_reward[pred_ts]
                if abs(navi_centerline_reward[pred_ts] - (-19)) < 1e-6 and pred_ts > 10:  # 实线区域偏航
                    reward_results["final_result"] = 2
                    reward_results["dones"][pred_ts + 1] = 1
                    if pred_ts < self._pred_t_num - 1:
                        reward_results["dones"][pred_ts + 2 :] = -1
                    reward_results["reward"][pred_ts + 2 :] = 0
                    reward_results["result"][pred_ts + 1 :] = 2
                    break
        if not reward_results["collision_except_rear"]:
            for pred_ts in range(0, self._pred_t_num):
                if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                    break
                if len(navi_lane_change_reward):
                    if not reward_results["is_danger_lc"][pred_ts + 1]:
                        reward_results["navi_lane_change_reward"][pred_ts + 1] = navi_lane_change_reward[pred_ts]
        np_plus_lcc_status = cur_env.get("np_plus_lcc_status", 0)
        # 拨杆期间不计算navi lane reward
        for pred_ts in range(0, self._pred_t_num):
            pred_ts_env = raw_env[pred_ts]
            turn_signal = pred_ts_env.get("lane_change_signal", 0)
            if turn_signal != 0:
                reward_results["navi_lane_reward"][pred_ts + 1] = 0
                reward_results["navi_lane_change_reward"][pred_ts + 1] = 0
            if np_plus_lcc_status:
                reward_results["navi_lane_change_reward"][pred_ts + 1] = 0
        return reward_results

    def compute_virtual_wall_reward(self, policy_update_output, raw_env, reward_results):
        """计算virtual wall reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]
        ego_gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        virtual_wall_reward = self.reward_funcs["virtual_wall_reward"](pred_polygon, ego_gt_polygon, raw_env)

        reward_results.update(
            {
                "virtual_wall_reward": np.zeros_like(reward_results["reward"]),
            }
        )

        for pred_ts in range(0, self._pred_t_num):
            if len(virtual_wall_reward):
                reward_results["virtual_wall_reward"][pred_ts + 1] = virtual_wall_reward[pred_ts]

        return reward_results

    def compute_choose_etc_reward(self, policy_update_output, raw_env, reward_results):
        """计算choose etc reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]
        speeds = policy_update_output["speeds"]

        (
            choose_etc_reward,
            choose_etc_gate,
            all_unknown_gate,
            no_entry_toll_scene,
            entry_gate_index,
            palza_expert_zone,
         ) = self.reward_funcs["choose_etc_reward"](pred_polygon, gt_polygon, speeds, raw_env, self._history_t_num)

        reward_results.update(
            {
                "choose_etc_reward": np.zeros_like(reward_results["reward"]),
                "choose_etc_gate": choose_etc_gate and (not no_entry_toll_scene) and (not all_unknown_gate),
                "entry_gate_index": entry_gate_index,
                "palza_expert_zone": palza_expert_zone,
            }
        )

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            if len(choose_etc_reward):
                reward_results["choose_etc_reward"][pred_ts + 1] = choose_etc_reward[pred_ts]
                if abs(choose_etc_reward[pred_ts] - (-19)) < 1e-6 and pred_ts > 10:  # 实线区域偏航
                    reward_results["final_result"] = 2
                    reward_results["dones"][pred_ts + 1] = 1
                    if pred_ts < self._pred_t_num - 1:
                        reward_results["dones"][pred_ts + 2 :] = -1
                    reward_results["reward"][pred_ts + 2 :] = 0
                    reward_results["result"][pred_ts + 1 :] = 2
                    break

        return reward_results

    def compute_gate_machine_reward(self, policy_update_output, raw_env, reward_results):
        """计算gate machine reward."""
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]
        ego_gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        gate_machine_reward, gate_gt_events = self.reward_funcs["gate_machine_reward"](pred_polygon, ego_gt_polygon,raw_env, reward_results)

        reward_results.update(
            {
                "gate_machine_reward": np.zeros_like(reward_results["reward"]),
                "gate_gt_events": gate_gt_events,
            }
        )

        scenario_id = policy_update_output['scenario_id']
        if not is_campus_scene_v2(scenario_id):
            return reward_results

        for pred_ts in range(0, self._pred_t_num):
            if len(gate_machine_reward):
                reward_results["gate_machine_reward"][pred_ts + 1] = gate_machine_reward[pred_ts]

        return reward_results

    def compute_navi_reward(self, policy_update_output, raw_env, reward_results):
        """计算navi reward."""
        cur_env = raw_env[self._history_t_num - 1]
        reward_results.update(
            {
                "navi_roadsign": np.zeros_like(reward_results["reward"]),
                "navi_link": np.zeros_like(reward_results["reward"]),
            }
        )

        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]
        # 计算自车穿过stopline的信息，后面多个函数用到
        stopline_crossings = self.get_stopline_cross_points(pred_polygon, raw_env, threshold=2)
        # 此处只计算了最终的导航状态，基于过程的导航奖励移到前面计算
        navi_road_sign_reward, navi_link_reward = self.reward_funcs["navi_reward"](
            pred_polygon, gt_polygon, raw_env, stopline_crossings
        )

        speeds = policy_update_output["speeds"]

        mask = reward_results["dones"][1:] >= 0
        speeds_sum = np.sum(speeds * mask)
        len_rewards = int(np.sum(mask))
        if stopline_crossings and stopline_crossings[0][0] < len_rewards:
            for i in range(len_rewards):
                r = navi_road_sign_reward * speeds[i] / speeds_sum if speeds_sum > 0 else 0
                reward_results["navi_roadsign"][i + 1] = r

        # navi link reward, 自车轨迹是否在推荐道路内行驶 （匝道， 主辅路）
        reward_results["navi_link"][len_rewards] = navi_link_reward

        reward_results.update(
            {
                "stopline_crossings": stopline_crossings,
            }
        )
        reward_results["navi_reward"] = reward_results["navi_link"] + reward_results["navi_roadsign"]

        # 拨杆期间不计算navi lane reward
        for pred_ts in range(0, self._pred_t_num):
            pred_ts_env = raw_env[pred_ts]
            turn_signal = pred_ts_env.get("lane_change_signal", 0)
            if turn_signal != 0:
                reward_results["navi_reward"][pred_ts + 1] = 0
                reward_results["navi_link"][pred_ts + 1] = 0
                reward_results["navi_roadsign"][pred_ts + 1] = 0

        return reward_results

    def compute_danger_lane_change_reward(self, policy_update_output, raw_env, reward_results):
        """计算危险变道惩罚"""
        #计算gt和pre的进度的diff
        #计算roll-out进度
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        gt_polygon = policy_update_output["gt_polygon"][self._history_t_num :]
        ego_pos =  np.mean(pred_polygon,axis=-2)
        ego_pos_diff = np.diff(ego_pos,axis=0)
        ego_pos_diff_abs = np.linalg.norm(ego_pos_diff,axis=1)
        ego_pos_s = np.insert(np.cumsum(ego_pos_diff_abs,axis=0),0,0)
        #计算gt进度
        gt_pos =  np.mean(gt_polygon,axis=-2)
        gt_pos_diff = np.diff(gt_pos,axis=0)
        gt_pos_diff_abs = np.linalg.norm(gt_pos_diff,axis=1)
        gt_pos_s = np.insert(np.cumsum(gt_pos_diff_abs,axis=0),0,0)
        # s_diff = gt_pos_s - ego_pos_s

        util_func_name = 'lc_reward'
        ego_polygons = policy_update_output["ego_polygon"]
        ego_speeds = policy_update_output["speeds"]
        reward_results.update(
            {
                "danger_lc_reward": np.zeros_like(reward_results["reward"]),
                "lc_status": np.zeros_like(reward_results["reward"]),
                "lc_back_obj_id": np.zeros_like(reward_results["reward"]),
                "lc_back_safe_dist": np.zeros_like(reward_results["reward"]),
                "lc_ahead_obj_id": np.zeros_like(reward_results["reward"]),
                "is_danger_lc": np.zeros_like(reward_results["reward"]).astype(bool),
            }
        )
        # 计算跟慢车场景隔离
        scenario_id=policy_update_output['scenario_id']
        if is_slow_scene(scenario_id) or is_campus_scene(scenario_id):
            return reward_results

        valid_len = min(len(gt_pos_s),len(ego_pos_s))

        danger_lc_result = self.reward_funcs["danger_lc_reward"](ego_polygons, ego_speeds, raw_env, self._history_t_num)
        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            if pred_ts<valid_len:
                # if gt_pos_s[pred_ts] - ego_pos_s[pred_ts] < 2:
                pred_ts_env = raw_env[pred_ts + self._history_t_num]
                turn_signal = pred_ts_env.get("lane_change_signal", 0)
                if turn_signal != 0:
                    reward_results["danger_lc_reward"][pred_ts + 1] = 0.0
                else:
                    reward_results["danger_lc_reward"][pred_ts + 1] = danger_lc_result["danger_lc_penalty"][pred_ts]

                reward_results["lc_status"][pred_ts + 1] = danger_lc_result["lc_status"][pred_ts]
                reward_results["lc_back_obj_id"][pred_ts + 1] = danger_lc_result["lc_back_obj_id"][pred_ts]
                reward_results["lc_back_safe_dist"][pred_ts + 1] = danger_lc_result["lc_back_safe_dist"][pred_ts]
                reward_results["lc_ahead_obj_id"][pred_ts + 1] = danger_lc_result["lc_ahead_obj_id"][pred_ts]
                reward_results["is_danger_lc"][pred_ts + 1] = danger_lc_result["is_danger_lc"][pred_ts]
        return reward_results

    def compute_wrong_way_reward(self, policy_update_output, raw_env, reward_results):
        """计算逆行reward."""
        reward_results.update(
            {
                "wrong_way_reward": np.zeros_like(reward_results["reward"]),
            }
        )

        scenario_id = policy_update_output['scenario_id']
        is_campus = is_campus_scene(scenario_id)
        if is_campus:
            return reward_results

        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        raw_env = raw_env[self._history_t_num :]
        assert "stopline_crossings" in reward_results
        stopline_crossings = reward_results["stopline_crossings"]

        wrongway_penalty, t_stop = self.reward_funcs["wrong_way_reward"](pred_polygon, raw_env, stopline_crossings)
        mask = reward_results["dones"][1:] >= 0
        len_rewards = int(np.sum(mask))
        reward_results["wrong_way_reward"][1 + min(t_stop, len_rewards - 1)] = -wrongway_penalty

        return reward_results

    def compute_humanoid_reward(self, policy_update_output, raw_env, reward_results):
        """计算humanoid reward."""
        reward_results.update(
            {
                "humanoid_reward": np.zeros_like(reward_results["reward"]),
                "passageway": None
            }
        )
        pred_polygon = policy_update_output["ego_polygon"]
        gt_polygon = policy_update_output["gt_polygon"]
        np_plus_lcc_status = raw_env[self._history_t_num-1].get("np_plus_lcc_status", 0)

        is_campus = is_campus_scene(policy_update_output["scenario_id"])

        if not np.any(reward_results["navi_lane_reward"] < 0) and not is_campus:
            humanoid_reward, passageway = self.reward_funcs["humanoid_reward"](pred_polygon, gt_polygon,
                                                                                raw_env, self._history_t_num)
            if passageway:
                for pred_ts in range(0, self._pred_t_num):
                    if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                        break
                    pred_ts_env = raw_env[pred_ts + self._history_t_num]
                    turn_signal = pred_ts_env.get("lane_change_signal", 0)
                    if np_plus_lcc_status or turn_signal != 0:
                        reward_results["humanoid_reward"][pred_ts + 1] = 0.0
                    else:
                        reward_results["humanoid_reward"][pred_ts + 1] = humanoid_reward[pred_ts]
                reward_results["passageway"] = passageway
        return reward_results

    def compute_humanoid_nudge_reward(self, policy_update_output, raw_env, reward_results):
        """计算humanoid reward."""
        reward_results.update(
            {
                "humanoid_nudge_reward": np.zeros_like(reward_results["reward"]),
            }
        )
        # if np.any(reward_results["scene_exclusion_flag"]):
        #     return reward_results
        #
        pred_polygon = policy_update_output["ego_polygon"]
        gt_polygon = policy_update_output["gt_polygon"]

        scenario_id = policy_update_output['scenario_id']
        is_car_queue_scene = ScenarioEnum.kCarQueueSidePass == scenario_id
        is_dead_car_scene = ScenarioEnum.kDeadCarBypass == scenario_id
        danger_narrow_flag = False
        if is_dead_car_scene:
            danger_narrow_flag = policy_update_output['scenario_info']['danger_narrow_flag']
        np_plus_lcc_status = raw_env[self._history_t_num-1].get("np_plus_lcc_status", 0)

        humanoid_reward, progress_diff = self.reward_funcs["humanoid_nudge_reward"](pred_polygon, gt_polygon,
                                                                        raw_env, self._history_t_num,reward_results)

        if np.any(humanoid_reward):
            for pred_ts in range(0, self._pred_t_num):
                if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                    break
                pred_ts_env = raw_env[pred_ts + self._history_t_num]
                turn_signal = pred_ts_env.get("lane_change_signal", 0)
                if np_plus_lcc_status or turn_signal != 0 or is_car_queue_scene or danger_narrow_flag:
                    reward_results["humanoid_nudge_reward"][pred_ts + 1] = 0.0
                else:
                    reward_results["humanoid_nudge_reward"][pred_ts + 1] = humanoid_reward[pred_ts]

        return reward_results

    def compute_continuous_lane_change_reward(self, policy_update_output, raw_env, reward_results):
        """
        计算连续换道惩罚reward.
        """
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        raw_env      = raw_env[self._history_t_num :]

        continuous_lc_penalty = self.reward_funcs["continuous_lane_change_reward"](
            pred_polygon, raw_env)

        reward_results.update({
            "continuous_lane_change_reward": np.zeros_like(reward_results["reward"]),
        })

        for pred_ts in range(0, self._pred_t_num):
            if "dones" in reward_results and reward_results["dones"][pred_ts + 1] != 0:
                break
            if len(continuous_lc_penalty):
                reward_results["continuous_lane_change_reward"][pred_ts + 1] = continuous_lc_penalty[pred_ts]


        return reward_results

    def process_gt_mask(self, policy_update_output, raw_env, train_mode=True):
        """Preprocess."""
        lane_infos = raw_env[0]["navi_infos"]["lane_infos"]
        left_on_right_boost = False

        for idx, info in enumerate(lane_infos):
            # 必须同时满足：当前车道方向==8 且 高亮方向==8 且 可行驶
            if info[0] == 8 and info[1] == 8 and info[2]:
                # 只要这条左转推荐车道不是最左侧（idx>0），就触发
                if idx > 0:
                    left_on_right_boost = True
                break # 找到第一条匹配的即可

        # 阈值修改
        if left_on_right_boost:   # 左转非左一
            critical_distance = 100.0
        else:
            critical_distance = 50.0
        ego_polygon = policy_update_output["ego_polygon"]
        gt_polygon = policy_update_output["gt_polygon"]
        pred_polygon = ego_polygon[self._history_t_num :]
        gt_polygon = gt_polygon[self._history_t_num :]

        pred_t_num = pred_polygon.shape[0]
        self._pred_t_num = pred_t_num

        reward_results = {
            "reward": np.zeros(self._pred_t_num + 1, dtype=np.float64),
            "result": np.zeros(self._pred_t_num + 1, dtype=np.float32) - 1,
            "dones": np.zeros(self._pred_t_num + 1, dtype=np.float32),
            "lose_gt": np.zeros(self._pred_t_num + 1, dtype=np.float32),
            "final_result": 1,
        }

        # 超出gt范围
        for pred_ts in range(0, pred_t_num):
            if pred_ts >= len(gt_polygon):
                reward_results["dones"][pred_ts + 1] = 1
                if pred_ts < pred_t_num - 1:
                    reward_results["dones"][pred_ts + 2 :] = -1
                break

            distance = compute_center_distance(pred_polygon[pred_ts], gt_polygon[pred_ts])
            if distance > critical_distance:
                if not raw_env[0]["gt_unreliable"]:
                    reward_results["final_result"] = 0
                    reward_results["dones"][pred_ts + 1] = 1
                    if pred_ts < pred_t_num - 1:
                        reward_results["dones"][pred_ts + 2 :] = -1
                    reward_results["result"][pred_ts + 1] = 0
                    reward_results["lose_gt"][pred_ts + 1] = 1
                break

        return reward_results

    def compute_junction_lane_select_reward(self, policy_update_output, raw_env, reward_results):
        junction_lane_select_reward = np.zeros_like(reward_results["reward"])
        pred_polygon = policy_update_output["ego_polygon"][self._history_t_num :]
        spd_limit_list = policy_update_output["spd_limit_list"]
        spd_limit = spd_limit_list[0] / 3.6
        pred_traj = get_xyyaw_from_polygon(pred_polygon)
        pred_dxdy = pred_traj[1:][:, :2] - pred_traj[:-1][:, :2]
        pred_vxvy = pred_dxdy * self.fps * 3.6  # m/s -> km/h
        pred_vels = np.linalg.norm(pred_vxvy, axis=1)
        vel = pred_vels[0]
        cur_env = raw_env[self._history_t_num]
        ego_polygon = pred_polygon[0]
        ego_pos = np.mean(ego_polygon, axis=0)
        ego_point = Point(ego_pos)
        passway = reward_results.get("passageway", None)
        gt_traj = np.mean(policy_update_output["gt_polygon"], axis=1)

        # 逆向车道与偏航隔离
        wrong_way_reward_sum = np.abs(reward_results["wrong_way_reward"].sum())
        is_wrong_way = wrong_way_reward_sum > 0.1

        # 类人reward
        is_human_reward_valid = np.abs(reward_results['humanoid_reward']).sum() > 0.1

        # 路口隔离
        _, line_data1, line_data2 = self._check_junction_scenario(cur_env, ego_point)
        # 精细化的路口判断
        is_junc_scene, next_lanenr_idx = self._is_in_junction(raw_env, pred_polygon)

        # OD场景隔离,自车所在车道前方有低速障碍物时，不居中，低于限速 0.5倍，低于自车速度0.7倍
        is_od_scene = False
        if is_junc_scene:
            is_od_scene = self._check_od_scenario(line_data1, line_data2, raw_env, ego_polygon, spd_limit, vel)

        if not is_wrong_way and is_junc_scene and not is_od_scene:
            selected_env = raw_env[next_lanenr_idx + self._history_t_num]
            # cur_env = raw_env[pred_ts + self._history_t_num]
            mask = reward_results["dones"][1:] != 0
            junction_lane_select_reward[1:] = self.reward_funcs["junction_lane_select_reward"](
                selected_env, pred_traj[:, :2], next_lanenr_idx, passway, is_human_reward_valid, gt_traj
            )
            junction_lane_select_reward[1:][mask] = 0

        reward_results.update(
            {
                "junction_lane_select_reward": junction_lane_select_reward,
            }
        )
        return reward_results

    def _calc_rewards(self, policy_update_output, raw_env, train_mode=True):
        """Calculate all rewards."""
        reward_results = self.process_gt_mask(policy_update_output, raw_env, train_mode=train_mode)
        reward_results = self.compute_scenario_exclusion_flags(policy_update_output, raw_env, reward_results)
        # collision_reward/navi_lane_reward/gt 有提前终止其他reward的逻辑，目前通过dones=1标记位置, 修改reward时注意踩坑
        iter_num = policy_update_output.get("iter_num", 0)

        reward_timing = {}

        for reward_type, _ in self.reward_funcs.items():
            start_t = time.perf_counter()
            if reward_type == "cross_solid_line_reward":
                reward_results = self.compute_cross_solid_line_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "danger_lc_reward":
                reward_results = self.compute_danger_lane_change_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "collision_reward":
                # 碰撞当前耦合了cross_solid_line_reward的结果，只能在放在压线后面计算
                reward_results = self.compute_collision_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "navi_lane_reward":
                # 根据plannn2的break顺序,navi_lane_reward需在collision_reward后面计算
                reward_results = self.compute_navi_lane_reward(policy_update_output, raw_env, reward_results)
                # navi_lane_change_reward update
                reward_results["reward"] += reward_results["navi_lane_change_reward"]
            if reward_type == "virtual_wall_reward":
                reward_results = self.compute_virtual_wall_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "choose_etc_reward":
                reward_results = self.compute_choose_etc_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "traffic_light_reward":
                reward_results = self.compute_traffic_light_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "speed_limit_reward":
                reward_results = self.compute_speed_limit_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "slow_follow_reward":
                reward_results = self.compute_slow_follow_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "acc_reward":
                reward_results = self.compute_acc_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "jerk_reward":
                reward_results = self.compute_jerk_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "etc_speed_reward":
                reward_results = self.compute_etc_speed_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "etc_take_reward":
                reward_results = self.compute_etc_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "etc_mindistance_reward":
                reward_results = self.compute_etc_dis_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "ttc_reward":
                reward_results = self.compute_ttc_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "min_distance_reward":
                reward_results = self.compute_min_distance_reward(policy_update_output, reward_results)
            if reward_type == "gate_machine_reward":
                reward_results = self.compute_gate_machine_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "navi_reward":
                reward_results = self.compute_navi_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "progress_reward":
                reward_results = self.compute_progress_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "wrong_way_reward":
                reward_results = self.compute_wrong_way_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "centralization_reward":
                reward_results = self.compute_centralization_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "continuous_lane_change_reward":
                reward_results = self.compute_continuous_lane_change_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "toggle_reward":
                reward_results = self.compute_toggle_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "humanoid_reward":
                reward_results = self.compute_humanoid_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "humanoid_nudge_reward":
                reward_results = self.compute_humanoid_nudge_reward(policy_update_output, raw_env, reward_results)
            if reward_type == "junction_lane_select_reward":
                reward_results = self.compute_junction_lane_select_reward(policy_update_output, raw_env, reward_results)

            self._apply_reward_extra_cfg(reward_type, policy_update_output['scenario_id'], reward_results)

            reward_weight = self.rewards_weights.get(reward_type, 1.0)
            reward_results[reward_type] *= reward_weight
            reward_results["reward"] += reward_results[reward_type]

            reward_timing[reward_type] = time.perf_counter() - start_t

        occur_collision = np.where(reward_results["collision_reward"] <= -10)[0]
        if occur_collision.size and policy_update_output["iter_num"] <= 2100:
            # 当存在碰撞的时候，把碰撞之前的所有奖励都置为0
            collision_point_index = occur_collision[0]
            reward_results["reward"][:collision_point_index] = np.where(
                reward_results["reward"][:collision_point_index] > 0,
                0,
                reward_results["reward"][:collision_point_index],
            )
        # has_humanoid_nudge_reward = np.abs(reward_results['humanoid_nudge_reward'])>1e-3
        # reward_results['slow_follow_reward'][has_humanoid_nudge_reward]=0

        # hard code for alignment to plannn2
        gt_invalid_mask = reward_results["lose_gt"] == 1
        if gt_invalid_mask.sum() > 0:
            reward_results["reward"][gt_invalid_mask] -= 10

        return reward_results, reward_timing

    def _aggregate_reward_timing(self, timing_list):
        if self._stopwatch is None:
            return

        agg = {}

        for timing in timing_list:
            for k, v in timing.items():
                agg[k] = agg.get(k, 0.0) + v

        for reward_type, cost in agg.items():
            key = f"rl_reward_model/calc_rewards/{reward_type}"
            self._stopwatch._add_time_mode_cost(key,"cpu", cost/len(timing_list))

    def _apply_reward_extra_cfg(self, reward_type:str, scenario_id:ScenarioEnum, reward_results:dict):
        if reward_type in self.reward_extra_cfg :
            extra_cfg=self.reward_extra_cfg[reward_type]
            for name, cfg in extra_cfg.items():
                if ScenarioEnum.from_string(name)==scenario_id:
                    if 'clamp_params' in cfg:
                        clamp_params = cfg['clamp_params']
                        if 'min' in clamp_params:
                            reward_results[reward_type] = np.clip(reward_results[reward_type], clamp_params['min'], None)
                        if 'max' in clamp_params:
                            reward_results[reward_type] = np.clip(reward_results[reward_type], None, clamp_params['max'])
                if cfg.get('omit_junc_merge', False):
                    scene_exclusion_flag = reward_results["scene_exclusion_flag"]
                    reward_results[reward_type][scene_exclusion_flag] = 0

    def _calc_deadcar_bypass_scenario(self, policy_update_output, raw_env):
        max_len = max(policy_update_output["ego_polygon"].shape[0], len(raw_env[0]["dobjs_full"]))

        is_narrow_road_scenario = False
        link_info = raw_env[self._history_t_num].get("link_info",None)
        if link_info is not None and len(link_info)>0 and link_info[0] is not None:
            is_narrow_road_scenario = link_info[0].get("road_class", 0) >= 7

        ego_gt_polygon = policy_update_output["gt_polygon"]
        ego_xyyaw= get_xyyaw_from_polygon(ego_gt_polygon)[:, [0,1,6]]

        obj_dct = { i:(t[1][:, 9], t[1][:,[0,1,6,3,4,7]]) for i,t in enumerate(raw_env[0]["dobjs_full"][:max_len])}

        max_len = min(len(ego_xyyaw), len(obj_dct))
        ego_xyyaw = ego_xyyaw[:max_len]
        if len(obj_dct)>max_len:
            for i in range(max_len, len(obj_dct)):
                obj_dct.pop(i)

        ts = np.arange(ego_xyyaw.shape[0], dtype=float)*0.2

        detector = DeadCarBypassDetectorV3()

        detect_input = dict(
            ego_traj=ego_xyyaw,  # np.ndarray[T,3]
            agent_dict=obj_dct,  # Dict[int, (ids, poses)]
            timestamps=ts,
            is_narrow_road_scenario=is_narrow_road_scenario,  # np.ndarray[T]
        )

        # import pickle as pkl
        # import os
        # fp = f"./tmp_deadcar/{raw_env[0]['idx']}.pkl"
        #
        # detect_input_dump = detect_input
        # pkl.dump(dict(input=detect_input_dump, v3_ceph=raw_env[0]['v3_ceph']), open(fp, 'wb'))
        # print(f"\n\n {os.getpid()} {raw_env[0]['idx']} - {fp}\n\n")

        overtake_flags, target_agent_ids, episodes, danger_narrow_flag = detector.detect(**detect_input)
        is_deadcar_bypass_scene = np.any(overtake_flags)
        # print(f"[{raw_env[0]['idx']}] is_deadcar_bypass_scene : {is_deadcar_bypass_scene}")
        if is_deadcar_bypass_scene:
            policy_update_output['scenario_id'] = ScenarioEnum.kDeadCarBypass
            policy_update_output['scenario_info']["target_agent_ids"] = target_agent_ids,
            policy_update_output['scenario_info']["episodes"] = episodes
            policy_update_output['scenario_info']['danger_narrow_flag'] = danger_narrow_flag

    def _calc_scenario_follow_slow(self, policy_update_output, raw_env):
        if ScenarioEnum.UNK != policy_update_output['scenario_id'] and ScenarioEnum.kFollowSlow != policy_update_output['scenario_id']:
            return

        navi_infos = raw_env[self._history_t_num]["navi_infos"]

        def is_long_dist_flag(action, dist):
            flag = (8!=action) and (action in (3,4,5,6,9,10))  # misc.py
            return flag and (0< dist < 1000)

        def is_short_dist_flag(action, dist):
            flag = (8!=action) and (action not in (3,4,5,6,9,10))  # misc.py
            return flag and (0< dist< 300)

        action1 = navi_infos.get("main_action_1", -1)
        dist1 = navi_infos.get("link_main_distance_1", 10000)
        if is_long_dist_flag(action1, dist1) or is_short_dist_flag(action1, dist1):
            return
        action2 = navi_infos.get("main_action_2", -1)
        dist2 = navi_infos.get("link_main_distance_2", 10000)
        if is_long_dist_flag(action2, dist2) or is_short_dist_flag(action2, dist2):
            return

        ego_polygon = policy_update_output["ego_polygon"]
        ego_gt_polygon = policy_update_output["gt_polygon"]

        ego_xyyaw= get_xyyaw_from_polygon(ego_polygon)[:, [0,1,6]]

        ego_length = np.linalg.norm(ego_polygon[0, 0] - ego_polygon[0, 1], axis=0)
        ego_width = np.linalg.norm(ego_polygon[0, 1] - ego_polygon[0, 2], axis=0)

        centerlines_dct = dict()
        for i,r in enumerate(raw_env):
            centerlines = [cl[0] for cl in r['centerlines']]
            centerlines_dct[i]=centerlines

        solid_lane_line_dct = dict()
        for i, r in enumerate(raw_env):
            solid_lines = []
            for pts, attr in r["lane_lines"]: # 只取实线段
                ltypes = attr[:, 0]
                solid_mask = (ltypes==1) | (ltypes==3) | (ltypes==4) | (ltypes==5) | (ltypes==7) | (ltypes==8)
                solid_mask_pad = np.concatenate([[False], solid_mask, [False]])
                diff = np.diff(solid_mask_pad.astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for start, end in zip(starts, ends):
                    if end-start>2 and np.linalg.norm(pts[end-1]-pts[start])>30:
                        solid_lines.append(pts[start:end])
            solid_lane_line_dct[i] = solid_lines

        road_edge_dct = dict()
        for i,r in enumerate(raw_env):
            road_edge_dct[i] = r['road_edge']

        obj_dct = { i:(t[1][:, 9], t[1][:,[0,1,6,3,4]]) for i,t in enumerate(raw_env[0]["dobjs_full"])}
        spd_limit = np.array([r['spd_limit'] for r in raw_env])
        ts = np.arange(ego_polygon.shape[0], dtype=float)*0.2

        navi_centerline_dct = {}
        for i,r in enumerate(raw_env):
            ncls = []
            for ln in r['navi_infos']['navi_centerlines']:
                if len(ln)>=3 and 0==ln[2]:
                    ncls.append((ln[0][:,:2], ln[1]))
            navi_centerline_dct[i] = ncls

        # detector = SlowFollowDetectorV3(
        detector = SlowFollowDetectorV3_1(
            ego_length=ego_length,
            ego_width=ego_width,
            min_ego_speed_ms=1.38,  # ~5kph #2.78,  # ~10 kph
            slow_ratio=0.9,  # 低于 0.5 * 限速算“慢”
            min_segment_duration=5.0,  # 满足条件至少 10 秒算跟慢车场景
            min_junction_dist=50.0,  # 距 navi 最晚变道点大于此值才算“非路口”（m）
        )

        detector.slow_ratio = 0.9
        if len(spd_limit)>0 and spd_limit[0]<=60.0:
            detector.slow_ratio = 0.8

        detect_input = dict(
            ego_traj=ego_xyyaw,  # np.ndarray[T,3]
            agent_dict=obj_dct,  # Dict[int, (ids, poses)]
            speed_limit_kph=spd_limit,  # np.ndarray[T]
            lane_centerlines=centerlines_dct,  # Dict[int, List[np.ndarray[P,3]]]
            # road_edges=road_edge_dct,  # Dict[int, List[np.ndarray[P,3]]]
            solid_lanes=solid_lane_line_dct,  # Dict[int, List[np.ndarray[P,2]]]
            navi_centerlines=navi_centerline_dct,  # Dict[int, List[(np.ndarray[P,3], int)]]
            timestamps=ts,  # np.ndarray[T]
        )

        # import pickle as pkl
        # import os
        # fp = f"./tmp4/{raw_env[0]['idx']}.pkl"
        #
        # detect_input_dump = dict(
        #     ego_traj=get_xyyaw_from_polygon(policy_update_output["gt_polygon"])[:, [0,1,6]],  # np.ndarray[T,3]
        #     agent_dict=obj_dct,  # Dict[int, (ids, poses)]
        #     speed_limit_kph=spd_limit,  # np.ndarray[T]
        #     lane_centerlines=centerlines_dct,  # Dict[int, List[np.ndarray[P,3]]]
        #     road_edges=road_edge_dct,  # Dict[int, List[np.ndarray[P,3]]]
        #     navi_centerlines=navi_centerline_dct,  # Dict[int, List[(np.ndarray[P,3], int)]]
        #     timestamps=ts,  # np.ndarray[T]
        # )
        # pkl.dump(dict(input=detect_input_dump, v3_ceph=raw_env[0]['v3_ceph']), open(fp, 'wb'))
        # print(f"\n\n {os.getpid()} {raw_env[0]['idx']} - {fp}\n\n")
        #
        is_slow_scene, segments, safe_drive_mask = detector.detect(**detect_input)
        if is_slow_scene:
            policy_update_output['scenario_id'] = ScenarioEnum.kFollowSlow
            segments_cut = []
            for seg in segments:
                if seg.end_idx<self._history_t_num:
                    continue
                seg_new = deepcopy(seg)
                seg_new.start_idx = seg.start_idx-self._history_t_num
                seg_new.end_idx = seg.end_idx-self._history_t_num
                segments_cut.append(seg_new)
            safe_drive_mask = safe_drive_mask[self._history_t_num:]
            policy_update_output['scenario_info']['follow_slow_segments']=segments_cut
            policy_update_output['scenario_info']['safe_drive_mask']=safe_drive_mask

    def _calc_scenario_car_queue(self, policy_update_output, raw_env):
        # 跟慢车是不考虑排队车场景
        if ScenarioEnum.UNK != policy_update_output['scenario_id'] and ScenarioEnum.kFollowSlow != policy_update_output['scenario_id'] and ScenarioEnum.kDeadCarBypass != policy_update_output['scenario_id']:
            # print(policy_update_output['scenario_id'])
            return

        # 小路直接干掉不管
        link_info = raw_env[self._history_t_num].get("link_info",None)
        if link_info is not None and len(link_info)>0 and link_info[0] is not None:
            is_narrow_road_scenario = link_info[0].get("road_class", 0) >= 7
            if is_narrow_road_scenario is True:
                # print("is_narrow_road_scenario, pass" )
                return

        # env_dict = dict()
        is_before_junction_array = np.zeros(len(raw_env))
        for i, raw_env_i in enumerate(raw_env):
            manner_scene_list = raw_env_i.get("manner_scene_info", None)
            if not manner_scene_list:
                continue
            manner_scene_list.sort(
                key=lambda x: x.get("gt_distance_to_entry", x.get("distance_to_entry",1e9))
            )
            is_junction_scenario, is_main_to_aux_scenario = False, False
            for scene in manner_scene_list:
                if is_main_to_aux_scenario or is_junction_scenario:
                    continue
                if scene.get("gt_distance_to_entry",None) is not None:
                    distance_to_entry = scene.get("gt_distance_to_entry")
                elif scene.get("distance_to_entry",None) is not None:
                    distance_to_entry = scene.get("distance_to_entry")
                else:
                    continue
                scene_type = scene["scene_type"]
                # 2: junction, 5: main_to_aux
                if 2==scene_type and 0< distance_to_entry < 150.0:
                    is_junction_scenario = True
                if 5==scene_type and 0<distance_to_entry < 150.0:
                    is_main_to_aux_scenario = True
                # if is_main_to_aux_scenario or is_junction_scenario:
                #     print(is_main_to_aux_scenario, is_junction_scenario)
                #     print(i, scene_type, distance_to_entry)
                is_before_junction_array[i] = is_main_to_aux_scenario or is_junction_scenario
        # print("is_before_junction_array ones sum: ",is_before_junction_array.sum())
        # print("is_before_junction_array zeros sum: ",(1-is_before_junction_array).sum())
        # gt 相关
        ego_gt_polygon = policy_update_output["gt_polygon"]
        ego_xyyaw= get_xyyaw_from_polygon(ego_gt_polygon)[:, [0,1,6]]
        ego_length = np.linalg.norm(ego_gt_polygon[0, 0] - ego_gt_polygon[0, 1], axis=0)
        ego_width = np.linalg.norm(ego_gt_polygon[0, 1] - ego_gt_polygon[0, 2], axis=0)

        # 导航线和车道中心线信息
        navi_centerline_dct = {}
        for i,r in enumerate(raw_env):
            ncls = []
            for ln in r['navi_infos']['navi_centerlines']:
                if len(ln)>=3 and 0==ln[2]:
                    ncls.append((ln[0][:,:2], ln[1]))
            navi_centerline_dct[i] = ncls

        centerlines_dct = dict()
        for i,r in enumerate(raw_env):
            centerlines = [cl[0] for cl in r['centerlines']]
            centerlines_dct[i]=centerlines

        # 30m长实线信息，暂时无用
        solid_lane_line_dct = dict()
        for i, r in enumerate(raw_env):
            solid_lines = []
            for pts, attr in r["lane_lines"]: # 只取实线段
                ltypes = attr[:, 0]
                solid_mask = (ltypes==1) | (ltypes==3) | (ltypes==4) | (ltypes==5) | (ltypes==7) | (ltypes==8)
                solid_mask_pad = np.concatenate([[False], solid_mask, [False]])
                diff = np.diff(solid_mask_pad.astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for start, end in zip(starts, ends):
                    if end-start>2 and np.linalg.norm(pts[end-1]-pts[start])>30:
                        solid_lines.append(pts[start:end])
            solid_lane_line_dct[i] = solid_lines

        # 路沿信息
        road_edge_dct = dict()
        for i,r in enumerate(raw_env):
            road_edge_dct[i] = r['road_edge']

        # 限速信息
        # spd_limit = np.array([r['spd_limit'] for r in raw_env])
        ts = np.arange(ego_gt_polygon.shape[0], dtype=float)*0.2

        car_queue_detector = CarQueueBypassDetector()
        max_len = max(policy_update_output["ego_polygon"].shape[0], len(raw_env[0]["dobjs_full"]))
        obj_dct_with_cls = { i:(t[1][:, 9], t[1][:,[0,1,6,3,4,7]]) for i,t in enumerate(raw_env[0]["dobjs_full"][:max_len])}
        max_len = min(len(ego_xyyaw), len(obj_dct_with_cls))
        ego_xyyaw = ego_xyyaw[:max_len]
        if len(obj_dct_with_cls)>max_len:
            for i in range(max_len, len(obj_dct_with_cls)):
                obj_dct_with_cls.pop(i)

        car_queue_detect_input = dict(
            ego_traj=ego_xyyaw,  # np.ndarray[T,3]
            agent_dict=obj_dct_with_cls,  # Dict[int, (ids, poses)]
            is_before_junction_array=is_before_junction_array,  # np.ndarray[T]
            lane_centerlines=centerlines_dct,  # Dict[int, List[np.ndarray[P,3]]]
            road_edges=road_edge_dct,  # Dict[int, List[np.ndarray[P,3]]]
            solid_lanes=solid_lane_line_dct,  # Dict[int, List[np.ndarray[P,2]]]
            navi_centerlines=navi_centerline_dct,  # Dict[int, List[(np.ndarray[P,3], int)]]
            timestamps=ts,  # np.ndarray[T]
        )
        overtake_flags, episodes = car_queue_detector.detect(**car_queue_detect_input)
        # print(overtake_flags)
        # print(episodes)
        if overtake_flags.sum():
            policy_update_output['scenario_id'] = ScenarioEnum.kCarQueueSidePass
            policy_update_output['scenario_info']['car_queue_times'] = overtake_flags
            policy_update_output['scenario_info']['car_queue_episodes']=episodes
        return overtake_flags, episodes

    def _pre_process(self, inputs_dict, train_mode=True):
        """Init output dict and process far from gt break."""
        pred_polygon = inputs_dict["pred_polygon"]
        gt_polygon = inputs_dict["gt_polygon"]
        raw_env = inputs_dict["raw_env"]
        iter_num = inputs_dict.get("iter_num", 0)

        if train_mode:
            # slice to one pred trajectory
            select_replan_time = inputs_dict["select_replan_time"]
            gt_property_np = inputs_dict["gt_propertys"]
            state_range = inputs_dict["state_range"]
            closedloop_simu_time = inputs_dict["closedloop_simu_time"]
            pred_polygon_np = pred_polygon
            gt_polygon_np = gt_polygon

        b = len(gt_polygon)
        batchsize = pred_polygon.shape[0]

        # ego_polygon, actions, gt_polygon
        gt_speeds = []
        for bid in range(batchsize):
            if train_mode:
                ego_gt_state_polygon = gt_polygon_np[bid % b, gt_property_np[bid % b, :, 3] == 1] * state_range  # 80
                ego_gt_state_polygon = ego_gt_state_polygon[
                    : closedloop_simu_time * select_replan_time + self._history_t_num
                ]  # 15+8*5=55
            else:
                ego_gt_state_polygon = gt_polygon[bid]

            dobj, dobj_next = get_xyyaw_from_polygon(
                ego_gt_state_polygon[self._history_t_num - 1 : self._history_t_num + 1]
            )
            pose_relative = get_relative_pose_from_obj(dobj, dobj_next)
            dx, dy = pose_relative[:2, 3]
            gt_speeds.append(np.linalg.norm([dx, dy]) * self.fps * 3.6)  # km/h

        params = []
        for bid in range(batchsize):
            if train_mode:
                ego_state_polygon = pred_polygon_np[bid] * state_range  # ego polygon  pred polygon, [8*5,4,2]
                ego_gt_state_polygon = gt_polygon_np[bid % b, gt_property_np[bid % b, :, 3] == 1] * state_range  # 80
                ego_gt_state_polygon = ego_gt_state_polygon[
                    : closedloop_simu_time * select_replan_time + self._history_t_num
                ]  # 15+8*5=55
                ego_state_polygon = np.concatenate(
                    (ego_gt_state_polygon[: self._history_t_num], ego_state_polygon), axis=0  # [15,4,2] + [120,4,2] -> [135,4,2]
                )  # 15 + 8*5
                ego_gt_state_polygon = ego_gt_state_polygon[: ego_state_polygon.shape[0]]
            else:
                ego_state_polygon = pred_polygon[bid]
                ego_gt_state_polygon = gt_polygon[bid]

            spd_limit_list = []
            for env in raw_env[bid][self._history_t_num :]:
                spd_limit_list.append(env["spd_limit"])

            speeds = []
            dx_list = []
            dy_list = []
            dyaw_list = []
            pred_ts_num = ego_state_polygon.shape[0]
            for pred_ts in range(0, pred_ts_num - self._history_t_num):
                dobj, dobj_next = get_xyyaw_from_polygon(
                    ego_state_polygon[self._history_t_num + pred_ts - 1 : self._history_t_num + pred_ts + 1]
                )
                pose_relative = get_relative_pose_from_obj(dobj, dobj_next)
                dx, dy = pose_relative[:2, 3]
                dyaw = R.from_matrix(pose_relative[:3, :3]).as_euler("xyz")[2]
                speed = np.linalg.norm([dx, dy]) * self.fps * 3.6  # km/h
                speeds.append(speed)
                dx_list.append(dx)
                dy_list.append(dy)
                dyaw_list.append(dyaw)

            dx_list = gaussian_filter1d(dx_list, sigma=1.0)
            dy_list = gaussian_filter1d(dy_list, sigma=1.0)
            dyaw_list = gaussian_filter1d(dyaw_list, sigma=1.0)

            # manner scene compensation
            if raw_env[bid][0]["navi_infos"]["sub_path_main_path_points"] is not None:
                subpath = raw_env[bid][0]["navi_infos"]["sub_path_main_path_points"][:, :2]
                subpath = interpolate_points(subpath, 5)
                subpath = cut_sub_path(subpath, 0, 300)  # n x 2
                subpath_cumdistance = [
                    0,
                ] + list(np.cumsum(np.linalg.norm(subpath[1:] - subpath[:-1], axis=1)))

                for pred_ts in range(0, pred_ts_num - self._history_t_num):
                    pred_polygon_center = ego_state_polygon[self._history_t_num + pred_ts].mean(axis=0)
                    sub_path_distance = np.linalg.norm(subpath - pred_polygon_center, axis=1)
                    sub_path_min_index = np.argmin(sub_path_distance)
                    pred_sub_path_min_distance = subpath_cumdistance[sub_path_min_index]

                    gt_polygon_center = ego_gt_state_polygon[
                        min(self._history_t_num + pred_ts, ego_gt_state_polygon.shape[0] - 1)
                    ].mean(axis=0)
                    sub_path_distance = np.linalg.norm(subpath - gt_polygon_center, axis=1)
                    sub_path_min_index = np.argmin(sub_path_distance)
                    gt_sub_path_min_distance = subpath_cumdistance[sub_path_min_index]

                    distance_compenstaion = pred_sub_path_min_distance - gt_sub_path_min_distance

                    manner_scene = raw_env[bid][self._history_t_num + pred_ts].get("manner_scene_info", None)
                    if manner_scene:
                        for scene in manner_scene:
                            scene['gt_distance_to_entry'] = scene["distance_to_entry"]
                            scene['gt_distance_to_exit'] = scene["distance_to_exit"]
                            scene['gt_distance_to_center'] = scene["distance_to_center"]

                            scene["distance_to_entry"] += distance_compenstaion
                            scene["distance_to_exit"] += distance_compenstaion
                            scene["distance_to_center"] += distance_compenstaion

                        # explicit assignment may not be necessary, keep it for readability
                        raw_env[bid][self._history_t_num + pred_ts]["manner_scene_info"] = manner_scene

            policy_update_output = {
                "ego_polygon": ego_state_polygon,
                "gt_polygon": ego_gt_state_polygon,
                "speeds": speeds,
                "dx_list": dx_list,
                "dy_list": dy_list,
                "dyaw_list": dyaw_list,
                "spd_limit_list": spd_limit_list,
                "iter_num": iter_num,
                # "scenario_id": raw_env[bid][0]["scenario_id"],
                "gt_mean_speed_batch": np.mean(gt_speeds),
                "scenario_id": raw_env[bid][0]["scenario_id"],
                "scenario_info": dict(),
            }
            self._calc_scenario_follow_slow(policy_update_output, raw_env[bid])
            self._calc_deadcar_bypass_scenario(policy_update_output, raw_env[bid])
            self._calc_scenario_car_queue(policy_update_output, raw_env[bid])
            params.append((policy_update_output, raw_env[bid], train_mode))  # 环境、预测信息、动作、真实信息

        return params[0]

    def _post_process(self, inputs_dict, reward_results, train_mode=True):
        """Collect reward results and compute advantages, kl."""
        pred_polygon = inputs_dict["pred_polygon"]
        gt_polygon = inputs_dict["gt_polygon"]

        batch_size = pred_polygon.shape[0]

        rewards = []
        dones = []
        results = []
        for result in reward_results:
            rewards.append(result["reward"])
            dones.append(result["dones"])
            results.append(result["final_result"])
        rewards = np.stack(rewards, axis=0)
        dones = np.stack(dones, axis=0)

        if train_mode:
            # slice to one pred trajectory
            replan_time = inputs_dict["select_replan_time"]
            pred_traj_polygon = pred_polygon[:, :, :replan_time].flatten(1, 2)
            pred_polygon = pred_traj_polygon
            device = pred_polygon.device
            rewards = torch.from_numpy(rewards).float().pin_memory().to(device, non_blocking=True)
            dones = torch.from_numpy(dones).float().pin_memory().to(device, non_blocking=True)
            results = torch.as_tensor(results, dtype=torch.long).pin_memory().to(device, non_blocking=True)

            ref_logprob = inputs_dict["ref_logprob"]
            pred_logprob = inputs_dict["pred_logprob"]
            kl = (torch.exp(ref_logprob) / torch.exp(pred_logprob) + pred_logprob - ref_logprob - 1).squeeze(-1)
            kl = torch.cat([torch.zeros((kl.shape[0], kl.shape[1], 1), device=kl.device), kl], dim=2)  # B x T+1
            kl_cpu_numpy = kl.cpu().numpy()
            for i in range(len(reward_results)):
                reward_results[i]["kl"] = kl_cpu_numpy[i]

            b = len(gt_polygon)
            pred_bs = pred_polygon.shape[0]
            assert pred_bs % b == 0
            rollout_num = pred_bs // b
            action_dones = dones[:, 1:].reshape(rollout_num, b, -1).transpose(1, 0)
            action_rewards = rewards[:, 1:].reshape(rollout_num, b, -1).transpose(1, 0)
            # compute GAE: A(s0, a0) ~ A(st-1, at-1)
            # advantages = self._calc_reward_advantage(action_rewards, action_dones)
            advantages = self._calc_gdpo_reward_advantage(reward_results, action_dones, b, rollout_num, device)
            advantages = advantages.transpose(1, 0).flatten(0, 1)

            # update loginfos
            for i in range(pred_bs):
                reward_results[i]["advantage"] = np.zeros_like(reward_results[i]["reward"])

                reward_results[i]["advantage"][1:] = advantages[i].cpu().numpy()

            if self._train_mode:
                extra_summary = self.get_extra_summary(rewards, dones, results, reward_results, inputs_dict)
            else:
                extra_summary = None

            return advantages, dones, results, rewards, kl, reward_results, extra_summary
        else:
            for bid in range(len(reward_results)):
                gt_traj = gt_polygon[bid][self._history_t_num :].mean(axis=1)
                pred_traj = pred_polygon[bid][self._history_t_num :].mean(axis=1)
                gt_mileage = np.linalg.norm(gt_traj[1:] - gt_traj[:-1], axis=1).sum()
                pred_mileage = np.linalg.norm(pred_traj[1:] - pred_traj[:-1], axis=1).sum()
                reward_results[bid]["gt_mileage"] = gt_mileage
                reward_results[bid]["pred_mileage"] = pred_mileage
            return rewards, dones, result, reward_results

    def _calc_reward_advantage(self, action_rewards, action_dones):
        action_mask = action_dones >= 0
        action_rewards = action_rewards * action_mask
        action_valid_num = action_mask.sum(axis=(-1, -2), keepdims=True)
        action_reward_mean = action_rewards.sum(axis=(-1, -2), keepdims=True) / (action_valid_num + 1e-8)
        action_reward_variance = (action_rewards - action_reward_mean) * action_mask
        # 无偏估计
        action_reward_std = (
            action_reward_variance.pow(2).sum(axis=(-1, -2), keepdims=True) / (action_valid_num - 1 + 1e-8)
        ).sqrt()
        norm_action_reward_2 = action_reward_variance / (action_reward_std + 1e-8)
        advantages = torch.cumsum(norm_action_reward_2.flip(-1), dim=-1).flip(-1)

        return advantages

    def _calc_reward_advantage_gamma(self, action_rewards, action_dones, reward_key=None):
        action_mask = action_dones >= 0
        action_rewards = action_rewards * action_mask

        norm_action_reward_2 = action_rewards
        advantages = self._apply_reward_timestep_decay(
            norm_action_reward_2,
            reward_key,
            decay_cfg=self.gdpo_reward_decay,
        )
        advantages = advantages * action_mask
        return advantages

    def _calc_gdpo_reward_advantage(self, reward_results, action_done, batchsize, rollout_num, device):
        """
        GDPO:https://arxiv.org/pdf/2601.05242
        """
        rewards_per_func = defaultdict(list)
        for item in reward_results:
            for key, value in item.items():
                if "_reward" in key[-7:] and key != "encounter_ttc_reward":
                    rewards_per_func[key].append(value)
        action_mask = action_done >= 0
        action_valid_num = action_mask.sum(axis=(-1, -2), keepdims=True)
        all_reward_advantage = []
        ## Calculate the mean and std of each reward group-wise separately
        # print(rewards_per_func.keys())
        for reward_key in rewards_per_func.keys():
            weight = self.gdpo_reward_weight[reward_key]
            reward_func_i = rewards_per_func[reward_key]
            reward_func_i = np.stack(reward_func_i, axis=0)
            reward_func_i = torch.from_numpy(reward_func_i).float().pin_memory().to(device, non_blocking=True)
            reward_func_i = reward_func_i[:, 1:].reshape(rollout_num, batchsize, -1).transpose(1, 0)
            reward_func_i = reward_func_i * action_mask
            reward_func_i_mean = reward_func_i.sum(axis=(-1, -2), keepdims=True) / (action_valid_num + 1e-8)
            reward_func_i_variance = (reward_func_i - reward_func_i_mean) * action_mask
            reward_func_i_std = (
                reward_func_i_variance.pow(2).sum(axis=(-1, -2), keepdims=True) / (action_valid_num - 1 + 1e-8)
            ).sqrt()
            norm_reward_func_i = reward_func_i_variance / (reward_func_i_std + 1e-8)
            norm_reward_func_i = norm_reward_func_i * weight
            # ===================== 改动 ：per-reward per-timestep decay =====================
            reward_adv_i = self._calc_reward_advantage_gamma(
                norm_reward_func_i,
                action_done,
                reward_key=reward_key,
            )
            all_reward_advantage.append(reward_adv_i)

        combined_reward_advantage = torch.stack(all_reward_advantage, dim=1)
        combined_reward_advantage_sum = combined_reward_advantage.nansum(dim=1)
        # GDPO batch normalization
        mean = combined_reward_advantage_sum.mean(dim=(0,1), keepdim=True)
        std = combined_reward_advantage_sum.std(dim=(0,1), keepdim=True)
        combined_reward_advantage_sum = (combined_reward_advantage_sum - mean) / (std + 1e-8)

        return combined_reward_advantage_sum

    def _apply_reward_timestep_decay(
        self,
        x: torch.Tensor,              # [B, G, T]
        reward_key: str,
        *,
        decay_cfg: dict = None,       # {reward_key: gamma}
    ) -> torch.Tensor:
        if decay_cfg is None:
            decay_cfg = getattr(self, "gdpo_reward_decay", None)
        if decay_cfg is None or reward_key not in decay_cfg:
            return torch.cumsum(x.flip(-1), dim=-1).flip(-1)

        decay_val = decay_cfg[reward_key]
        T = x.shape[-1]

        if isinstance(decay_val, (float, int)):
            gamma = float(decay_val)
            advantages = torch.zeros_like(x)
            advantages[..., -1] = x[..., -1]
            for t in range(T - 2, -1, -1):
                advantages[..., t] = x[..., t] + gamma * advantages[..., t + 1]
            return advantages


    def get_extra_summary(self, rewards, dones, results, reward_results, inputs_dict):
        """Extra summary info plot for RL training."""
        rewards_masked = torch.where(dones[:, 1:] >= 0, rewards[:, 1:], 0)
        rewards_traj = rewards_masked.sum(dim=1)
        reward_mean = rewards_traj.mean().item()

        collision_rate = (results == 2).sum().item() / len(rewards)
        tld_fault = sum(reward_result["traffic_light_reward"].sum() < 0 for reward_result in reward_results) / len(
            reward_results
        )

        n = len(reward_results) if len(reward_results) > 0 else 1
        pred_mileage = float(sum(r.get("pred_mileage", 0.0) for r in reward_results) / n)
        gt_mileage = float(sum(r.get("gt_mileage", 0.0) for r in reward_results) / n)
        pred_gt_mileage_ratio = float(sum(r.get("pred_gt_mileage_ratio", 0.0) for r in reward_results) / n)
        pred_gt_speed_ratio = float(sum(r.get("pred_gt_speed_ratio", 0.0) for r in reward_results) / n)

        extra_summary = {
            "reward_mean": reward_mean,
            "collision_rate": collision_rate,
            "tld_fault": tld_fault,
            "pred_mileage": pred_mileage,
            "gt_mileage": gt_mileage,
            "pred_gt_mileage_ratio": pred_gt_mileage_ratio,
            "pred_gt_speed_ratio": pred_gt_speed_ratio,
        }

        pred_polygon = inputs_dict["pred_polygon"]
        gt_polygon = inputs_dict["gt_polygon"]
        pred_logprob = inputs_dict["pred_logprob"]
        batch_size_with_rollout = pred_polygon.shape[0]
        batch_size = gt_polygon.shape[0]
        rollout_num = batch_size_with_rollout // batch_size

        pred_logprob = pred_logprob.reshape(rollout_num, batch_size, -1).transpose(1, 0)
        top1_indices = pred_logprob.sum(-1).argmax(-1, keepdim=True)

        dones = dones[..., 1:].reshape(rollout_num, batch_size, -1).transpose(1, 0)
        mask = dones >= 0

        for key in reward_results[-1].keys():
            if key.endswith("_reward"):
                sub_reward = np.concatenate([reward_result[key][None] for reward_result in reward_results])

                sub_reward = (
                    torch.from_numpy(sub_reward[..., 1:]).reshape(rollout_num, batch_size, -1).transpose(1, 0).to(dones)
                )
                sub_reward_masked = sub_reward * mask
                sub_reward_sum = sub_reward_masked.sum(-1)
                sub_reward_std = sub_reward_sum.std(-1)
                sub_reward_top = sub_reward_sum.gather(-1, top1_indices)
                sub_reward_sparsity = ((sub_reward_masked != 0).sum(-1)) / (mask.sum(-1) + 1e-8)
                sub_reward_adv = self._calc_reward_advantage(sub_reward, mask).mean(-1)
                sub_reward_adv_top = sub_reward_adv.gather(-1, top1_indices)

                extra_summary.update(
                    {
                        f"{key}_mean": sub_reward_sum.mean().item(),
                        f"{key}_top": sub_reward_top.mean().item(),
                        f"{key}_std": sub_reward_std.mean().item(),
                        f"{key}_sparsity": sub_reward_sparsity.mean().item(),
                        f"{key}_adv": sub_reward_adv_top.mean().item(),
                        f"{key}_max": sub_reward.max().item(),
                        f"{key}_min": sub_reward.min().item(),
                    }
                )

        return extra_summary

    def close(self):
        """关闭多进程池，释放资源."""
        if self.proc_pool is not None:
            self.proc_pool.close()
            self.proc_pool.join()
            self.proc_pool = None

    def __del__(self):
        """析构函数，确保进程池被正确关闭."""
        self.close()

    # multiprocessing 兼容：避免在序列化 self 时把进程池一并 pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        # 进程池对象不可被 pickle，序列化时剔除
        state["proc_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 反序列化后不自动重建进程池，由调用方按需创建
        self.proc_pool = None

    def split_inputs_by_gt_batch(self, inputs_dict, train_mode = True):
        gt_polygon = inputs_dict["gt_polygon"]
        gt_propertys = inputs_dict.get("gt_propertys", None)
        pred_polygon = inputs_dict["pred_polygon"]
        raw_env = inputs_dict["raw_env"]
        gt_bs = gt_polygon.shape[0]
        pred_bs = pred_polygon.shape[0]
        if train_mode:
            select_replan_time = inputs_dict["select_replan_time"]
            pred_polygon_np = pred_polygon[:, :, :select_replan_time].flatten(1, 2).cpu().numpy()
            gt_polygon_np = gt_polygon.cpu().numpy()
            if gt_propertys is not None:
                gt_propertys_np = gt_propertys.cpu().numpy()
        else:
            pred_polygon_np = pred_polygon
            gt_polygon_np = gt_polygon

        out_list = []

        for b in range(pred_bs):
            one = {}

            one["gt_polygon"] = gt_polygon_np[b % gt_bs:b % gt_bs+1]
            if gt_propertys is not None:
                one["gt_propertys"] = gt_propertys_np[b % gt_bs:b % gt_bs+1]

            one["pred_polygon"] = pred_polygon_np[b:b+1]
            one["raw_env"] = raw_env[b:b+1]
            for k, v in inputs_dict.items():
                if k in one:
                    continue
                if k in {
                    "pred_logprob", "ref_logprob"
                }:
                    continue
                one[k] = v

            out_list.append(one)

        return out_list


    def forward(self, inputs_dict, train_mode=True):
        """Forward pass to compute the total reward for trajectory planning.

        Args:
            inputs_dict (dict): Output from rollout result, incudeing trajectories and environments.
                Required keys:
                - 'pred_polygon':
                - 'gt_polygon':
                - 'gt_property':
                - 'raw_env':
                - 'state_range':
                - 'closedloop_simu_time':
                - 'select_replan_time':
        Returns:

        """

        if self._stopwatch:
            self._stopwatch.tic("rl_reward_model/pre_process")

        batch_input_dicts = self.split_inputs_by_gt_batch(inputs_dict,train_mode = train_mode)
        args = []
        for inp in batch_input_dicts:
            args.append((inp, train_mode))
        if self.multi_process and self.proc_pool is not None:
            batch_reward_params = self.proc_pool.starmap(self._pre_process, args)
        else:
            batch_reward_params = [self._pre_process(*param) for param in args]  # single process

        if self._stopwatch:
            self._stopwatch.record_time_cost("rl_reward_model/pre_process", "rl_reward_model/calc_rewards")


        if self.multi_process and self.proc_pool is not None:
            results = self.proc_pool.starmap(self._calc_rewards, batch_reward_params)
        else:
            results = [self._calc_rewards(*param) for param in batch_reward_params]  # single process

        if self._stopwatch:
            self._stopwatch.record_time_cost("rl_reward_model/calc_rewards", None)

        reward_results_list = []
        timing_list = []

        for reward_res, timing in results:
            reward_results_list.append(reward_res)
            timing_list.append(timing)

        # ===== 聚合 reward timing =====
        self._aggregate_reward_timing(timing_list)

        if self._stopwatch:
            self._stopwatch.tic("rl_reward_model/post_process")

        outputs = self._post_process(inputs_dict, reward_results_list, train_mode=train_mode)

        if self._stopwatch:
            self._stopwatch.record_time_cost("rl_reward_model/post_process", None)

        return outputs
