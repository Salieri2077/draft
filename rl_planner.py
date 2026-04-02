# -*- coding: utf-8 -*-
from tpp_onemodel.model.loss.plannn2_rl_loss import EgoBCLoss
from tpp_onemodel.model.loss.plannn2_rl_loss import GRPORLLoss
# from tpp_onemodel.model.loss.plannn2_rl_loss import TLSBCLoss
from tpp_onemodel.model.loss.plannn2_rl_loss import RLPlannerLoss
from tpp_onemodel.model.loss.reward_loss import AccelerationReward
from tpp_onemodel.model.loss.reward_loss import JerkReward
from tpp_onemodel.model.loss.reward_loss import CollisionReward
from tpp_onemodel.model.loss.reward_loss import ContinuousLaneChangeReward
from tpp_onemodel.model.loss.reward_loss import CrossSolidLineReward
from tpp_onemodel.model.loss.reward_loss import CentralizationReward
from tpp_onemodel.model.loss.reward_loss import EtcSpeedReward
from tpp_onemodel.model.loss.reward_loss import EtcTakeOffReward
from tpp_onemodel.model.loss.reward_loss import EtcMindistReward
from tpp_onemodel.model.loss.reward_loss import MinDistReward
from tpp_onemodel.model.loss.reward_loss import NaviLaneReward
from tpp_onemodel.model.loss.reward_loss import NaviReward
from tpp_onemodel.model.loss.reward_loss import ProgressReward
from tpp_onemodel.model.loss.reward_loss import SpeedLimitReward
from tpp_onemodel.model.loss.reward_loss import TrafficLightReward
from tpp_onemodel.model.loss.reward_loss import TTCReward
from tpp_onemodel.model.loss.reward_loss import VelocityReward
from tpp_onemodel.model.loss.reward_loss import VirtualWallReward
from tpp_onemodel.model.loss.reward_loss import GateMachineReward
from tpp_onemodel.model.loss.reward_loss import WrongWayReward
from tpp_onemodel.model.loss.reward_loss import DangerLaneChangeReward
from tpp_onemodel.model.loss.reward_loss import ToggleReward
from tpp_onemodel.model.loss.reward_loss import HumanoidReward
from tpp_onemodel.model.loss.reward_loss import HumanoidNudgeReward
from tpp_onemodel.model.loss.reward_loss import ChooseEtcReward
from tpp_onemodel.model.module.rl_planner_reward_model import RLPlannerRewardModel
from tpp_onemodel.model.loss.reward_loss import SlowFollowReward
from tpp_onemodel.model.loss.reward_loss import JunctionLaneSelectReward


def create_reward_model_cfg(
    history_t_num=15,
    pred_t_num=25,
):
    """Create action model config."""
    reward_model_cfg = dict(
        type=RLPlannerRewardModel,
        history_t_num=history_t_num,
        pred_t_num=pred_t_num,
        summary_cfg=dict(
            dt=0.2,
            fine_tune=False,
            fps=5,
            rewards_types_cfg=dict(
                cross_solid_line_reward=dict(
                    func=dict(
                        type=CrossSolidLineReward,
                    )
                ),
                danger_lc_reward=dict(
                    func=dict(
                        type=DangerLaneChangeReward,
                    )
                ),
                collision_reward=dict(
                    func=dict(
                        type=CollisionReward,
                    )
                ),
                traffic_light_reward=dict(
                    func=dict(
                        type=TrafficLightReward,
                    )
                ),
                navi_lane_reward=dict(
                    func=dict(
                        type=NaviLaneReward,
                    )
                ),
                virtual_wall_reward=dict(
                    func=dict(
                        type=VirtualWallReward,
                    )
                ),
                speed_limit_reward=dict(
                    func=dict(
                        type=SpeedLimitReward,
                    )
                ),
                ttc_reward=dict(
                    func=dict(
                        type=TTCReward,
                    )
                ),
                acc_reward=dict(
                    func=dict(
                        type=AccelerationReward,
                        fps=5,
                    )
                ),
                jerk_reward=dict(
                    func=dict(
                        type=JerkReward,
                        fps=5,
                    )
                ),
                min_distance_reward=dict(
                    func=dict(
                        type=MinDistReward,
                    )
                ),
                navi_reward=dict(
                    func=dict(
                        type=NaviReward,
                    )
                ),
                progress_reward=dict(
                    func=dict(
                        type=ProgressReward,
                    )
                ),
                wrong_way_reward=dict(
                    func=dict(
                        type=WrongWayReward,
                    )
                ),
                centralization_reward=dict(
                    func=dict(
                        type=CentralizationReward,
                    )
                ),
                continuous_lane_change_reward=dict(
                    func=dict(
                        type=ContinuousLaneChangeReward,
                    )
                ),
                toggle_reward=dict(
                    func=dict(
                        type=ToggleReward,
                    ),
                ),
                # velocity_reward=dict(
                #     func=dict(
                #         type=VelocityReward,
                #     )
                # ),
                humanoid_reward=dict(
                    func=dict(
                        type=HumanoidReward,
                    )
                ),
                gate_machine_reward=dict(
                    func=dict(
                        type=GateMachineReward,
                    )
                ),
                humanoid_nudge_reward=dict(
                    func=dict(
                        type=HumanoidNudgeReward,
                    )
                ),
                slow_follow_reward=dict(
                    func=dict(
                        type=SlowFollowReward,
                        linear_penalty_cfg=dict(
                            vru_max_penalty=2.0,
                            vru_min_penalty=1.0,
                            vru_penalty_interval=1,
                            veh_max_penalty=6.0,
                            veh_min_penalty=4.0,
                            veh_penalty_interval=1,
                        ),
                    )
                ),
                junction_lane_select_reward=dict(
                    func=dict(
                        type=JunctionLaneSelectReward,
                    )
                ),
            ),
            rewards_weights_cfg=dict(
                collision_reward=1.0,
                ttc_reward=1.0,
                min_distance_reward=1.0,
                cross_solid_line_reward=1.0,
                junction_lane_select_reward=10.0,
            ),
        ),
        gdpo_reward_weight = dict(
            cross_solid_line_reward=1,
            danger_lc_reward=1,
            collision_reward=1,
            traffic_light_reward=1,
            navi_lane_reward=1,
            navi_lane_change_reward=1,
            virtual_wall_reward=1,
            speed_limit_reward=1,
            ttc_reward=1,
            acc_reward=1,
            jerk_reward=1,
            min_distance_reward=1,
            navi_reward=1,
            progress_reward=1,
            wrong_way_reward=1,
            centralization_reward=1,
            continuous_lane_change_reward=1,
            toggle_reward=1,
            humanoid_reward=1,
            gate_machine_reward=1,
            humanoid_nudge_reward=1,
            slow_follow_reward=1,
            junction_lane_select_reward=1,
        ),
        gdpo_reward_decay = dict(
            cross_solid_line_reward=0.9, # 25step, 5s, 10%
            danger_lc_reward=0.95, # 25step, 5s, 10%
            collision_reward=0.99, # 20step, 4s, 10%
            traffic_light_reward=0.9, # 25step, 5s, 10% no use
            navi_lane_reward=0.99, # 120step, 24s, 50%
            navi_lane_change_reward=0.9, # 25step, 5s, 10%
            virtual_wall_reward=0.99, # 120step, 24s, 50%
            speed_limit_reward=0.9, # 25step, 5s, 10%
            ttc_reward=0.99, # 20step, 4s, 10%
            acc_reward=0.90, # 15step, 3s, 10%
            jerk_reward=0.9, # 15step, 3s, 10%,
            min_distance_reward=0.95, # 20step, 4s, 10%
            navi_reward=0.99, # 120step, 24s, 50%
            progress_reward=0.97, # 25step, 5s, 10%
            wrong_way_reward=0.99, # 120step, 24s, 50%
            centralization_reward=0.95, # 50step, 10s, 10%,
            continuous_lane_change_reward=0.96, # 50step, 10s, 10%,
            toggle_reward=0.95, # 50step, 10s, 10%,
            humanoid_reward=0.95, # 50step, 10s, 10%,
            gate_machine_reward=0.95, # 50step, 10s, 10%,
            humanoid_nudge_reward=0.95, # 50step, 10s, 10%,
            slow_follow_reward=0.97, # 50step, 10s, 10%,
            junction_lane_select_reward=0.95, # 50step, 10s, 10%,
        ),
        multi_process=True,
        reward_extra_cfg=dict(centralization_reward=dict(kFollowSlow=dict(clamp_params=dict(min=0, max=0)))),
    )
    return reward_model_cfg


def create_rl_planner_loss_cfg(future_steps=25, grpo_weight_decay=0.92):
    """Create rl planner loss config."""
    rl_planner_loss_cfg = dict(
        type=RLPlannerLoss,
        future_steps=future_steps,
        loss_func_cfg=dict(
            pg_loss=dict(
                type=GRPORLLoss,
                weight_decay=grpo_weight_decay,
                future_steps=future_steps,
                clip_sigma=0.1,
            ),
            ego_kl_loss=dict(
                type=EgoBCLoss,
            ),
            # tls_kl_loss=dict(
            #     type=TLSBCLoss,
            # ),
        ),
        loss_scales_cfg=dict(pg_loss=1.0, ego_kl_loss=0.2, tls_kl_loss=0.2),
    )
    return rl_planner_loss_cfg

def get_dataset():
    import os
    expert_data=[
        "/share-global/shaoqian.li/plannn2/dataset/gt_good_data/1023/good_case.txt",
    ]
    failure_data=[
        # "/share-global/yuxiang.yan/share/reward_eval/bad_case.txt",
    ]

    if os.getenv("EVAL_GT_REWARDS"):
        if expert_data_path := os.getenv("EXPERT_DATA_PATH"):
            expert_data = [expert_data_path]
        if failure_data_path := os.getenv("FAILURE_DATA_PATH"):
            failure_data = [failure_data_path]
        return dict(
            expert_data=expert_data,
            failure_data=failure_data,
        )

    return dict(
        expert_data=expert_data,
        failure_data=failure_data,
    )


def create_rl_planner_reward_eval_cfg():
    rl_planner_reward_eval_cfg = dict(
        eval_gt_lower_only_cross=False,
        gt_env_enable=True,
        eval_rewards_mode=dict(
            agent_collision=dict(
                dataset=get_dataset(),
                # reward_types=None,
                reward_list=[
                    dict(
                        type="collision_reward",
                        property="penalty",  # "bonus"：奖励类， "penalty"：惩罚类
                    ),
                    dict(
                        type="ttc_reward",
                        property="penalty",
                    ),
                    dict(
                        type="min_distance_reward",
                        property="penalty",
                    ),
                ],
            ),
            # lane_keeping=dict(
            #     dataset=dict(
            #         expert_data=[],
            #         failure_data=[],
            #     ),
            #     reward_list=[
            # dict(
            #     type="lane_keeping_reward",
            #     property="penalty", # "bonus"：奖励类， "penalty"：惩罚类
            # ),
            #     ],
            # ),
        ),
    )
    return rl_planner_reward_eval_cfg
