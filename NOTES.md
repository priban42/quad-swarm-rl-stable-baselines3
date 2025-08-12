original remote: https://github.com/Zhehui-Huang/quad-swarm-rl

micromamba activate swarm-rl

{topdown,chase,side,global,corner0,corner1,corner2,corner3,topdownfollow}

test multiple 4:
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_neighbor_pos_4_ --experiment=00_quad_neighbor_pos_4_q.c.rew_5.0 --quads_view_mode global

train multiple 3:
python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_modified --max_parallel=4 --pause_between=1 --experiments_per_gpu=4 --num_gpus=1

test multiple 3:
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_baseline_4_ --experiment=00_quad_baseline_4_q.c.rew_5.0 --quads_view_mode global

train multiple 2:
python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline_attn_8 --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1

test multiple 2:
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=./train_dir/test_anneal_20250715_1206/quad_mix_baseline-8_mixed_attn_ --experiment=00_quad_mix_baseline-8_mixed_attn_see_0 --quads_view_mode global

train multiple:
python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_modified --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1

test multiple:
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=./train_dir/quads_multi_mix_baseline_8a_local_v116/quad_mix_baseline-8_mixed_ --experiment=00_quad_mix_baseline-8_mixed_q.c.rew_5.0 --quads_view_mode global

train single:
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1

test single:
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=./train_dir/paper_quads_multi_mix_baseline_8a_attn_v116/single_ --experiment=00_single_see_0 --quads_view_mode global