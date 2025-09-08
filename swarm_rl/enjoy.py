import sys

from sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
import numpy as np

def main():
    """Script entry point."""
    render = True
    register_swarm_components()
    cfg = parse_swarm_cfg(argv=sys.argv[1:] + ["--seed=1"], evaluation=True)
    # cfg = parse_swarm_cfg(evaluation=True)
    # cfg.cli_args["quads_num_agents"] = 8
    np.random.seed(1)
    cfg.test = True
    if render:
        cfg.max_num_episodes = 3
        cfg.save_video = True
        cfg.cli_args["quads_render"] = True
        cfg.mode_index = 7
    else:
        cfg.max_num_episodes = 128
        cfg.save_video = False
        cfg.cli_args["quads_render"] = False
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
