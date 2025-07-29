import sys

from sample_factory.enjoy import enjoy

from swarm_rl.train import parse_swarm_cfg, register_swarm_components
import numpy as np

def main():
    """Script entry point."""
    register_swarm_components()
    cfg = parse_swarm_cfg(argv=sys.argv[1:] + ["--seed=1"], evaluation=True)
    # cfg = parse_swarm_cfg(evaluation=True)
    cfg.max_num_episodes = 3
    np.random.seed(0)
    cfg.cli_args["quads_num_agents"] = 8
    # cfg.save_video = True
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
