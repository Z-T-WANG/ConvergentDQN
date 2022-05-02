#!/usr/bin/env python3.6
import torch
from common.utils import set_global_seeds
from common.wrappers import make_atari, wrap_atari_dqn
from arguments import get_args
from test import test
import sys, datetime

def main():
    torch.set_num_threads(2) # we need to constrain the number of threads; it can default to a large value
    args = get_args()

    # keep a history of the commands that has been executed
    with open("commandl_history.txt", "a") as f:
        f.write(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S\t")+' '.join(sys.argv)+"\n")

    env = make_atari(args.env, args.max_episode_steps, clip_reward = (not (args.no_clip or args.transform_Q)) and (not args.evaluate))
    env = wrap_atari_dqn(env, args)

    set_global_seeds(args.seed)
    env.seed(args.seed)

    if args.evaluate:
        test(env, args)
        env.close()
        return

    from train import train as train
    train(env, args)

    env.close()


if __name__ == "__main__":
    main()
