import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Deep Q-Learning')

    parser.add_argument('--algorithm', type=str, default="")
    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')

    # Training Arguments
    parser.add_argument('--max-steps', type=int, default=50000000, metavar='STEPS',
                        help='Number of steps to train (equal to actual_frames / 4)')
    parser.add_argument('--buffer-size', type=int, default=1000000, metavar='CAPACITY',
                        help='Maximum memory buffer size')
    parser.add_argument('--randomly-discard-experience', action='store_true',
                        help='Randomly discard half of collected experience data')
    parser.add_argument('--randomly-replace-memory', action='store_true',
                        help='Randomly replace old experiences by new experiences when the memory replay is full (by default it is first-in-first-out)')
    parser.add_argument('--update-target', type=int, default=8000, metavar='STEPS',
                        help='Interval of target network update')
    parser.add_argument('--train-freq', type=int, default=4, metavar='STEPS',
                        help='Number of steps between optimization steps')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Discount factor')
    parser.add_argument('--learning-start', type=int, default=50000, metavar='N',
                        help='How many steps of the model to collect transitions for before learning starts')
    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Start value of epsilon')
    parser.add_argument('--eps-mid', type=float, default=0.1,
                        help='Mid value of epsilon (at one million-th step)')
    parser.add_argument('--eps-final', type=float, default=0.01,
                        help='Final value of epsilon')

    # Algorithm Arguments
    parser.add_argument('--double', action='store_true',
                        help='Enable Double Q Learning')
    parser.add_argument('--dueling', action='store_true',
                        help='Enable Dueling Network with Default Evaluation (Avg.) on Advantages')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Enable prioritized experience replay')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Alpha value for prioritized replay')
    parser.add_argument('--ratio-min-prio', type=int, default=10,
                        help='Allowed maximal ratio between the smallest sampling priority and the average priority, which equals the maximal importance sampling weight')
    parser.add_argument('--prio-eps', type=float, default=1e-10,
                        help='A small number added before computing the priority')
    parser.add_argument('--beta-start', type=float, default=0.4,
                        help='Start value of beta for prioritized replay')
    parser.add_argument('--beta-frames', type=float, default=50000000,
                        help='End step of beta schedule for prioritized replay')
    parser.add_argument('--IS-weight-only-smaller', action='store_true',
                        help='Divide all importance sampling correction weights by the largest weight, so that they are smaller or equal to one')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='Environment Name')
    parser.add_argument('--episode-life', type=int, default=1,
                        help="Whether losing one life is considered as an end of an episode(1) or not(0) from the agent's perspective")
    parser.add_argument('--grey', type=int, default=1,
                        help='Change the observation to greyscale (default 1)')
    parser.add_argument('--frame-stack', type=str, default="4",
                        help='Number of adjacent observations to stack')
    parser.add_argument('--frame-downscale', type=int, default=84, # we will always crop the frame when it has a height of 250 instead of the default 210 (crop by top 28, bottom 12) 
                        help='Downscaling ratio of the frame observation (if <= 10) or image size as the downscaling target (if >10)')
    parser.add_argument('--max-episode-steps', type=int, default=20000,
                        help="The maximum number of steps allowd before resetting the real episode.")

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, nargs="+",
                        help='Pretrained model names to load (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--num-trial', type=int, default=400)
    parser.add_argument('--evaluation_interval', type=int, default=10000,
                        help='Steps for printing statistics')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=6.25e-5, metavar='η',
                        help='Learning rate')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optimizer')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4,
                        help='Epsilon of adam optimizer')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU to use')
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--grad-clip', type=float, default=10., # when transform-Q is used, it should be 40.
                        help="Gradient clipping norm; 0 corresponds to no gradient clipping")

    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--comment', type=str, default="")

    # A simple option to reproduce the original DQN
    parser.add_argument('--originalDQN', action='store_true', help="To reproduce the original DQN")

    parser.add_argument('--save-best', action='store_true', help="To save the model when it performs best during training, averaged over 40 episodes")

    args = parser.parse_args()

    if args.originalDQN:
        del args.originalDQN
        # the most important arguments for reproducing the original prioritized dueling DDQN
        args.algorithm = "DQN"
        args.lr = 6.25e-5
        args.adam_eps = 1.5e-4
        args.grad_clip = 10.
        args.prioritized_replay = True 
        args.alpha = 0.6
        args.beta_start = 0.4
        args.beta_frames = 50000000.
        args.auto_init = False
        args.gamma = 0.99
        args.double = True 
        args.dueling = True 
        args.episode_life = 1
        args.randomly_replace_memory = False
        args.no_clip = False
        args.transform_Q = False

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu_id) if args.cuda else "cpu")

    return args
