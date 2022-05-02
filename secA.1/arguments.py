import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Deep Q-Learning')

    parser.add_argument('--alg', '--algorithm', choices=["RG", "DQN", "CDQN"], default="CDQN")
    # Basic Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='Number of complete runs to repeat')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training')

    # Training Arguments
    parser.add_argument('--dataset-size', type=int, default=20000,
                        help='Number of transitions to collect for training')
    parser.add_argument('--epochs', type=int, default=2000)
    #parser.add_argument('--update-target', type=int, default=1000,
    #                    help='Interval of target network update')
    parser.add_argument('--gamma', type=float, default=0.97, metavar='γ',
                        help='Discount factor')

    # Environment Arguments
    parser.add_argument('--env', type=str, default='wetchicken1d',
                        help='Environment Name')
    parser.add_argument('--max-episode-steps', type=int, default=3000,
                        help="The maximum number of steps allowd before resetting the real episode.")

    # Evaluation Arguments
    parser.add_argument('--load-model', type=str, nargs="+",
                        help='Pretrained model names to load (state dict)')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate only')
    parser.add_argument('--num-trial', type=int, default=200, help = "The number of test episodes to evaluate the performance")

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=1e-3, metavar='η',
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size')
    parser.add_argument('--adam-eps', type=float, default=1e-8,
                        help='Epsilon of adam optimizer')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='Which GPU to use')
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--beta1', type=float, default=0.9)

    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--comment', type=str, default="")

    parser.add_argument('--no-test', action='store_true', help="Do not evaluate performance")

    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu_id) if args.cuda else "cpu")

    return args
