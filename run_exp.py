from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from env.env_util import make_env


def train(args):
    from algo import pposgd_simple
    from nn import cnn_policy, cnn_lstm_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()

    task_name = "ppo." + args.taskname + "." + args.env_id.split("-")[0]  +".seed_" + ("%d"%args.seed) 
    args.log_dir = osp.join(args.log_dir, task_name)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank() if args.seed is not None else None
    set_global_seeds(workerseed)
    env = make_env(args.env_id, seed=args.seed, frame_stack=False)()
    def policy_fn(name, ob_space, ac_space):#pylint: disable=W0613
        return cnn_policy.CnnPolicy(name, ob_space, ac_space, hid_size=64, num_hid_layers=1)
        #return cnn_lstm_policy.CnnSenLSTMPolicy(name, ob_space, ac_space, hid_size=64, num_hid_layers = 1)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(args.num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear', save_per_iter=100, 
        ckpt_dir=args.checkpoint_dir, log_dir=args.log_dir, task_name=task_name,
        task=args.task, load_model_path=args.load_model_path, sample_stochastic=True
    )
    env.close()
 

def main():
    import argparse
    import rospy
    from env.ros_utils import launch_from_py
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env_id", type=str, default="PipelineTrack-v1")
    parser.add_argument("--num_timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='/home/uwsim/workspace/results/pipeline_track/checkpoint')
    parser.add_argument('--log_dir', help='the directory to save plotting data', default='/home/uwsim/workspace/results/pipeline_track/log_dir') 
    parser.add_argument('--taskname', help='name of task', type=str, default="origin") 
    parser.add_argument('--task', help="train or sample trajectory", type=str, default="train")
    parser.add_argument('--load_model_path', type=str, default=None)

    args = parser.parse_args()

    #launch ros node
    launch = launch_from_py("auv", "/home/uwsim/uwsim_ws/install_isolated/share/RL/launch/basic.launch")
    launch.start()
    rospy.loginfo("auv started!")
    rospy.sleep(10)

    train(args)
    launch.shutdown()

if __name__ == "__main__":
    main()
