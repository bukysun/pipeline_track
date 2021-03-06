from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os, sys
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from common.statistics import stats

# Sample one trajectory (until trajectory end)
def traj_episode_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode

    time.sleep(0.1)
    ob = env.reset()
    state = pi.get_initial_state()
    time.sleep(0.1)
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode

    # Initialize history arrays
    obs = []; rews = []; news = []; acs = []

    while True:
        prevac = ac
        if pi.recurrent:
            ac, vpred, state = pi.act(stochastic, ob, state)
        else:
            ac, vpred = pi.act(stochastic, ob)
        
        obs.append(ob)
        news.append(new)
        acs.append(ac)
        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if t > 0 and new: #or t % horizon == 0):
            # convert list into numpy array
            #obs = np.array(obs)
            rews = np.array(rews)
            news = np.array(news)
            acs = np.array(acs)
            yield {"ob":obs, "rew":rews, "new":news, "ac":acs,
                    "ep_ret":cur_ep_ret, "ep_len":cur_ep_len}
            time.sleep(0.1)
            ob = env.reset()
            state = pi.get_initial_state()
            time.sleep(0.1)
            #print(env.unwrapped.target_heading)
            cur_ep_ret = 0; cur_ep_len = 0; t = 0

            # Initialize history arrays
            obs = []; rews = []; news = []; acs = []
        t += 1


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    time.sleep(0.1)
    ob = env.reset()
    state = pi.get_initial_state()
    time.sleep(0.1)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    if isinstance(ob, tuple):
        obs = [ob for _ in range(horizon)]
    else:
        obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    states = []
    if pi.recurrent:
        states = [state for _ in range(horizon)]

    while True:
        prevac = ac
        if pi.recurrent:
            ac, vpred, state_out = pi.act(stochastic, ob, state)
        else:
            ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "state": states}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        if pi.recurrent:
            states[i] = state
            state = state_out

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
           
            # delay for ros to reset the environment
            time.sleep(0.1)
            ob = env.reset()
            state = pi.get_initial_state()
            time.sleep(0.1)
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        save_per_iter=50, max_sample_traj = 10, 
        ckpt_dir=None, log_dir=None, task_name = "origin",
        sample_stochastic=True, load_model_path=None, task="train"
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed clipping parameter epsilon

    ob_p = U.get_placeholder_cached(name="ob_physics")
    ob_f = U.get_placeholder_cached(name="ob_frames")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    if pi.recurrent:
        state_c = U.get_placeholder_cached(name="state_c")
        state_h = U.get_placeholder_cached(name="state_h")
        lossandgrad = U.function([ob_p, ob_f, state_c, state_h, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
        compute_losses = U.function([ob_p, ob_f, state_c, state_h, ac,  atarg, ret, lrmult], losses)
    else:
        lossandgrad = U.function([ob_p, ob_f, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
        compute_losses = U.function([ob_p, ob_f, ac, atarg, ret, lrmult], losses)

    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    
    writer = U.FileWriter(log_dir)
    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)
    traj_gen = traj_episode_generator(pi, env, timesteps_per_actorbatch, stochastic=sample_stochastic)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    loss_stats = stats(loss_names)
    ep_stats = stats(["Reward", "Episode_Length", "Episode_This_Iter"])
    if task == "sample_trajectory":
        sample_trajectory(load_model_path, max_sample_traj, traj_gen, task_name, sample_stochastic)
        sys.exit()


    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)

        logger.log2("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.next()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob_p, ob_f = zip(*ob)
        ob_p = np.array(ob_p)
        ob_f = np.array(ob_f)
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        
        
        if pi.recurrent:
            sc, sh = zip(*seg["state"])
            d = Dataset(dict(ob_p=ob_p, ob_f=ob_f, ac=ac, atarg=atarg, vtarg=tdlamret, state_c=np.array(sc), state_h=np.array(sh)), shuffle=not pi.recurrent)
        else:
            d = Dataset(dict(ob_p=ob_p, ob_f=ob_f, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob_p) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log2("Optimizing...")
        logger.log2(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                if pi.recurrent:
                    lg = lossandgrad(batch["ob_p"], batch["ob_f"], batch["state_c"], batch["state_h"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                else:
                    lg = lossandgrad(batch["ob_p"], batch["ob_f"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                newlosses = lg[:-1]
                g = lg[-1]
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log2(fmt_row(13, np.mean(losses, axis=0)))

        logger.log2("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            if pi.recurrent:
                newlosses = compute_losses(batch["ob_p"], batch["ob_f"], batch["state_c"], batch["state_h"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            else:
                newlosses = compute_losses(batch["ob_p"], batch["ob_f"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log2(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
            loss_stats.add_all_summary(writer, meanlosses, iters_so_far)
            ep_stats.add_all_summary(writer, [np.mean(rewbuffer), np.mean(lenbuffer), len(lens)], iters_so_far)

    return pi

def sample_trajectory(load_model_path, max_sample_traj, traj_gen, task_name, sample_stochastic):

    assert load_model_path is not None
    U.load_state(load_model_path)
    sample_trajs = []
    for iters_so_far in range(max_sample_traj):
        logger.log2("********** Iteration %i ************"%iters_so_far)
        traj = traj_gen.next()
        ob, new, ep_ret, ac, rew, ep_len = traj['ob'], traj['new'], traj['ep_ret'], traj['ac'], traj['rew'], traj['ep_len']
        logger.record_tabular("ep_ret", ep_ret)
        logger.record_tabular("ep_len", ep_len)
        logger.record_tabular("immediate reward", np.mean(rew))
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
        traj_data = {"ob":ob, "ac":ac, "rew": rew, "ep_ret":ep_ret}
        sample_trajs.append(traj_data)

    sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
    logger.log2("Average total return: %f"%(sum(sample_ep_rets)/len(sample_ep_rets)))
 
def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
