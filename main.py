import argparse, os
import wandb
import gym
import numpy as np
# import agents as Agents
import agent_per as Agents
from utils2 import plot_learning_curve, make_env
from components import components
from values_slr import slr
from values_surge import surge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Deep Q Learning: From Paper to Code')

    # Define environment-specific parameters
    parser.add_argument('-b1', type=float, default=1.0, help='Base height for Region 1 lower dike')
    # Dike in region 2 are placed adjacent to dike in region 1 with their top height matching to prevent lateral flow
    parser.add_argument('-b2', type=float, default=3.25, help='Base height for Region 1 higher dike')
    # Base elevation of region 2 is hard coded to be 0.75 m higher than region 1
    parser.add_argument('-r_1_h0', type=float, default=0.0, help='Base elevation for Region 1')

    # the hyphen makes the argument optional
    parser.add_argument('-n_games', type=int, default=30000,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.02,
                        help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.90,
                        help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=500000,  # ~13Gb
                        help='Maximum size for memory replay buffer')
    # parser.add_argument('-repeat', type=int, default=4,
    #                         help='Number of frames to repeat & stack')
    parser.add_argument('-bs', type=int, default=128,
                        help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=20,
                        help='interval for replacing target network')
    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='models/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DDQNAgent',
                        help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    # parser.add_argument('-clip_rewards', type=bool, default=False,
    #                     help='Clip rewards to range -1 to 1')
    parser.add_argument('-no_ops', type=int, default=0,
                        help='Max number of no ops for testing')
    # parser.add_argument('-fire_first', type=bool, default=False,
    #                     help='Set first action of episode to fire')
    parser.add_argument('-climate_ssp', type=str, default='245',
                        help='119/245/585')
    parser.add_argument('-env', type=str, default='environment_2_regions_4_dikes_elevated',
                        help='Name of the file defining the environment')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Initialize environment-specific variables
    env_variant = f"Env_b1_{args.b1}_b2_{args.b2}_r1h0_{args.r_1_h0}"
    results_folder = f"results/{env_variant}"
    os.makedirs(results_folder, exist_ok=True)

    wandb.init(
        # set the wandb project where this run will be logged
        project="coastal_city_DDQN",
        name=env_variant,
        # track hyperparameters and run metadata
        config={
            "environment": env_variant,
            "learning_rate": 0.0001,
            "main_net_update": 1,
            "target_net_update": 20,
            "max_episodes": 20000,
        }
    )
    # env = make_env(env_name=args.env, climate_model=args.climate_ssp)
    env = make_env(env_name=args.env, climate_model=args.climate_ssp, b1=args.b1, b2=args.b2, r_1_h0=args.r_1_h0)

    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                   epsilon=args.eps,
                   lr=args.lr,
                   input_dims=env.observation_space.n+1+4,
                   n_actions=len(env.action_space),
                   mem_size=args.max_mem,
                   eps_min=args.eps_min,
                   batch_size=args.bs,
                   replace=args.replace,
                   eps_dec=args.eps_dec,
                   chkpt_dir=args.path,
                   algo=args.algo,
                   env_name=env_variant)

    if args.load_checkpoint:
        # print(args.load_checkpoint)
        agent.load_models()

    # Create a log file name using environment and save it to the path defined in args.path
    log_file_path = os.path.join(results_folder, f"evaluation_policy_log.txt")
    log_file = open(log_file_path, "w")
    figure_file = os.path.join(results_folder, f"plot.png")
    scores_file = os.path.join(results_folder, f"scores.npy")

    # log_file_path = os.path.join(args.path, f"{args.env}_evaluation_policy_log.txt")
    #
    # # Open the log file for writing
    # log_file = open(log_file_path, "w")
    #
    # fname = args.algo + '_' + args.env + '_alpha' + str(args.lr) + '_' \
    #         + str(args.n_games) + 'games'
    # figure_file = 'plots/' + fname + '.png'
    # scores_file = fname + '_scores.npy'

    main_net_update_feq = 1
    # main_net_update_feq = 1
    main_net_updates = 0
    n_eval_episodes = 20
    scores, eps_history = [], []
    steps = 0
    for i in range(args.n_games):
        year = 0
        done = False
        observation = env.reset()
        score = 0
        while not done:
            steps += 1
            slr_value = observation[0]
            surge_value = observation[1]

            state = env.get_state_vector(observation, time=year)
            action = agent.choose_action(env, state)
            print(f'year: {year}; action: {action}; combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} m')
            observation_, reward, done, info = env.step(action)
            score += reward

            if not args.load_checkpoint:
                # print(args.load_checkpoint)
                agent.store_transition(env.get_state_vector(observation, year), action,
                                       reward, env.get_state_vector(observation_, year), int(done))

                if steps % main_net_update_feq == 0 or done:
                    agent.learn(env)
                    main_net_updates += 1
            observation = observation_
            year += 1

        scores.append(score)
        avg_score = np.mean(scores[-10:])# 10 episodes average score
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon)

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        # evaluation iteration - evaluate 100 episodes and return average returns
        # run every 5 training episodes
        if i % 100:
            eval_scores = []
            for eval_epi in range(n_eval_episodes):
                observation = env.reset()
                done = False
                eval_rewards = 0
                steps = 0
                year = 0
                while not done:
                    steps += 1
                    year += 1
                    # [slr_state, surge_state] = components(observation)
                    slr_value = slr(observation[0])
                    surge_value = surge(observation[1])
                    print(
                        f'year: {year}, combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
                    action = agent.choose_action_eval(env, env.get_state_vector(observation, year))
                    print(f'action taken by agent: {action}')
                    # print(f'system: {env.system}')
                    # print(f'year: {env.year}')
                    new_observation, reward, done, info = env.step(action)
                    eval_rewards += reward
                    if i%1000==0:
                        if year <= 20:  # Log only up to year 10
                            log_file.write(
                                f'Episode: {i} year: {year}, action: {action}, Next state: {env.next_system}\n')
                            log_file.flush()
                    observation = new_observation
                    if done:
                        print('epi {} ends with total rewards {}'.format(eval_epi, eval_rewards))
                        break
                eval_scores.append(eval_rewards)
            mean_score_eval = np.mean(eval_scores)
            std_score_eval = np.std(eval_scores)

            wandb.log({"avg eval score": mean_score_eval})

        wandb.log({"avg score": avg_score, "best score": best_score})
        # if args.load_checkpoint and n_s
        #steps >= 18000:
        #     break


    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(x, scores, eps_history, figure_file)
    # np.save(scores_file, np.array(scores))

    wandb.finish()


    # testing on the environment with the best model seen:
    # load the best model saved
    agent.load_models()
    agent.epsilon = 0.0 #no random exploration
    # model = agent.q_eval

    test_episodes = 100
    scores = []
    for episode in range(test_episodes):
        observation = env.reset()
        done = False
        rewards = 0
        steps = 0
        year = 0
        while not done:
            steps += 1
            year += 1
            # [slr_state, surge_state] = components(observation)
            slr_value = slr(observation[0])
            surge_value = surge(observation[1])
            print(f'year: {year}, combined state: {observation}, slr: {slr_value} cm, surge: {surge_value} cm')
            action = agent.choose_action(env, env.get_state_vector(observation, year))
            # print(f'action taken by agent: {action}')
            # print(f'system: {env.system}')
            # print(f'year: {env.year}')
            new_observation, reward, done, info = env.step(action)
            rewards += reward
            observation = new_observation

            if done:
                print('epi {} ends with total rewards {}'.format(episode, rewards))
                break

        scores.append(rewards)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f'mean of returns (testing): {mean_score}')
    print(f'std of returns (testing): {std_score}')


