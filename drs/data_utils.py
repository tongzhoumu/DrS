import numpy as np
import pickle

def load_demo_dataset(path, keys=['observations', 'actions'], num_traj=None, success_only=False):
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    if success_only:
        trajectories = [t for t in trajectories if t['infos'][-1]['success']]
    if num_traj is not None:
        trajectories = trajectories[:num_traj]
    # trajectories is a list of trajectory
    # trajectories[0] has keys like: ['actions', 'dones', ...]
    dataset = {}
    for key in keys:
        if key in ['observations', 'states'] and \
                len(trajectories[0][key]) > len(trajectories[0]['actions']):
            dataset[key] = np.concatenate([
                t[key][:-1] for t in trajectories
            ], axis=0)
        elif key[:5] == 'next_' and key not in trajectories[0]:
            if isinstance(trajectories[0][key[5:]], dict):
                dataset[key] = {
                    k: np.concatenate([
                        t[key[5:]][k][1:] for t in trajectories
                    ], axis=0) for k in trajectories[0][key[5:]].keys()
                }
            else:
                dataset[key] = np.concatenate([
                    t[key[5:]][1:] for t in trajectories
                ], axis=0)
        else:
            dataset[key] = np.concatenate([
                t[key] for t in trajectories
            ], axis=0)
    return dataset

def load_raw_trajectories(path, num_traj=None, success_only=False):
    with open(path, 'rb') as f:
        trajectories = pickle.load(f)
    if success_only:
        trajectories = [t for t in trajectories if t['infos'][-1]['success']]
    if num_traj is not None:
        trajectories = trajectories[:num_traj]
    # trajectories is a list of trajectory
    # trajectories[0] has keys like: ['actions', 'dones', ...]
    return trajectories