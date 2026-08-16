import numpy as np
import argparse
import yaml

class ODEGen:
    def __init__(
            self, n_trajectories=100, t_steps=100, 
            noise=0.01, seed=42,
            x_min=0.0, x_max=100.0, 
            u0_min=1.0, u0_max=5.0
    ):
        self.n_trajectories = n_trajectories
        self.t_steps = t_steps # points per trajectory
        self.noise = noise # relative additive gaussian noise
        self.x_min = x_min
        self.x_max = x_max
        self.u0_min = u0_min
        self.u0_max = u0_max

        self._points = None
        np.random.seed(seed)

    def _save(self, name):
        '''Save generated points to data/'''
        assert self._points is not None, "No points generated yet, run a generator function first"
        save_path = f'data/{name}.npy'
        np.save(save_path, self._points)
        print(f'Points saved to {save_path}')

    def population_equation(self, a=1, name='population'):
        '''
        Generate synthetic points on manifold F(x, u, u') = u' - au = 0
        Default 100 stacked trajectories of 100 t_steps each
        Each trajectory starts from u0
        Output shape (N, 3) where N = t_steps * n_trajectories
        Columns [x, u, dxu]
        '''
        n_trajectories = self.n_trajectories

        spacing = (self.u0_max - self.u0_min) / n_trajectories
        u0_nudge = np.random.normal(0, 0.3*spacing, n_trajectories)
        trajectories = []

        for u0 in np.linspace(self.u0_min, self.u0_max, n_trajectories) + u0_nudge:
            x = np.linspace(self.x_min, self.x_max, self.t_steps)
            points = np.zeros((self.t_steps, 3))
            points[:, 0] = x
            u = u0 * np.exp(a * x)
            dxu = a * u
            noise_u   = np.random.normal(0.0, self.noise * u.std(), self.t_steps)
            noise_dxu = np.random.normal(0.0, self.noise * dxu.std(), self.t_steps)
            points[:, 1] = u + noise_u
            points[:, 2] = dxu + noise_dxu
            trajectories.append(points)
        
        self._points = np.vstack(trajectories)  # (t_steps * n_trajectories, 3)
        print(f'Points generated, shape {self._points.shape}. First 2 points: {self._points[: 2]}')
        self._save(name=name)

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(eq: str):
    cfg = load_config(f'configs/{eq}.yaml')
    gen = ODEGen(**cfg['data'])
    gen.population_equation(name=cfg['equation'])
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--eq', default='population')
    args = p.parse_args()
    main(**vars(args))