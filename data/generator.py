import numpy as np
import argparse
import yaml

class ODEGen:
    def __init__(
            self, n_trajectories=100, t_steps=100, 
            noise=0.01, seed=42,
            x_min=0.0, x_max=100.0,
    ):
        self.n_trajectories = n_trajectories
        self.t_steps = t_steps # points per trajectory
        self.noise = noise # relative additive gaussian noise
        self.x_min = x_min
        self.x_max = x_max

        self._points = None
        np.random.seed(seed)

    def _save(self, name):
        '''Save generated points to data/'''
        assert self._points is not None, "No points generated yet, run a generator function first"
        save_path = f'data/{name}.npy'
        np.save(save_path, self._points)
        print(f'Points saved to {save_path}')

    def _add_noise(self, arr):
        '''Relative additive Gaussian noise. Constant columns are left as is since scale=0'''
        scale = arr.std()
        return arr + np.random.normal(0.0, self.noise * scale, arr.shape)

    def _grid_sides(self):
        '''Split n_trajectories into a near-square 2-parameter grid'''
        n = int(round(np.sqrt(self.n_trajectories)))
        return max(n, 2)

    def constant_velocity_equation(self, name='constant_velocity', u0_min=-1.0, u0_max=2.0, v0_min=0.0, v0_max=3.0):
        '''
        Manifold F(x, u, u', u'') = u'' = 0
        Solutions u = u0 + v0*x, a 2-parameter family.
        Output (N, 4), columns [x, u, dxu, d2xu]
        '''
        n = self._grid_sides()
        x = np.linspace(self.x_min, self.x_max, self.t_steps)
        u0_grid = np.linspace(u0_min, u0_max, n)
        v0_grid = np.linspace(v0_min, v0_max, n)

        u0s, v0s = np.meshgrid(u0_grid, v0_grid, indexing='ij')
        u0s = u0s.ravel() + np.random.normal(0, 0.3*(u0_grid[1]-u0_grid[0]), n*n)
        v0s = v0s.ravel() + np.random.normal(0, 0.3*(v0_grid[1]-v0_grid[0]), n*n)

        trajectories = []
        for u0, v0 in zip(u0s, v0s):
            pts = np.zeros((self.t_steps, 4))
            pts[:, 0] = x
            pts[:, 1] = self._add_noise(u0 + v0 * x)
            pts[:, 2] = self._add_noise(np.full_like(x, v0))
            pts[:, 3] = self._add_noise(np.zeros_like(x))
            trajectories.append(pts)

        self._points = np.vstack(trajectories)
        print(f'Points generated, shape {self._points.shape}. First 2 points: {self._points[:2]}')
        self._save(name=name)

    def ermakov_pinney_equation(self, name='ermakov_pinney', A_min=0.5, A_max=2.0, B_min=-1.0, B_max=1.0):
        '''
        Manifold F(x, u, u', u'') = u'' - u^-3 = 0
        General solution u = sqrt(A x^2 + 2Bx + C) with AC - B^2 = 1,
        so C = (1 + B^2)/A. Free parameters (A, B), A > 0.
        Output (N, 4), columns [x, u, dxu, dxxu]
        '''
        n = self._grid_sides()
        x = np.linspace(self.x_min, self.x_max, self.t_steps)
        A_grid = np.linspace(A_min, A_max, n)
        B_grid = np.linspace(B_min, B_max, n)

        As, Bs = np.meshgrid(A_grid, B_grid, indexing='ij')
        As = As.ravel() + np.random.normal(0, 0.3*(A_grid[1]-A_grid[0]), n*n)
        Bs = Bs.ravel() + np.random.normal(0, 0.3*(B_grid[1]-B_grid[0]), n*n)
        As = np.clip(As, A_min * 0.5, None)

        trajectories = []
        for A, B in zip(As, Bs):
            C = (1.0 + B**2) / A
            w = A * x**2 + 2*B*x + C
            u = np.sqrt(w)
            dxu = (A * x + B) / u
            dxxu = u**-3
            pts = np.zeros((self.t_steps, 4))
            pts[:, 0] = x
            pts[:, 1] = self._add_noise(u)
            pts[:, 2] = self._add_noise(dxu)
            pts[:, 3] = self._add_noise(dxxu)
            trajectories.append(pts)

        self._points = np.vstack(trajectories)
        print(f'Points generated, shape {self._points.shape}. First 2 points: {self._points[:2]}')
        self._save(name=name)

    def population_equation(self, a=1, u0_min=1.0, u0_max=5.0, name='population'):
        '''
        Generate synthetic points on manifold F(x, u, u') = u' - au = 0
        Default 100 stacked trajectories of 100 t_steps each
        Each trajectory starts from u0
        Output shape (N, 3) where N = t_steps * n_trajectories
        Columns [x, u, dxu]
        '''
        n_trajectories = self.n_trajectories

        spacing = (u0_max - u0_min) / n_trajectories
        u0_nudge = np.random.normal(0, 0.3*spacing, n_trajectories)
        trajectories = []

        for u0 in np.linspace(u0_min, u0_max, n_trajectories) + u0_nudge:
            x = np.linspace(self.x_min, self.x_max, self.t_steps)
            points = np.zeros((self.t_steps, 3))
            points[:, 0] = x
            u = u0 * np.exp(a * x)
            dxu = a * u
            points[:, 1] = self._add_noise(u)
            points[:, 2] = self._add_noise(dxu)
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

    if eq == 'population':
        gen.population_equation(name=eq, **cfg['gen_params'])

    if eq == 'constant_velocity':
        gen.constant_velocity_equation(name=eq, **cfg['gen_params'])

    if eq == 'ermakov_pinney':
        gen.ermakov_pinney_equation(name=eq, **cfg['gen_params'])
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--eq', default='population')
    args = p.parse_args()
    main(**vars(args))