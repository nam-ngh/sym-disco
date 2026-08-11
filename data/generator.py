import numpy as np

class DataGenerator:
    def __init__(self, npts=10000, noise=0.01, x_min=0.0, x_max=100.0, seed=42):
        self.npts = npts
        self.noise = noise
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

    def population_equation(self, n_trajectories=5, a=1):
        '''
        Generate synthetic points on manifold F(x, u, u') = u' - au = 0
        Default 5 stacked trajectories of npts each, with different intial conditions u0
        '''
        trajectories = []

        for u0 in range(1, n_trajectories + 1):
            x = np.linspace(self.x_min, self.x_max, self.npts)
            points = np.zeros((self.npts, 3))
            points[:, 0] = x
            u = u0 * np.exp(a * x)
            dxu = a * u
            noise_u   = np.random.uniform(-self.noise, self.noise, self.npts) * np.abs(u)
            noise_dxu = np.random.uniform(-self.noise, self.noise, self.npts) * np.abs(dxu)
            points[:, 1] = u + noise_u
            points[:, 2] = dxu + noise_dxu
            trajectories.append(points)
        
        self._points = np.vstack(trajectories)  # (npts * n_trajectories, 3)
        print(f'Points generated, shape {self._points.shape}. First 2 points: {self._points[: 2]}')
        self._save(name='population')

if __name__ == "__main__":
    gen = DataGenerator(npts=72, x_max=3.0, noise=0.02)
    gen.population_equation(n_trajectories=72)