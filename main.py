from src.core import SymSolver
import numpy as np
import argparse
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def config_path(eq: str) -> str:
    return f'configs/{eq}.yaml'

def main(
        eq: str, 
        plot_eigvals=False,
        plot_eigbasis=False, 
        plot_eigframe=False, 
        plot_w=False, 
        skip_solve=False,
):
    # load config
    cfg = load_config(config_path(eq))

    # load data
    data = np.load(f'data/{eq}.npy')
    assert data.shape[0] % cfg['data']['t_steps'] == 0, \
        f'{data.shape[0]} points not divisible by t_steps={cfg["data"]["t_steps"]}'
    n_traj = data.shape[0] // cfg['data']['t_steps']
    print(f'{n_traj} trajectories x {cfg["data"]["t_steps"]} steps')
    print('First row: ', data[0])

    # run algorithm
    solver = SymSolver()
    run_params = cfg['solver']
    
    P_sym, d, K_hat = solver.diffmap_preprocess(
        data, 
        t_steps=cfg['data']['t_steps'], 
        epsilon=run_params['epsilon'], 
        alpha=run_params['alpha'],
        diagnose=run_params['eps_diagnose'],
        knn=run_params['knn']        
    )
    solver.compute_laplacian_spectrum(
        P_sym, d, K_hat, 
        J=run_params['M'], 
        plot_spectrum=plot_eigvals
    )
    solver.reconstruct_error(solver.data[:, 0], d, run_params['M'], name='x')
    # coordinate functions projected to spectral space
    X_hat = solver.fourier_tf(solver.data, d)

    if plot_eigbasis:
        solver.plot_laplacian_eigenfuncs()

    frame_coeffs, H = solver.compute_frame(K_hat, d, thres=run_params['frame_retain_thres'])
    W_op = solver.compute_contact_form_kernel(d, run_checks=True,)

    if plot_eigframe:
        k = run_params['plot_frame_no']
        V_tilde = solver.framecoeff_to_vec(frame_coeffs, H, d, mode=k)
        solver.plot_vector_field(
            V_tilde, scale=run_params['plot_frame_vec_scale'], 
            title=f'{k}-th frame coefficients pushed forward'
        )

    if plot_w:
        W_arrows = solver.inv_fourier_pushfwd(X_hat, W_op)
        solver.plot_vector_field(W_arrows, scale=run_params['plot_W_vec_scale'], title='Contact form kernel field')

    if not skip_solve:
        kernel, Vks = solver.solve(
            frame_coeffs, H, d, W_op, 
            max_kernel_dim=run_params['max_kernel_dim']
        )
        # params can be linear combo of different kernel dims
        plot_dim = run_params['plot_kernel_dim']
        params = kernel[plot_dim]

        V_op = sum(ck * Vk for ck, Vk in zip(params, Vks))
        V = solver.inv_fourier_pushfwd(X_hat, V_op)
        solver.plot_vector_field(
            V, scale=run_params['plot_sym_vec_scale'], 
            subsample=run_params['plot_sym_subsample'],
            normalise='norm',
            title='Internal Symmetry'
        )
        print('V, W similarity', np.abs(np.sum(V_op * W_op)) / (np.linalg.norm(V_op) * np.linalg.norm(W_op)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--eq', default='population')
    p.add_argument('--plot_eigvals', action='store_true')
    p.add_argument('--plot_eigbasis', action='store_true')
    p.add_argument('--plot_eigframe', action='store_true')
    p.add_argument('--plot_w', action='store_true')
    p.add_argument('--skip_solve', action='store_true')
    args = p.parse_args()

    main(**vars(args))