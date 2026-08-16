from src.core import SymSolver
import numpy as np
import argparse
import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main(
        path_config: str, 
        plot_eigvals=False,
        plot_eigbasis=False, 
        plot_eigframe=False, 
        plot_w=False, 
        skip_solve=False,
):
    # load config
    cfg = load_config(path_config)

    # load data
    equation_name = cfg['equation']
    data = np.load(f'data/{equation_name}.npy')
    assert data.shape[0] == cfg['data']['n_trajectories'] * cfg['data']['t_steps'], 'Config expects different data shape'
    print('First row: ', data[0])

    # run algorithm
    solver = SymSolver()
    run_params = cfg['solver']
    
    P_sym, d, K_hat = solver.diffmap_preprocess(
        data, 
        t_steps=cfg['data']['t_steps'], 
        epsilon=run_params['epsilon'], 
        alpha=run_params['alpha']
    )
    solver.compute_laplacian_spectrum(
        P_sym, d, K_hat, 
        J=run_params['J'], 
        plot_spectrum=plot_eigvals
    )
    solver.reconstruct_error(solver.data[:, 0], d, run_params['J'], name='x')
    # coordinate functions projected to spectral space
    X_hat = solver.fourier_tf(solver.data, d)

    if plot_eigbasis:
        solver.plot_laplacian_eigenfuncs()

    frame_coeffs, H = solver.compute_frame(K_hat, d, n_frames=run_params['n_frames'])
    W_op = solver.compute_contact_form_kernel(d)

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
            title='Internal Symmetry'
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--path_config', default='configs/population.yaml')
    p.add_argument('--plot_eigvals', action='store_true')
    p.add_argument('--plot_eigbasis', action='store_true')
    p.add_argument('--plot_eigframe', action='store_true')
    p.add_argument('--plot_w', action='store_true')
    p.add_argument('--skip_solve', action='store_true')
    args = p.parse_args()

    main(**vars(args))