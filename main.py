from src.core import SymSolver
from data import analytic_sym
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
    
    n_trj = data.shape[0] // cfg['data']['t_steps']
    print(f"Expect from config {cfg['data']['n_trajectories']} x {cfg['data']['t_steps']}, ")
    print(f'got {n_trj} trajectories x {cfg["data"]["t_steps"]} steps.')
    print('First row: ', data[0])

    # run algorithm
    solver = SymSolver()
    run_cf = cfg['solver']
    
    P_sym, d, K_hat = solver.diffmap_preprocess(
        data, 
        t_steps=cfg['data']['t_steps'], 
        n_trajectories=n_trj,
        epsilon=run_cf['epsilon'], 
        alpha=run_cf['alpha'],
        diagnose=run_cf['eps_diagnose'],
        knn=run_cf['knn']        
    )
    solver.compute_laplacian_spectrum(
        P_sym, d, K_hat, 
        J=run_cf['M'], 
        plot_spectrum=plot_eigvals
    )
    solver.reconstruct_error(solver.data[:, 0], d, run_cf['M'], name='x')

    # all coordinate functions projected to spectral space
    X_hat = solver.fourier_tf(solver.data, d)

    if plot_eigbasis:
        solver.plot_laplacian_eigenfuncs()

    # SEC frame
    frame_coeffs, H = solver.compute_frame(K_hat, d, thres=run_cf['frame_retain_thres'])
    Vks = solver.frames_to_operators(frame_coeffs, H) # list of k (M, M) matrix operators

    # restricted cartan dist
    WX = solver.compute_WX()

    # validate on known symmetries:
    an_gens, an_coeffs = None, None
    if run_cf['analytic_validation']:
        if eq == 'ermakov_pinney':
            an_gens = analytic_sym.ermakov_pinney(solver.data)

            # how well can analytic syms be represented by frames
            an_coeffs, errs = solver.fit_analytic_syms(an_gens, Vks, X_hat, ridge=0.0)
            print('Xi fit errs:', np.round(errs, 4))
            print('coeff norms:', [np.linalg.norm(c) for c in an_coeffs])

        if an_coeffs is not None:
            # How well does each known syms close under contact form with W
            for i in range(an_gens.shape[0]):
                Xi = sum(ck*Vk for ck, Vk in zip(an_coeffs[i], Vks))
                res_abs, res_scale = solver.determining_residual(Xi, WX, X_hat, d)
                print(f'  Known sym X{i+1} determining residual rel:', res_abs/res_scale, 'abs, scale:', res_abs, res_scale)

    # plot options
    if plot_eigframe:
        k = run_cf['plot_frame_no']
        V_tilde = solver.framecoeff_to_vec(frame_coeffs, H, X_hat, mode=k)
        sub = run_cf['plot_subsample']*2 if run_cf['plot_subsample'] < data.shape[0]//2 else run_cf['plot_subsample']
        solver.plot_vector_field(
            V_tilde, scale=run_cf['plot_frame_vec_scale'], 
            title=f'{k}-th frame coefficients pushed forward',
            subsample=sub
        )

    if plot_w:
        solver.plot_vector_field(
            WX, scale=run_cf['plot_W_vec_scale'], title='Contact form kernel field',
            subsample=run_cf['plot_subsample']
        )

    if not skip_solve:

        # Intrinsic Lie bracket on manifold in coordinates, for each Vk together with WX pairing:
        Z_all = solver.jetlie_bracket_batch(
            Vks, WX, X_hat, d,
            sg_poly=3, sg_win=run_cf['sg_window'],
            chunk=run_cf['chunkz'], 
            verbose=run_cf['verbose']
        )
        # solve
        params_set = solver.solve(Z_all, max_kernel_dim=run_cf['max_kernel_dim'], plot_A_sv=True)
        del Z_all

        # operator rep of all symmetries
        sym_ops = []
        for i, params in enumerate(params_set):
            V_op = sum(ck * Vk for ck, Vk in zip(params, Vks)) 
            sym_ops.append(V_op)
            res_abs, res_scale = solver.determining_residual(V_op, WX, X_hat, d)
            print(f'  {i}-th AC eigval determining residual rel:', res_abs/res_scale, 'abs, scale:', res_abs, res_scale)

        print('Alignment with W: ')
        solver.angle_to_W(sym_ops, WX, X_hat)
        if an_gens is not None:
            # how well does the span line up between recovered and known, expect 0
            solver.principal_angles(sym_ops, X_hat, an_gens)
              
        # symmetries can be any linear combinations of sym_ops, plot
        for i, V_op in enumerate(sym_ops):
            V = solver.inv_fourier_pushfwd(X_hat, V_op)
            solver.plot_vector_field(
                V, scale=run_cf['plot_sym_vec_scale'], 
                subsample=run_cf['plot_subsample'],
                normalise='norm',
                title=f'Internal Symmetry {i}'
            )

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