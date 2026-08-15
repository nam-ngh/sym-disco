import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from typing import List

class Map:
    def __init__(self):
        self.eps = None
        self.data = None
        self.eigvals = None
        self.eigvecs = None
        self.M = None
        self.N = None
        self.n_frames = None
        self.antisym_idx = None
        self.t_size = None
        self.n_trajectory = None

    def diffmap_preprocess(self, data, trajectory_size, epsilon=1.0, alpha=1.0, standardize=True, diagnose=False):
        '''
        Preprocess data by diffusion maps, returning objects no longer in coordinate space.
        '''

        if standardize:
            mu, sd = data.mean(0), data.std(0)
            data = data.copy()
            data[:, 0] = (data[:, 0] - mu[0]) / sd[0] # x
            data[:, 1] = (data[:, 1] - mu[1]) / sd[1] # u
            data[:, 2] = data[:, 2] * sd[0] / sd[1]

        # pairwise distances
        pair_dist = squareform(pdist(data))

        if diagnose:
            # plot to determine optimal epsilon
            D2 = pair_dist**2
            print(np.median(D2))
            eps_grid = np.logspace(-5, 2, 40)
            S = [np.exp(-D2/e).sum() for e in eps_grid]
            plt.loglog(eps_grid, S, 'o-')
        
        # kernel matrix K 
        K_raw = np.exp(-(pair_dist ** 2)/(4*epsilon))

        # degree vec
        q = np.sum(K_raw, axis=1)

        # normalised kernel & degree vec
        K_hat = K_raw / (np.outer(q ** alpha, q ** alpha))
        del K_raw, q

        d = np.sum(K_hat, axis=1)

        # sparsify? if yes also recompute d and Psym
        # K = np.where(K < K.max() * 1e-10, 0.0, K)
        # print('sparsity:', (K != 0).mean())
        # self.K_hat = csr_matrix(K)

        # transition matrix
        P_sym = K_hat / (np.sqrt(d)[:, None]) / (np.sqrt(d)[None, :])

        # ensure symmetry
        P_sym = (P_sym + P_sym.T) / 2.0
        print(f'P_sym shape: {P_sym.shape}, d shape: {d.shape}')

        # store
        self.eps = epsilon
        self.data = data
        self.N = data.shape[0]
        self.t_size = trajectory_size

        return P_sym, d, K_hat

    def compute_laplacian_spectrum(
            self, P_sym, d, K_hat,
            J=30, 
            trunc_idx: List[int] = None, 
            plot_spectrum=False
    ):
        '''
        Computes the J eigenbasis of smooth functions on the manifold.
        Sets self.eigvals, self.eigvecs, self.M.
        '''
        eigvals, eigvecs_sym = eigsh(
            P_sym, k=J, which='LA',
            ncv=min(6*J, P_sym.shape[0]),
        )

        # convert back to eigenvectors of P
        d_inv_sqrt = 1.0 / np.sqrt(d)
        eigvecs = eigvecs_sym * d_inv_sqrt[:, np.newaxis]

        if trunc_idx is not None:
            self.eigvals = eigvals[trunc_idx[0]:trunc_idx[1]][::-1]
            self.eigvecs = eigvecs[:, trunc_idx[0]:trunc_idx[1]][:, ::-1]
        else:
            self.eigvals = eigvals[-J:][::-1]
            self.eigvecs = eigvecs[:, -J:][:, ::-1]

        # convert to Laplacian eigenvalues
        # self.eigvals = -np.log(np.clip(self.eigvals, 1e-16, None)) / self.eps
        self.eigvals = (1 - self.eigvals)/self.eps
        self.M = self.eigvals.shape[0]

        # numerical checks
        self.check_laplacian_res(K_hat, d)
        eig_res = self.check_eigsh_res(eigvals, eigvecs_sym, P_sym)
        assert eig_res.max() < 1e-8, 'eigsh did not converge'
        assert np.allclose(self.eigvecs.T @ (d[:, None] * self.eigvecs), np.eye(self.M), atol=1e-8), \
        'basis eigenfunctions are not orthogonal under the weighted inner product'

        if plot_spectrum:
            _, ax = plt.subplots(figsize=(6, 3))
            ax.plot(self.eigvals, 'o-')
            ax.set_xlabel('Index'); ax.set_ylabel('Eigenvalue')
            ax.set_title('Eigenvalue spectrum')
            plt.tight_layout()
            plt.show()

    def check_eigsh_res(self, eigvals, eigvecs_sym, P_sym):
        res = np.abs(P_sym @ eigvecs_sym - eigvals[None, :] * eigvecs_sym).max(axis=0)
        print(f'eigsh residual: max {res.max():.2e} at mode {res.argmax()}')
        return res

    def check_laplacian_res(self, K_hat, d):
        E = self.eigvecs
        LgE = (d[:, None] * E - K_hat @ E) / (self.eps * d[:, None])
        res = np.abs(LgE - self.eigvals[None, :] * E).max(axis=0)
        print(f'Laplacian residual: max {res.max():.2e} at mode {res.argmax()}')

    def reconstruction_error(self, f, d, J):
        '''
        Relative error in reconstructing a coordinate function f from the
        leading M eigenfunctions. Diagnostic for basis adequacy.
        '''
        phi = self.eigvecs[:, :J]
        f_hat = phi.T @ (d * f)
        print(f'Reconstruction Err J = {J}', np.abs(phi @ f_hat - f).max() / np.linalg.norm(f - f.mean()))

    def product_energy(self, c, p):
        # c shape (M, M, M)
        return np.einsum('s,ijs,skl->ijkl', self.eigvals**p, c, c)
    
    @staticmethod
    def flatten_antisym_4tensor(T, M):
        '''
        T shape (M,M,M,M) flattened to (P,P)
        since b_hat is antisymmetric,
        P = M(M-1)/2 = no of unique pairing by i & j, with i<j.
        '''
        idx_i, idx_j = np.triu_indices(M, k=1) # k=1 excludes the diagonal i=j
        T_flat = T[idx_i[:, None], idx_j[:, None], idx_i[None, :], idx_j[None, :]]
        return T_flat, (idx_i, idx_j)
    
    @staticmethod
    def unflatten_antisym_1tensor(T, M, idx_i, idx_j):
        '''
        Converts a flattened antisymmetric tensor: (P,) -> (M, M).
        '''
        T_unflat = np.zeros((M, M))
        T_unflat[idx_i, idx_j] = T
        T_unflat[idx_j, idx_i] = -T
        return T_unflat

    @staticmethod
    def check_c1c2(c1, c2, tol=1e-10):
        test = np.abs(c1 - np.einsum('jikl->ijkl', c1)).max()
        print('c1 (1, 2) symm? ', test, test < tol)

        test = np.abs(c1 - np.einsum('klij->ijkl', c1)).max()
        print('c1 symm by pairswap? ', test, test < tol)

        test = np.abs(c2 - np.einsum('jikl->ijkl', c2)).max()
        print('c2 (1, 2) symm? ', test, test < tol)

        test = np.abs(c2 - np.einsum('klij->ijkl', c2)).max()
        print('c2 symm by pairswap? ', test, test < tol)

    @staticmethod
    def check_G(G, tol=1e-10):
        test = np.abs(G - np.einsum('kjil->ijkl', G)).max()
        print('G (1,3) symm? ', test, test < tol)
        test = np.abs(G - np.einsum('ilkj->ijkl', G)).max()
        print('G (2,4) symm? ', test, test < tol)

    @staticmethod
    def check_G_hat_flat(G_hat_flat, tol=1e-10, psd_tol=1e-6):
        test = np.abs(G_hat_flat).max()
        print('G_hat_flat nonzero? ', test, test > tol)

        test = np.abs(G_hat_flat - G_hat_flat.T).max()
        print('G_hat_flat symm? ', test, test < tol)
        
        w = np.linalg.eigvalsh((G_hat_flat + G_hat_flat.T)/2)
        print('G hat flat PSD? ', np.min(w) >= -psd_tol * np.abs(w).max())
        print('G hat flat eigvals spectrum: ', w[:3], w[-3:])

    @staticmethod
    def check_E_hat(E_hat, tol=1e-10):
        test = np.abs(E_hat).max()
        print('E_hat nonzero? ', test, test > tol)

        test = np.abs(E_hat + np.einsum('jikl->ijkl', E_hat)).max()
        print('E_hat antisym (1,2)? ', test, test < tol)

        test = np.abs(E_hat + np.einsum('ijlk->ijkl', E_hat)).max()
        print('E_hat antisym (3,4)? ', test, test < tol)

    def compute_frame(
            self, K_hat, d, 
            n_keep=48, thres=None, 
            spectral_energy=False
    ):
        '''
        Computes the frame rep coefficients of eigen 1-forms of the Hodge Laplacian
        Returns:
            frame_coeffs: (P, n_keep)
        '''

        M = self.M

        if spectral_energy:
            c = np.einsum('mi,mj,m,mk->ijk', self.eigvecs, self.eigvecs, d, self.eigvecs)
            c0 = self.product_energy(c, 0)
            c1 = self.product_energy(c, 1)
            c2 = self.product_energy(c, 2)
        else:
            Pij = (self.eigvecs[:, :, None] * self.eigvecs[:, None, :]).reshape(self.N, -1)
            LP  = (d[:, None] * Pij - K_hat @ Pij) / self.eps # = Lsym @ Pij
            LgP = LP / d[:, None] # = Lg @ Pij

            c0 = (Pij.T @ (d[:, None] * Pij)).reshape(M,M,M,M)
            c1  = (Pij.T @ LP).reshape(M,M,M,M)
            c2  = (LgP.T @ (d[:, None] * LgP)).reshape(M,M,M,M)

        self.check_c1c2(c1, c2)
        # (i,j,k,l) -> (i,l,j,k) - can transpose but this is cleaner
        c1_iljk = np.einsum('iljk->ijkl', c1)
        c1_ikjl = np.einsum('ikjl->ijkl', c1)
        # (i,j,k,l) -> (i,k,j,l)
        c2_iljk = np.einsum('iljk->ijkl', c2)
        c2_ikjl = np.einsum('ikjl->ijkl', c2)

        # We need to broadcast lambda_j (axis 1) and lambda_l (axis 3)
        lam_j = self.eigvals[np.newaxis, :, np.newaxis, np.newaxis]  # shape (1, M, 1, 1)
        lam_l = self.eigvals[np.newaxis, np.newaxis, np.newaxis, :]  # shape (1, 1, 1, M)

        # Hodge Grammian G (shape: M, M, M, M)
        G = ((lam_j + lam_l) * c0 - c1_ikjl) / 2.0
        G_jilk = np.einsum('jilk->ijkl', G)
        G_jikl = np.einsum('jikl->ijkl', G)
        G_ijlk = np.einsum('ijlk->ijkl', G)
        G_hat = G + G_jilk - G_jikl - G_ijlk
        self.check_G(G)

        lam = self.eigvals
        lam_i = lam[:, None, None, None]
        lam_j = lam[None, :, None, None]
        lam_k = lam[None, None, :, None]
        lam_l = lam[None, None, None, :]

        term1 = (lam_i + lam_j + lam_k + lam_l) * (c1_iljk - c1_ikjl)
        term2 = c2_ikjl - c2_iljk

        E_hat = term1 + term2
        self.check_E_hat(E_hat)
        H = G - np.einsum('jikl->ijkl', G)

        G_hat_flat, idxs = self.flatten_antisym_4tensor(G_hat, M)
        E_hat_flat, _ = self.flatten_antisym_4tensor(E_hat, M)
        print(f'G_hat_flat shape: {G_hat_flat.shape},')
        print(f'H shape: {H.shape}')
        self.antisym_idx = idxs
        self.check_G_hat_flat(G_hat_flat)

        G1_flat = G_hat_flat + E_hat_flat
        G1_flat = (G1_flat + G1_flat.T) / 2.0
        print('G1_flat shape: ', G1_flat.shape)

        h, U = np.linalg.eigh(G1_flat)
        idx = np.argsort(h)[::-1]
        h, U = h[idx], U[:, idx]
        if thres is not None:
            h11 = h[0]
            keep = h > (h11 * thres)
            print(f'Keeping {np.sum(keep)} out of {len(h)} eigenvectors with threshold {thres}')
            U_tilde = U[:, keep]
        else:
            U_tilde = U[:, :n_keep] # (P, n_keep)
            self.n_frames = n_keep
            
        print(f'Shape U_tilde: {U_tilde.shape}')

        L = U_tilde.T @ E_hat_flat @ U_tilde
        B = U_tilde.T @ G_hat_flat @ U_tilde
        L = (L + L.T) / 2.0
        B = (B + B.T) / 2.0
        print(f'Shape L: {L.shape}, Shape B: {B.shape}')

        nu, a = sp.linalg.eigh(L, B) # (n_keep,), (n_keep, n_keep)
        print(f'Shape nu: {nu.shape}, Shape a: {a.shape}')
        print(f'nu spectrum: {nu[:8]}, {nu[-2:]}')
        print('nu nullspace dim:', np.sum(np.abs(nu) < 1e-6))

        # coefficients for all basis eigen 1-forms
        frame_coeffs = U_tilde @ a # (P, n_keep)
        print(f'Eigen frame coefficients shape: {frame_coeffs.shape}')

        return frame_coeffs, H

    def compute_contact_form_kernel(self, d):
        '''
        Computes operator rep of vector field W, at each point spanning ker(gamma) up to a scale factor.
        Returns:
            W_hat: (M, M)
        '''
        x = self.data[:, 0].reshape(-1, self.t_size) # (n_t, t_size)
        self.n_trajectory = x.shape[0]
        phi = self.eigvecs.reshape(-1, self.t_size, self.M) # (n_t, t_size, M)
        dphi = np.stack(
            [np.gradient(phi[t], x[t], axis=0) for t in range(phi.shape[0])]
        )
        dphi = dphi.reshape(-1, self.eigvecs.shape[1]) # (N, M)

        return self.eigvecs.T @ (d[:, None] * dphi)

    def lie_bracket(self, V, W):
        '''
        Computes the Lie bracket of vector fields V and W
        as matrix commutator via their op rep
        Returns:
            Z: (M, M) operator
        '''
        return V @ W - W @ V

    def fourier_tf(self, f, d):
        '''
        Fourier transform function(s) to spectral space: (?, N) @ (N, M) -> (?, M)
        f: may be (N, 1) ... (N, dim) hence ?
        '''
        f = f.reshape(len(f), -1) # ensure 2D array
        return (f * d[:, None]).T @ self.eigvecs
        
    def inv_fourier_pushfwd(self, f_hat, V):
        '''
        Apply V (pushforward) then inverse fourier transform back to jet space: 
        (?, M) @ (M, M) @ (M, N) -> (?, N)
        ? may be 1...dim
        '''
        return f_hat @ V.T @ self.eigvecs.T

    @staticmethod
    def sharp(H, form_coeff):
        '''
        Convert 1-form to vector: (M, M) -> (M, M)
        '''
        return np.einsum('ijkl,ij->kl', H, form_coeff / 2.0)

    def solve(self, frame_coeffs, H, d, W):
        idx_i, idx_j = self.antisym_idx
        x = self.data[:, 0]
        u = self.data[:, 1]
        ux = self.data[:, 2]
        u_hat = self.fourier_tf(u, d)
        x_hat = self.fourier_tf(x, d)

        # mask to discard start and ends of trajectories, where derivative is unstable
        mask = np.ones((self.n_trajectory, self.t_size), bool)
        mask[:, :3] = mask[:, -3:] = False
        mask = mask.ravel()

        # numerical checks
        xr = self.data[:, 0].reshape(-1, self.t_size)
        ur = self.data[:, 1].reshape(-1, self.t_size)
        du = np.stack([np.gradient(ur[t], xr[t]) for t in range(ur.shape[0])]).ravel()
        print('raw du/dx vs ux:', np.linalg.norm(du - ux) / np.linalg.norm(ux))
        print('masked du/dx vs ux', np.linalg.norm((du-ux)[mask]) / np.linalg.norm(ux[mask]))
        print('u recon res:', np.linalg.norm(self.eigvecs @ u_hat.ravel() - u) / np.linalg.norm(u - u.mean()))
        print('x recon res:', np.linalg.norm(self.eigvecs @ x_hat.ravel() - x) / np.linalg.norm(x - x.mean()))
        Wx = self.inv_fourier_pushfwd(x_hat, W).ravel()
        print('Wx masked', Wx[mask].mean(), Wx[mask].std())
        print('Wx all', Wx.mean(), Wx.std())
        res = self.inv_fourier_pushfwd(u_hat, W).ravel() - ux
        print('Wu masked:', np.linalg.norm(res[mask]) / np.linalg.norm(ux[mask]))
        print('Wu all:   ', np.linalg.norm(res) / np.linalg.norm(ux))

        cols = []
        Vks = []
        for k in range(self.n_frames):
            rho_k_flat = frame_coeffs[:, k]
            rho_k = self.unflatten_antisym_1tensor(rho_k_flat, self.M, idx_i, idx_j) # (M, M)
            Vk = self.sharp(H, rho_k) # (M, M)
            Zk = self.lie_bracket(Vk, W)
            Zku = self.inv_fourier_pushfwd(u_hat, Zk)
            Zkx = self.inv_fourier_pushfwd(x_hat, Zk)
            col = (Zku - ux * Zkx).ravel()
            cols.append(col)
            Vks.append(Vk)

        A = np.column_stack(cols)[mask]

        def is_trivial(c, tol=1e-2):
            V = sum(ck * Vk for ck, Vk in zip(c, Vks))
            Vu = self.inv_fourier_pushfwd(u_hat, V).ravel()
            Vx = self.inv_fourier_pushfwd(x_hat, V).ravel()
            gV = Vu - ux * Vx # gamma(V), pointwise
            Vnorm = np.sqrt(Vu**2 + Vx**2)
            return np.abs(gV[mask]).mean() / Vnorm[mask].mean() < tol

        ### SOLVE ###
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
        singular_values_scaled = S/S[0]
        print(singular_values_scaled[-10:])
        null_space_idx = np.where(singular_values_scaled < 1e-2)[0]
        if len(null_space_idx) == 0:
            raise ValueError('No non-trivial null space found.')
        nontrivial = [Vt[i] for i in null_space_idx if not is_trivial(Vt[i])]
        trivial_idx = [i for i in null_space_idx if is_trivial(Vt[i])]
        print(f'null dim {len(null_space_idx)}, trivial {len(trivial_idx)}')
        return nontrivial, Vks
            
    def form_to_vec(self, frame_coeffs, H, d, mode=0):
        # select mode to convert
        ef_mode = frame_coeffs[:, mode]
        idx_i, idx_j = self.antisym_idx
        ef_mat = self.unflatten_antisym_1tensor(ef_mode, self.M, idx_i, idx_j)
        V = self.sharp(H, ef_mat)
        X_hat = self.fourier_tf(self.data, d)
        return self.inv_fourier_pushfwd(X_hat, V)

    @staticmethod
    def normalize_by_norm(field):
        norm = np.linalg.norm(field, axis=0)
        norm[norm == 0] = 1.0
        return field / norm

    @staticmethod
    def normalize_by_percentile(field, percentile=75):
        mags = np.linalg.norm(field, axis=0)
        mag_scale = np.percentile(mags, percentile)
        return field / mag_scale

    @staticmethod
    def normalize_by_mean(field):
        mags = np.linalg.norm(field, axis=0)
        mag_scale = np.mean(mags)
        return field / mag_scale
        
    def plot_vector_field(self, V, normalise=75, scale=1.0, x_scale=1.0):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        if type(normalise) == int:
            V_plot = self.normalize_by_percentile(V, percentile=75)
        elif normalise == 'mean':
            V_plot = self.normalize_by_mean(V)
        elif normalise == 'norm':
            V_plot = self.normalize_by_norm(V)

        X, Y, Z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        U, V, W = V_plot[0]/x_scale, V_plot[1], V_plot[2] # arrow components at each point
        
        # points, faint, for context
        ax.scatter(X, Y, Z, s=3, c='navy', alpha=0.8)

        # arrows
        ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=False, color='coral', arrow_length_ratio=0.05)

        ax.set_xlabel('x'); ax.set_ylabel('u'); ax.set_zlabel('dxu')
        ax.set_box_aspect([np.ptp(X)*x_scale, np.ptp(Y), np.ptp(Z)])
        ax.set_title(f'Tangent Vector Field')
        plt.tight_layout()
        plt.show()

    def plot_laplacian_eigenfuncs(self,):
        fig = plt.figure(figsize=(15,4))
        for n, j in enumerate([1, 5, 10, 19]):
            ax = fig.add_subplot(1, 4, n+1, projection='3d')
            ax.scatter(
                self.data[:,0], self.data[:,1], self.data[:,2],
                c=self.eigvecs[:, j], cmap='RdBu', s=2
            )
            ax.set_title(f'phi_{j}')
        plt.show()

def main(plot_laplacian_spectrum=False, plot_eigenform=False, plot_W=False, solve=False):
    # load data
    data = np.load('data/population.npy')
    print(data[:3])

    # run algorithm
    map = Map()
    P_sym, d, K_hat = map.diffmap_preprocess(data, trajectory_size=60, epsilon=0.01, alpha=1)
    map.compute_laplacian_spectrum(P_sym, d, K_hat, J=36,)
    map.reconstruction_error(map.data[:, 0], d, 36)

    if plot_laplacian_spectrum:
        map.plot_laplacian_eigenfuncs()

    frame_coeffs, H = map.compute_frame(K_hat, d, n_keep=48)
    W = map.compute_contact_form_kernel(d)

    if plot_eigenform:
        V_tilde = map.form_to_vec(frame_coeffs, H, d, mode=36)
        map.plot_vector_field(V_tilde, scale=0.1,)

    if plot_W:
        X_hat = map.fourier_tf(map.data, d)
        W_arrows = map.inv_fourier_pushfwd(X_hat, W)
        map.plot_vector_field(W_arrows, scale=0.1)

    if solve:
        kernel, Vks = map.solve(frame_coeffs, H, d, W)
        params = kernel[0] # can be linear combo
        V_op = sum(ck * Vk for ck, Vk in zip(params, Vks))
        X_hat = map.fourier_tf(map.data, d)
        V = map.inv_fourier_pushfwd(X_hat, V_op)
        map.plot_vector_field(V, scale=0.1)

if __name__ == "__main__":
    main(plot_eigenform=True, solve=True)