import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from typing import List
from scipy.sparse import csr_matrix

class Map:
    def __init__(self):
        self.eps = None
        self.data = None
        self.eigvals = None
        self.eigvecs = None
        self.M = None
        self.N = None
        self.antisym_idx = None

    def diffmap_preprocess(self, data, epsilon=1.0, alpha=1.0, standardize=True, diagnose=False):
        if standardize:
            data = (data - data.mean(0)) / data.std(0)

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

        self.eps = epsilon
        self.data = data
        self.N = data.shape[0]

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

    def form_to_vec(self, frame_coeffs, H, d, mode=0):
        # select mode to convert
        ef_mode = frame_coeffs[:, mode]

        # unflatten to antisymmetric matrix
        idx_i, idx_j = self.antisym_idx
        ef_mat = self.unflatten_antisym_1tensor(ef_mode, self.M, idx_i, idx_j) # (M, M)

        # Sharp to vector field
        V = np.einsum('ijkl,ij->kl', H, ef_mat / 2.0) # (M, M)

        # Fourier transform coordinate functions to spectral space
        # (dim, N) @ (N, M) -> (dim, M)
        X_hat = (self.data * d[:, None]).T @ self.eigvecs

        # Apply vector field to coord functions - the pushforward
        # and inverse Fourier transform back to embedding space
        # (dim, M) @ (M, M) @ (M, N) -> (dim, N)
        V_tilde = X_hat @ V.T @ self.eigvecs.T
        
        print(f'Shape V_tilde: {V_tilde.shape}')
        return V_tilde

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
        
    def plot_eigen_forms(self, V_tilde, scale=1.0, x_scale=1.0):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        V_tilde_plot = self.normalize_by_percentile(V_tilde, percentile=75)
        X, Y, Z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        U, V, W = V_tilde_plot[0]/x_scale, V_tilde_plot[1], V_tilde_plot[2] # arrow components at each point
        
        # points, faint, for context
        ax.scatter(X, Y, Z, s=3, c='navy', alpha=0.8)

        # arrows
        ax.quiver(X, Y, Z, U, V, W, length=scale, normalize=False, color='coral', arrow_length_ratio=0.05)

        ax.set_xlabel('x'); ax.set_ylabel('u'); ax.set_zlabel('dxu')
        ax.set_box_aspect([np.ptp(X)*x_scale, np.ptp(Y), np.ptp(Z)])
        ax.set_title(f'Eigenform vector field')
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

def main():
    # load data
    data = np.load('data/population.npy')
    print(data[:3])

    # run algorithm
    map = Map()
    P_sym, d, K_hat = map.diffmap_preprocess(data, epsilon=0.01, alpha=1)
    JS = [36]
    fields = []
    for j in JS:
        map.compute_laplacian_spectrum(P_sym, d, K_hat, J=j,)
        # map.plot_laplacian_eigenfuncs()
        frame_coeffs, H = map.compute_frame(K_hat, d, n_keep=48)
        V_tilde = map.form_to_vec(frame_coeffs, H, d, mode=36)
        fields.append(V_tilde)
        map.plot_eigen_forms(V_tilde, scale=0.1,)

    if len(JS) > 1:
        overlap = abs(fields[0].ravel() @ fields[1].ravel()) / (np.linalg.norm(fields[0]) * np.linalg.norm(fields[1]))
        print(f'Overlap between truncations: {overlap:.4f}')

if __name__ == "__main__":
    main()