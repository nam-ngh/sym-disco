import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from typing import List
from scipy.sparse import csr_matrix

class Map:

    def process_coordinates(self, data, epsilon=1.0, alpha=1.0, standardize=True, diagnose=False):
        if standardize:
            data = (data - data.mean(0)) / data.std(0)

        # pairwise distances
        pair_dist = squareform(pdist(data))

        if diagnose:
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
        K = K_raw / (np.outer(q ** alpha, q ** alpha))
        del K_raw, q
        self.K_hat = K
        self.d = np.sum(K, axis=1)

        # sparsify? if yes also recompute d and Psym
        # K = np.where(K < K.max() * 1e-10, 0.0, K)
        # print('sparsity:', (K != 0).mean())
        # self.K_hat = csr_matrix(K)

        # transition matrix
        self.P_sym = K / (np.sqrt(self.d)[:, None]) / (np.sqrt(self.d)[None, :])

        # ensure symmetry
        self.P_sym = (self.P_sym + self.P_sym.T) / 2.0
        self.data = data
        self.eps = epsilon
        print(f'P_sym shape: {self.P_sym.shape}, d shape: {self.d.shape}')

    def compute_eigen_basis(self, J=30, trunc_idx: List[int] = None, plot_spectrum=False):
        eigvals, eigvecs_sym = eigsh(
            self.P_sym, k=J, which='LA',
            ncv=min(4*J, self.P_sym.shape[0])
        )

        # convert back to eigenvectors of P
        d_inv_sqrt = 1.0 / np.sqrt(self.d)
        eigvecs = eigvecs_sym * d_inv_sqrt[:, np.newaxis]

        if trunc_idx is not None:
            self.eigvals = eigvals[trunc_idx[0]:trunc_idx[1]][::-1]
            self.eigvecs = eigvecs[:, trunc_idx[0]:trunc_idx[1]][:, ::-1]
        else:
            self.eigvals = eigvals[-J:][::-1]
            self.eigvecs = eigvecs[:, -J:][:, ::-1]

        v, mu = eigvecs_sym[:, 0], eigvals[0]
        res = np.abs(self.P_sym @ v - mu * v).max()
        print(f'eigsh basis residual (worst mode): {res:.2e}')
        assert res < 1e-8, 'eigsh did not converge'

        # self.lam = -np.log(np.clip(self.eigvals, 1e-16, None)) / self.eps
        self.lam = (1 - self.eigvals)/self.eps

        if plot_spectrum:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(self.lam, 'o-')
            ax.set_xlabel('Index'); ax.set_ylabel('Eigenvalue')
            ax.set_title('Eigenvalue spectrum')
            plt.tight_layout()
            plt.show()

    def product_energy(self, c, p):
        # c shape (M, M, M)
        return np.einsum('s,ijs,skl->ijkl', self.lam**p, c, c)
    
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

    def check_laplacian_res(self,):
        j = min(5, len(self.lam)-1)
        v = self.eigvecs[:, j]
        Lgv = (self.d * v - self.K_hat @ v) / (self.eps * self.d)
        print('Laplacian residual:', np.abs(Lgv - self.lam[j] * v).max())

    def compute_eigen_forms(self, mode=0, n_keep=48, thres=None, spectral_energy=False):

        N = len(self.d)
        M = len(self.eigvals)

        if spectral_energy:
            c = np.einsum('mi,mj,m,mk->ijk', self.eigvecs, self.eigvecs, self.d, self.eigvecs)
            c0 = self.product_energy(c, 0)
            c1 = self.product_energy(c, 1)
            c2 = self.product_energy(c, 2)
        else:
            Pij = (self.eigvecs[:, :, None] * self.eigvecs[:, None, :]).reshape(N, -1)
            LP  = (self.d[:, None] * Pij - self.K_hat @ Pij) / self.eps # = Lsym @ Pij
            LgP = LP / self.d[:, None] # = Lg @ Pij

            c0 = (Pij.T @ (self.d[:, None] * Pij)).reshape(M,M,M,M)
            c1  = (Pij.T @ LP).reshape(M,M,M,M)
            c2  = (LgP.T @ (self.d[:, None] * LgP)).reshape(M,M,M,M)
            self.check_laplacian_res()

        self.check_c1c2(c1, c2)
        # (i,j,k,l) -> (i,l,j,k) - can transpose but this is cleaner
        c1_iljk = np.einsum('iljk->ijkl', c1)
        c1_ikjl = np.einsum('ikjl->ijkl', c1)
        # (i,j,k,l) -> (i,k,j,l)
        c2_iljk = np.einsum('iljk->ijkl', c2)
        c2_ikjl = np.einsum('ikjl->ijkl', c2)

        # We need to broadcast lambda_j (axis 1) and lambda_l (axis 3)
        lam_j = self.lam[np.newaxis, :, np.newaxis, np.newaxis]  # shape (1, M, 1, 1)
        lam_l = self.lam[np.newaxis, np.newaxis, np.newaxis, :]  # shape (1, 1, 1, M)

        # Hodge Grammian G (shape: M, M, M, M)
        G = ((lam_j + lam_l) * c0 - c1_ikjl) / 2.0
        G_jilk = np.einsum('jilk->ijkl', G)
        G_jikl = np.einsum('jikl->ijkl', G)
        G_ijlk = np.einsum('ijlk->ijkl', G)
        G_hat = G + G_jilk - G_jikl - G_ijlk
        self.check_G(G)

        lam = self.lam
        lam_i = lam[:, None, None, None]
        lam_j = lam[None, :, None, None]
        lam_k = lam[None, None, :, None]
        lam_l = lam[None, None, None, :]

        term1 = (lam_i + lam_j + lam_k + lam_l) * (c1_iljk - c1_ikjl)
        term2 = c2_ikjl - c2_iljk

        E_hat = term1 + term2
        self.check_E_hat(E_hat)
        H = G - np.einsum('jikl->ijkl', G)

        G_hat_flat, (idx_i, idx_j) = self.flatten_antisym_4tensor(G_hat, M)
        E_hat_flat, _ = self.flatten_antisym_4tensor(E_hat, M)
        print(f'G_hat_flat shape: {G_hat_flat.shape},')
        print(f'H shape: {H.shape}')
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
            U_tilde = U[:, :n_keep]
        print(f'Shape U_tilde: {U_tilde.shape}')

        L = U_tilde.T @ E_hat_flat @ U_tilde
        B = U_tilde.T @ G_hat_flat @ U_tilde
        L = (L + L.T) / 2.0
        B = (B + B.T) / 2.0
        print(f'Shape L: {L.shape}, Shape B: {B.shape}')

        nu, a = sp.linalg.eigh(L, B)
        print(f'Shape nu: {nu.shape}, Shape a: {a.shape}')
        print(f'nu spectrum: {nu[:8]}, {nu[-2:]}')
        print('nu nullspace dim:', np.sum(np.abs(nu) < 1e-6))

        # operator rep of v
        phi_coeff_all = U_tilde @ a # (keep.sum, P)
        # select mode to visualize
        phi_mode = phi_coeff_all[:, mode]
        phi_mat = self.unflatten_antisym_1tensor(phi_mode, M, idx_i, idx_j) # (M, M)
        V = np.einsum('ijkl,ij->kl', H, phi_mat / 2.0) # (M, M)

        D = diags(self.d)
        X_hat = self.data.T @ D @ self.eigvecs # (dim, N) @ (N, N) @ (N, M) -> (dim, M)

        V_tilde = X_hat @ V.T @ self.eigvecs.T # (dim, M) @ (M, M) @ (M, N) -> (dim, N)
        
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

    def plot_raw_eigbasis(self,):
        fig = plt.figure(figsize=(15,4))
        for n, j in enumerate([1, 5, 10, 19]):
            ax = fig.add_subplot(1, 4, n+1, projection='3d')
            ax.scatter(self.data[:,0], self.data[:,1], self.data[:,2],
                    c=self.eigvecs[:, j], cmap='RdBu', s=2)
            ax.set_title(f'phi_{j}')
        plt.show()

def main():
    data = np.load('data/population.npy')
    print(data[:3])
    map = Map()
    map.process_coordinates(data, epsilon=0.01, alpha=1)
    JS = [36]
    fields = []
    for j in JS:
        map.compute_eigen_basis(J=j, plot_spectrum=False)
        # map.plot_raw_eigbasis()
        V_tilde = map.compute_eigen_forms(mode=3, n_keep=48, spectral_energy=False)
        fields.append(V_tilde)
        map.plot_eigen_forms(V_tilde, scale=0.1,)

    if len(JS) > 1:
        overlap = abs(fields[0].ravel() @ fields[1].ravel()) / (np.linalg.norm(fields[0]) * np.linalg.norm(fields[1]))
        print(f'Overlap between truncations: {overlap:.4f}')

if __name__ == "__main__":
    main()