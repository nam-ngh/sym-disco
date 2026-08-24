import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, diags
from scipy.signal import savgol_filter
from typing import List

class SymSolver:
    def __init__(self):
        self.eps = None
        self.data = None
        self.max_order = None
        self.data_mu = None
        self.data_sd = None
        self.eigvals = None
        self.eigvecs = None
        self.M = None
        self.N = None
        self.n_frames = None
        self.antisym_idx = None
        self.t_size = None
        self.n_trajectory = None
        self.linsys_keep_mask = None


    ########################### Diffusion maps algorithm ###########################


    def diffmap_preprocess(self, data, t_steps, n_trajectories, epsilon=1.0, alpha=1.0, standardize=True, diagnose=False, knn=None):
        '''
        Preprocess data by diffusion maps, returning objects no longer in coordinate space.
        '''
        self.max_order = int(data.shape[1] - 2)
        print(f'Data shape: {data.shape}. Max derivative order: {self.max_order}')

        if standardize:
            mu, sd = data.mean(0), data.std(0)
            self.data_mu = mu
            self.data_std = sd
            data = data.copy()
            data[:, 0] = (data[:, 0] - mu[0]) / sd[0] # x
            data[:, 1] = (data[:, 1] - mu[1]) / sd[1] # u
            for p in range(self.max_order):
                data[:, 2+p] = data[:, 2+p] * (sd[0]**(p+1)) / sd[1]

        N = data.shape[0]
        if knn is None:
            # pairwise distances
            pair_dist = squareform(pdist(data))

            if diagnose:
                # plot to determine optimal epsilon
                D2 = pair_dist**2
                print('Pairwise distance median: ', np.median(D2))
                eps_grid = np.logspace(-5, 2, 40)
                S = [np.exp(-D2/e).sum() for e in eps_grid]
                plt.loglog(eps_grid, S, 'o-')
                plt.show()
            
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

        else:
            nbrs = NearestNeighbors(n_neighbors=knn).fit(data)
            dist, idx = nbrs.kneighbors(data)

            if diagnose:
                D2 = dist**2
                print('kNN distance median: ', np.median(D2))
                eps_grid = np.logspace(-5, 2, 40)
                S = [np.exp(-D2/e).sum() for e in eps_grid]
                plt.loglog(eps_grid, S, 'o-')
                plt.show()
            print('max kNN radius:', dist.max(), ' 3*sqrt(4*eps):', 3*np.sqrt(4*epsilon))

            rows = np.repeat(np.arange(N), knn)
            vals = np.exp(-(dist.ravel()**2)/(4*epsilon))
            K_hat = csr_matrix((vals, (rows, idx.ravel())), shape=(N, N))
            K_hat = K_hat.maximum(K_hat.T) # symmetrise

            q = np.asarray(K_hat.sum(1)).ravel()
            Dq = diags(q**(-alpha))
            K_hat = Dq @ K_hat @ Dq

            d = np.asarray(K_hat.sum(1)).ravel()
            Dd = diags(1.0/np.sqrt(d))
            P_sym = Dd @ K_hat @ Dd
            del Dd, q

            P_sym = (P_sym + P_sym.T) * 0.5
            print(f'sparsity: {P_sym.nnz / N**2:.2e}')

        # store
        self.eps = epsilon
        self.data = data
        self.N = N
        self.t_size = t_steps
        self.n_trajectory = n_trajectories
        self.linsys_keep_mask = self.make_mask()

        print(f'P_sym shape: {P_sym.shape}, d shape: {d.shape}')
        return P_sym, d, K_hat

    def make_mask(self, edge=5):
        # mask to discard start and ends of trajectories, where dphi is unstable
        assert self.t_size > 2*edge, f't_steps {self.t_size} too small for edge {edge}'
        mask = np.ones((self.n_trajectory, self.t_size), bool)
        mask[:, :edge] = mask[:, -edge:] = False
        mask = mask.ravel()
        return mask


    ########################### Spectral basis construction for SEC ###########################


    def check_eigsh_res(self, eigvals, eigvecs_sym, P_sym):
        res = np.abs(np.asarray(P_sym @ eigvecs_sym) - eigvals[None, :] * eigvecs_sym).max(axis=0)
        print(f'eigsh residual: max {res.max():.2e} at mode {res.argmax()}')
        return res
    
    def check_laplacian_res(self, K_hat, d):
        E = self.eigvecs
        LgE = (d[:, None] * E - np.asarray(K_hat @ E)) / (self.eps * d[:, None])
        res = np.abs(LgE - self.eigvals[None, :] * E).max(axis=0)
        print(f'Laplacian residual: max {res.max():.2e} at mode {res.argmax()}')

    def reconstruct_error(self, f, d, J, name: str=''):
        '''
        Relative error in reconstructing a coordinate function f from the
        leading M eigenfunctions. Diagnostic for basis adequacy.
        '''
        phi = self.eigvecs[:, :J]
        f_hat = phi.T @ (d * f)
        print(f'{name} reconstruction error at J={J}: ', np.linalg.norm(phi @ f_hat - f) / np.linalg.norm(f - f.mean()))

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

        print('Shape eigvecs: ', self.eigvecs.shape)
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


    ########################### Spectral frame construction with SEC ###########################


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
            n_frames=48, thres=None, 
            spectral_energy=False
    ):
        '''
        Computes the frame rep coefficients of eigen 1-forms of the Hodge Laplacian
        Returns:
            frame_coeffs: (P, n_frames)
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
        del c1
        # (i,j,k,l) -> (i,k,j,l)
        c2_iljk = np.einsum('iljk->ijkl', c2)
        c2_ikjl = np.einsum('ikjl->ijkl', c2)
        del c2

        lam = self.eigvals
        lam_i = lam[:, None, None, None]
        lam_j = lam[None, :, None, None]
        lam_k = lam[None, None, :, None]
        lam_l = lam[None, None, None, :]

        # term1 = (lam_i + lam_j + lam_k + lam_l) * (c1_iljk - c1_ikjl)
        diff = c1_iljk - c1_ikjl
        term1 = np.zeros_like(diff)
        buf = np.empty_like(diff)
        for ax, lm in enumerate([
            lam[:,None,None,None], lam[None,:,None,None],
            lam[None,None,:,None], lam[None,None,None,:]
        ]):
            np.multiply(diff, lm, out=buf)
            term1 += buf
        del diff, buf
        term2 = c2_ikjl - c2_iljk
        del c1_iljk, c2_iljk, c2_ikjl, lam_i, lam_k

        term1 += term2
        E_hat = term1
        self.check_E_hat(E_hat)
        E_hat_flat, _ = self.flatten_antisym_4tensor(E_hat, M)
        del term1, term2, E_hat

        # Hodge Grammian G (shape: M, M, M, M)
        G = c0 * (lam_j + lam_l)
        G -= c1_ikjl
        G /= 2.0
        del c0, c1_ikjl, lam_j, lam_l
        self.check_G(G)

        G_hat = G.copy()
        G_hat += np.einsum('jilk->ijkl', G)
        G_hat -= np.einsum('jikl->ijkl', G)
        G_hat -= np.einsum('ijlk->ijkl', G)
        G_hat_flat, idxs = self.flatten_antisym_4tensor(G_hat, M)
        self.check_G_hat_flat(G_hat_flat)
        print(f'G_hat_flat shape: {G_hat_flat.shape},')
        del G_hat

        G1_flat = G_hat_flat + E_hat_flat
        G1_flat += G1_flat.T
        G1_flat /= 2.0
        print('G1_flat shape: ', G1_flat.shape)

        H = G - np.einsum('jikl->ijkl', G)
        print(f'H shape: {H.shape}')
        del G

        self.antisym_idx = idxs
        h, U = np.linalg.eigh(G1_flat)
        del G1_flat
        idx = np.argsort(h)[::-1]
        h, U = h[idx], U[:, idx]
        if thres is not None:
            h11 = h[0]
            keep = h > (h11 * thres)
            print(f'Keeping {np.sum(keep)} out of {len(h)} eigenvectors with threshold {thres}')
            U_tilde = U[:, keep]
        else:
            U_tilde = U[:, :n_frames] # (P, n_frames)

        self.n_frames = U_tilde.shape[1]
        print(f'Shape U_tilde: {U_tilde.shape}')

        L = U_tilde.T @ E_hat_flat @ U_tilde
        B = U_tilde.T @ G_hat_flat @ U_tilde
        L = (L + L.T) / 2.0
        B = (B + B.T) / 2.0
        print(f'Shape L: {L.shape}, Shape B: {B.shape}')
        del E_hat_flat

        nu, a = sp.linalg.eigh(L, B) # (n_frames,), (n_frames, n_frames)
        print(f'Shape nu: {nu.shape}, Shape a: {a.shape}')
        print(f'nu spectrum: {nu[:8]}, {nu[-2:]}')
        print('nu nullspace dim:', np.sum(np.abs(nu) < 1e-6))

        # coefficients for all basis eigen 1-forms
        frame_coeffs = U_tilde @ a # (P, n_frames)
        Gram_forms = frame_coeffs.T @ G_hat_flat @ frame_coeffs
        del G_hat_flat
        print('rho orthonormal under G_hat?', np.abs(Gram_forms - np.eye(self.n_frames)).max())
        print(f'Eigen frame coefficients shape: {frame_coeffs.shape}')
        del L, B, U_tilde

        return frame_coeffs, H

    def framecoeff_to_vec(self, frame_coeffs, H, X_hat, mode=0):
        # select mode to convert
        ef_mode = frame_coeffs[:, mode]
        idx_i, idx_j = self.antisym_idx
        ef_mat = self.unflatten_antisym_1tensor(ef_mode, self.M, idx_i, idx_j)
        V = self.sharp(H, ef_mat)
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

    def plot_laplacian_eigenfuncs(self, idxs: List = [0, 10, 20, 30]):
        fig = plt.figure(figsize=(15,4))
        for n, j in enumerate(idxs):
            ax = fig.add_subplot(1, 4, n+1, projection='3d')
            ax.scatter(
                self.data[:,0], self.data[:,1], self.data[:,2],
                c=self.eigvecs[:, j], cmap='RdBu', s=2
            )
            ax.set_title(f'{j}-th basis eigenfunction')
        plt.show()
        
    def plot_vector_field(self, V, normalise=75, scale=1.0, x_scale=1.0, subsample: int=None, title: str=''):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')

        if type(normalise) == int:
            V_plot = self.normalize_by_percentile(V, percentile=normalise)
        elif normalise == 'mean':
            V_plot = self.normalize_by_mean(V)
        elif normalise == 'norm':
            V_plot = self.normalize_by_norm(V)

        X, Y, Z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        U, V, W = V_plot[0]/x_scale, V_plot[1], V_plot[2] # arrow components at each point

        if subsample:
            idx = np.random.choice(len(X), subsample, replace=False)
            # points
            ax.scatter(X[idx], Y[idx], Z[idx], s=3, c='navy', alpha=0.8)
            # arrows
            ax.quiver(
                X[idx], Y[idx], Z[idx], 
                U[idx], V[idx], W[idx], 
                length=scale, normalize=False, color='coral', arrow_length_ratio=0.05
            )
        else:
            ax.scatter(X, Y, Z, s=3, c='navy', alpha=0.8)
            ax.quiver(
                X, Y, Z, U, V, W, 
                length=scale, normalize=False, color='coral', arrow_length_ratio=0.05
            )

        ax.set_xlabel('x'); ax.set_ylabel('u'); ax.set_zlabel('dxu')
        ax.set_box_aspect([np.ptp(X)*x_scale, np.ptp(Y), np.ptp(Z)])
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


    ########################### SEC related operations ###########################


    def fourier_tf(self, f, d):
        '''
        Fourier transform function(s) to spectral space: (?, N) @ (N, M) -> (?, M)
        f: may be (N, 1) ... (N, dim) hence ?
        '''
        f = f.reshape(len(f), -1) # ensure 2D array
        return (f * d[:, None]).T @ self.eigvecs
        
    def inv_fourier_pushfwd(self, f_hat, V):
        '''
        Apply V operator (pushforward) then inverse fourier transform back to jet space: 
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


    ########################### Linear system setup methods ###########################


    def frames_to_operators(self, frame_coeffs, H):
        idx_i, idx_j = self.antisym_idx
        Vks = []
        for k in range(self.n_frames):
            rho_k_flat = frame_coeffs[:, k]
            rho_k = self.unflatten_antisym_1tensor(rho_k_flat, self.M, idx_i, idx_j) # (M, M)
            Vk = self.sharp(H, rho_k) # (M, M)
            Vks.append(Vk)
        return Vks

    def compute_WX(self,):
        '''
        W = d_x + u1 d_u + u2 d_u1 + ... so its components W(x) ... W(u_p) ARE the data columns,
        shifted by one, with W^x = 1. No operator, no differentiation, no error.
        Has 1 less dimension than the jet space.
        '''
        dims = (self.max_order + 1)
        N = self.N
        comps = np.zeros((dims, N))
        comps[0] = 1.0
        for a in range(1, dims):
            # W(u_{a-1}) = u_a so we just stack data
            comps[a] = self.data[:, a + 1]
        print('W(X) shape = (equation order + 1, N)? ', comps.shape)
        return comps

    def df_dx(self, F, sg_win=11, sg_poly=3):
        '''
        Derivative of sampled functions along trajectories, i.e. W(F) = D_x F.

        Uses the fact that trajectories ARE the integral curves of W, so the
        along-trajectory derivative is exactly the action of W. No operator
        composition involved.

        F: (dim_F, N) or (N,) values sampled at the data points
        returns: same shape, dF/dx along each trajectory
        '''
        F = F.reshape(-1, self.N)
        dim_F = F.shape[0]
        x = self.data[:, 0].reshape(-1, self.t_size)
        dx = np.diff(x, axis=1).mean()

        Fr = F.reshape(dim_F, -1, self.t_size) # (dim_F, n_trajectories, t)
        dF = savgol_filter(
            Fr, window_length=sg_win, 
            polyorder=sg_poly,
            deriv=1, delta=dx, axis=2
        )
        out = dF.reshape(dim_F, -1) # (dim_F, N)
        return out

    def jetlie_bracket(
            self, V_op, WX, X_hat, d,
            sg_win=11, sg_poly=3
        ):
        '''
        Compute Z = [V, W] in coordinate jet space, using only SINGLE applications
        of each operator, avoiding the matrix product V_op @ W_op which SEC DOES NOT SUPPORT.

            [V,W] = V(W(X)) - W(V(X))

        - V(W(X)): spectral pushforward of W's components through V_op.
        One application of V. SEC supports this.
        - W(V(X)): derivative of V's components ALONG trajectories, since the
        trajectories are integral curves of W. One application of W, computed
        by finite differences rather than by the operator.

        V_op: (M, M) operator representation of V
        WX: (dims, N) components of W in coordinates, i.e. WX = W(x), W(u), W(u1), ...
        returns Z: (dims, N) components of the bracket in coordinates
        '''
        dims = WX.shape[0]

        # term 1: V(WX) single application of V to each component of vec field coordinates WX
        WX_hat = self.fourier_tf(WX.T, d) # (dims, M)
        VW = self.inv_fourier_pushfwd(WX_hat, V_op) # (dims, N)

        # term 2: W(V^a) V's components, differentiated along trajectories
        VX = self.inv_fourier_pushfwd(X_hat, V_op) # (max_order + 2, N)
        WV = self.df_dx(VX, sg_win, sg_poly)[:dims] # (dims, N)
        return VW - WV

    def jetlie_bracket_batch(self, Vks, WX, X_hat, d, sg_win=11, sg_poly=3, chunk=32, verbose=True):
        '''
        Batched version of jetlie_bracket: computes Z_k = [V_k, W] 
        in jet coordinates for every frame element, vectorised together.

        Processes #chunk frame elements at a time to bound peak memory:
        a full batch needs K * dims_X * N floats, which is very large in GB

        Params:
            - Vks: list of K (M, M) operators, or an array (K, M, M)
            - WX: (dims, N) components of W, dims = max_order + 1
            - X_hat: (dims_X, M) spectral coefficients of the coordinate functions, dims_X = max_order + 2
        Returns:
            Z_all: (K, dims, N)
        '''
        Vs = np.asarray(Vks) # (K, M, M)
        K = Vs.shape[0]
        dims = WX.shape[0]
        N = self.N

        WX_hat = self.fourier_tf(WX.T, d) # (dims, M)

        Z_all = np.empty((K, dims, N))

        n_chunks = int(np.ceil(K / chunk))
        for ci in range(n_chunks):
            lo, hi = ci * chunk, min((ci + 1) * chunk, K)
            VsT = np.transpose(Vs[lo:hi], (0, 2, 1)) # (k, M, M)

            # term 1: V(W(X)) -- one application of each V to W's components
            VW = np.einsum('am,kmn,pn->kap', WX_hat, VsT, self.eigvecs)

            # term 2: W(V(X)) -- V's components differentiated along trajectories
            VX = np.einsum('am,kmn,pn->kap', X_hat, VsT, self.eigvecs)
            k, A, _ = VX.shape
            WV = self.df_dx(VX.reshape(k * A, N), sg_win, sg_poly)
            WV = WV.reshape(k, A, N)[:, :dims]

            Z_all[lo:hi] = VW - WV

            if verbose and n_chunks > 1:
                print(f'  bracket chunk {ci+1}/{n_chunks}', end='\r')

        if verbose:
            print(f'Batched Lie brackets: Z_all {Z_all.shape}' + ' ' * 20)

        return Z_all

    def build_columns(self, Z_all):
        '''
        Turn batched brackets into the columns of the linear system.

        For each contact form p = 1..P:  gamma^p(Z) = Z^{u_{p-1}} - u_p * Z^x
        stacked over p, giving one (P*N,) column per frame element.

        Z_all: (K, dims, N)
        returns: (P*N, K)
        '''
        Zx = Z_all[:, 0] # (K, N)
        blocks = [
            Z_all[:, p] - self.data[None, :, 1 + p] * Zx
            for p in range(1, self.max_order + 1)
        ] # each (K, N)
        return np.concatenate(blocks, axis=1).T # (P*N, K)

    def determining_residual(self, V_op, W_components, X_hat, d, **kw):
        '''
        Determining-equation residual computed with the coordinate-space bracket.

        For each contact form p = 1..P:  gamma^p(Z) = Z^{u_{p-1}} - u_p * Z^x
        Returns the relative residual over the masked points.
        '''
        Z = self.jetlie_bracket(V_op, W_components, X_hat, d, **kw)
        m = self.linsys_keep_mask

        Zx = Z[0]
        res, nrm = [], Zx**2
        for p in range(1, self.max_order + 1):
            Zu = Z[p]
            res.append((Zu - self.data[:, 1 + p] * Zx)[m])
            nrm = nrm + Zu**2
        res = np.concatenate(res)
        return np.linalg.norm(res), np.linalg.norm(np.sqrt(nrm)[m])

    def fit_analytic_syms(self, gens, Vks, X_hat, ridge=0.0):
        '''
        Fit analytic generators into the frame's span, in pushforward space.
        gens: (G, N, max_order + 2) generators evaluated on the data
        Returns coeffs (G, n_frames) and per-generator relative fit error.
        '''

        Vpf = np.stack([self.inv_fourier_pushfwd(X_hat, Vk) for Vk in Vks])
        m = np.tile(self.linsys_keep_mask, self.max_order + 2)
        Vmat = Vpf.reshape(self.n_frames, -1).T[m] # (dims*N, n_frames)

        coeffs, errs = [], []
        for g in gens:
            target = g.T.ravel()[m]
            if ridge > 0:
                reg = np.sqrt(ridge * np.linalg.norm(Vmat)**2 / self.n_frames)
                A = np.vstack([Vmat, reg*np.eye(self.n_frames)])
                b = np.concatenate([target, np.zeros(self.n_frames)])
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
            else:
                c, *_ = np.linalg.lstsq(Vmat, target, rcond=None)
            coeffs.append(c)
            errs.append(np.linalg.norm(Vmat @ c - target) / np.linalg.norm(target))
        return np.array(coeffs), np.array(errs)

    def principal_angles(self, V_ops, X_hat, gens):
        V_recs = []
        for V_op in V_ops:
            V = self.inv_fourier_pushfwd(X_hat, V_op)
            V_recs.append(V)

        dims = X_hat.shape[0]
        m = np.tile(self.linsys_keep_mask, dims)

        R = np.stack([V[:dims].ravel()[m] for V in V_recs]).T
        A_ = np.stack([g[:, :dims].T.ravel()[m] for g in gens]).T
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        A_ = A_ / np.linalg.norm(A_, axis=0, keepdims=True)

        Qr, _ = np.linalg.qr(R)
        Qa, _ = np.linalg.qr(A_)
        s = np.linalg.svd(Qr.T @ Qa, compute_uv=False)
        print('principal angles (deg):', np.round(np.degrees(np.arccos(np.clip(s, -1, 1))), 2))


    ########################### Determining null-space problem ###########################

    
    def solve(self, Z_all, max_kernel_dim: int=6, svd_thres: float=None, plot_A_sv=False):
        '''
        Build and solve linear system with SVD for constant coefficients of V
        '''
        mask = self.linsys_keep_mask
        tiled_mask = np.tile(mask, self.max_order)

        ### BUILD LINEAR SYSTEM ###
        print('')
        print('##### BUILDING LINEAR SYSTEM #####')
        A = self.build_columns(Z_all)[tiled_mask]
        print(f'A norm: {np.linalg.norm(A)}')

        ### SOLVE ###
        print('')
        print('##### SOLVING LINEAR SYSTEM #####')
        MACHINE_ZERO = 1e-12

        # T, T_errs = self.trivial_subspace(Vks, WX, X_hat)
        # A_def, P = self.deflate_trivial(A, T)
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
        sv = S / S[0]
        valid = np.where(sv > MACHINE_ZERO)[0]
        print(f'A_def rank {len(valid)} of {len(S)}')
        print('lowest valid sv:', sv[valid[-max_kernel_dim-2:]])
        params_set = [Vt[i] for i in valid[-max_kernel_dim:][::-1]]
        if plot_A_sv:
            plt.semilogy(S/S[0], 'o-')
            plt.xlabel('index')
            plt.ylabel('Scaled singular values')
            plt.show()

        return params_set