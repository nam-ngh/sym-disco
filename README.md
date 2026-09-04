## Internal symmetry discovery algorithm for ordinary differential systems

Recovers internal symmetries from the point cloud of unknown ordinary differential systems by resolving the geometry of the equation manifold and imposing internal contact conditions.

#### Running the algorithm on provided datasets
1. Generate data by calling the data/generator.py file with the equation name argument. For example,
   ```
   python data/generator.py --eq population
   ```
   Should the user like to change the data generation config for the same equation, they should adjust the corresponding ```configs/population.yaml``` file accordingly.

2. Run the algorithm by calling the main.py file.
   The supported flags include
   --plot_w (the vector field spanning the Cartan distribution),
   --plot_eigvals (spectrum of the 0-Laplacian),
   --plot_eigbasis (eigenfunctions of the 0-Laplacian),
   --plot_eigframe (basis tangent vector fields from the 1-Laplacian)
   and --skip_solve (to disable building the linear system, used for plotting purposes).
   For example,
   ```
   python main.py --eq population --plot_eigframe
   ```
   will run the algorithm on the population.npy dataset generated, along with showing a plot of a basis tangent vector field.

#### Running the algorithm on new datasets
1. Store your dataset in data/ in .npy format, for example ```data/custom.npy```.
   Note the dataset should contain stacked trajectories over a range of initial conditions instead of 1 long trajectory,
   ideally with uniform spacing in the first column (the independent variable).
   E.g., If you have 10 data points per trajectories, the first 10 rows should increase linearly in $x$, then $x$ resets on row 11 for the second trajectory and so on.
   Columns should be jet space coordinates $[x, u, u_1, ..., u_P]$. Data shape should be (N, P+2) where N = t_steps $\times$ n_trajectories.
3. Make an appropriate ```custom.yaml``` in configs/, consult example.yaml for recommendations
4. Call main to solve
   
   ```
   python main.py --eq custom 
   ```
