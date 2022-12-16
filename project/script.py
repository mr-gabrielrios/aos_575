import numpy as np

def poisson_ms(F):
    ''' Poisson matrix system generator for a doubly-periodic 2D domain. '''

    # Input grid (this is the field you're analyzing for, typically pressure)
    # F = np.zeros(shape=(5, 5)) 
    # Get dimensions (height and width) of the grid
    h, w = F.shape
    # Initialize Poisson LHS grid (this is the 'A' in the matrix system 'Ax = b')
    A = np.full(shape=(h*w, h*w), fill_value=0)

    for j in range(0, h):
        for i in range(0, w):
            # Create empty array for the point at (j, i)
            arr = np.full(shape=(h*w), fill_value=0)
            # Calculate array index (in other words, what element number are we looking at)
            m = j*h + i
            # Get left, right, upper, and lower indices
            lt, rt, up, dn = (i-1) % w + j*h, (i+1) % w + j*h, ((j+1) * h + i) % (w*h), ((j-1) * h + i) % (w*h)
            # Assign weights
            if j == 0:
                arr[m], arr[lt], arr[rt], arr[up], arr[dn] = -4, 1, 1, 1, 0
            elif j == h-1:
                arr[m], arr[lt], arr[rt], arr[up], arr[dn] = -4, 1, 1, 0, 1
            else:
                arr[m], arr[lt], arr[rt], arr[up], arr[dn] = -4, 1, 1, 1, 1
            # Pop the array into the m-th row of the Poisson LHS grid
            A[m] = arr

    return A

def p_rhs(data, j, i, n, h, w, constants):
    rho_0, g = constants
    
    ''' Calculate the right-hand side (b). '''
    # Horizontal component
    d2p_x = (d1(data, 'u', 'x', j, i, n)**2 + 
             data['u'][n, j, i]*d2(data, 'u', 'z', j, i, n) + 
             d1(data, 'w', 'x', j, i, n)*d1(data, 'u', 'z', j, i, n) +
             data['w'][n, j, i]*md2(data, 'u', j, i, n))
    # Vertical component
    d2p_z = (d1(data, 'u', 'z', j, i, n)*d1(data, 'w', 'x', j, i, n) + 
             data['u'][n, j, i]*md2(data, 'w', j, i, n) + 
             d1(data, 'w', 'z', j, i, n)**2 +
             data['w'][n, j, i]*d2(data, 'w', 'z', j, i, n) - 
             d1(data, 'b', 'z', j, i, n))
    # Combine
    d2p = -rho_0 * (d2p_x + d2p_z)
    # print('\t d2p/dx2: {0:.4e}; d2p/dz2: {1:.4e}; d2p = {2:.4e}'.format(d2p_x, d2p_z, d2p))
    
    return d2p

def d1(data, var_name, dim_name, j, i, n, method='cdf2'):
    ''' First-order differentiation of a field.
    
    Inputs:
    - data:     the data dictionary
    - var_name: string with field name used for indexing 'data'
    - dim_name: dimension over which differentiation occurs
    - j, i:     index in the 2D grid
    - method:   differentiation scheme (default: second-order centered-difference, 'cdf2')
    '''
    
    # Define working data
    var = data[var_name][n, :, :]
    # Get field grid shape
    h, w = var.shape
    # Define working dimension
    dim = data[dim_name]
    # Initialize result
    res = np.nan
    
    if method == 'cdf2':
        if dim_name == 'x':
            res = (var[j, ((i+1) % w)] - var[j, ((i-1) % w)])/(2*(dim[j, ((i+1) % w)] - dim[j, i]))
        elif dim_name == 'z':
            res = (var[((j+1) % h), i] - var[((j-1) % h), i])/(2*(dim[((j+1) % h), i] - dim[j, i]))
    
    return res

def d2(data, var_name, dim_name, j, i, n, method='cdf2'):
    ''' Second-order differentiation of a field.
    
    Inputs:
    - data:     the data dictionary
    - var_name: string with field name used for indexing 'data'
    - dim_name: dimension over which differentiation occurs
    - j, i:     index in the 2D grid
    - method:   differentiation scheme (default: second-order centered-difference, 'cdf2')
    '''
    
    # Define working data
    var = data[var_name][n, :, :]
    # Get field grid shape
    h, w = var.shape
    # Define working dimension
    dim = data[dim_name]
    # Initialize result
    res = np.nan
    
    if method == 'cdf2':
        if dim_name == 'x':
            res = (var[j, ((i+1) % w)] - 2*var[j, i] + var[j, ((i-1) % w)])/(dim[j, ((i+1) % w)] - dim[j, i])**2
        elif dim_name == 'z':
            res = (var[((j+1) % h), i] - 2*var[j, i] + var[((j-1) % h), i])/(dim[((j+1) % h), i] - dim[j, i])**2
            
    return res

def md2(data, var_name, j, i, n, method='cdf2'):
    ''' Second-order differentiation for mixed partials.
    
    Inputs:
    - data:     the data dictionary
    - var_name: string with field name used for indexing 'data'
    - j, i:     index in the 2D grid
    - method:   differentiation scheme (default: second-order centered-difference, 'cdf2')
    '''
    
    # Define working data
    var = data[var_name][n, :, :]
    # Get field grid shape
    h, w = var.shape
    # Initialize result
    res = np.nan
    
    if method == 'cdf2':
        res = (var[((j+1) % h), ((i+1) % w)] - var[((j+1) % h), ((i-1) % w)] - var[((j-1) % h), ((i+1) % w)] - var[((j-1) % h), ((i-1) % w)])/(4*(data['z'][((j+1) % h), i] - data['z'][j, i])*(data['x'][j, ((i+1) % w)] - data['x'][j, i]))
        
    return res

def fxn_u(data, j, i, n, constants):
    rho_0, g = constants
    return (-data['u'][n, j, i]*d1(data, 'u', 'x', j, i, n) - 
            data['w'][n, j, i]*d1(data, 'u', 'z', j, i, n) - 
            (1/rho_0)*data['p'][n, j, i]*d1(data, 'p', 'x', j, i, n))

def fxn_w(data, j, i, n, constants):
    rho_0, g = constants
    return (-data['u'][n, j, i]*d1(data, 'w', 'x', j, i, n) - 
            data['w'][n, j, i]*d1(data, 'w', 'z', j, i, n) - 
            (1/rho_0)*data['p'][n, j, i]*d1(data, 'p', 'z', j, i, n) + 
            data['b'][n, j, i])

def temp_disc(value, dt, method='rk4'):
    # Initialize result
    res = np.nan
    
    if method == 'rk4':
        k1 = value
        k2 = value + (dt/2)*k1
        k3 = value + (dt/2)*k2
        k4 = value + (dt)*k3
        res = (k1 + 2*k2 + 2*k3 + k4)/6
        
    return res


def grid_setup():

    ''' Grid basis setup. '''
    
    # Define grid point parameters
    x_min, x_max = [0, 6]
    z_min, z_max = [0, 6]
    dx, dz = 0.2, 0.2
    # Define basis "vectors" (bv) to outline grid formation (step size added to maximum bound to include it)
    bv_x = np.arange(x_min, x_max+dx, dx)
    bv_z = np.arange(z_min, z_max+dz, dz)
    # Build base grid meshgrid
    base_x, base_z = np.meshgrid(bv_x, bv_z)
    
    # CFL number
    cfl = 0.1
    # Timestep (assume c = 1)
    dt = cfl*dx
    # Maximum time
    t_max = 3
    # Create time array
    times = np.arange(0, t_max+dt, dt)
    
    ''' Initialize field grids. '''
    
    # Define starter grid with dimensions of (X x Z x t)
    base_grid = np.full(shape=(len(times), len(bv_x), len(bv_z)), fill_value=0, dtype=float)
    
    # Define dynamic fields
    data = {}
    # Define fields of interest
    field_names = ['x', 'z', 'u', 'w', 'p', 'b']
    # Construct dictionary with initial values for each field
    data = {key: base_grid if key not in ['x', 'z'] 
            else base_x if key in ['x'] 
            else base_z for key in field_names}
    
    ''' Constants. '''
    # Reference density (rho_0)
    rho_0 = 1.225
    # Gravitational acceleration (m s^-2)
    g = 9.81
    
    # Define constants
    constants = [rho_0, g]
    
    return data, constants, [dt, times]

    
def run(data, constants, n, dt):
    
    # Initialize basis vectors for iteration
    x, z = data['x'][0, :], data['z'][:, 0]
    
    # Initialize right-hand side holding Poisson values
    b = np.full(shape=(len(z)*len(x)), fill_value=0, dtype=float)
    
    # Define start and end indices for each axis
    start_x, end_x = 0, len(x)
    start_z, end_z = 0, len(z)
    
    # Iterate over grid
    for j in range(start_z, end_z):
        for i in range(start_x, end_x):
            
            # print('Step: {0} | Position: ({1}, {2}) | Grid position: ({3}, {4}) | Reference values - start_z: {5}; end_z: {6}'.format(n, j, i, x[i], z[j], start_z, end_z))
            
            ''' Current time-stepping scheme: forward Euler. '''
            # Horizontal velocity equation
            data['u'][n+1, j, i] = data['u'][n, j, i] + dt*temp_disc(fxn_u(data, j, i, n, constants), dt, method='rk4')
    
            # Vertical velocity equation
            data['w'][n+1, j, i] = data['w'][n, j, i] + dt*temp_disc(fxn_w(data, j, i, n, constants), dt, method='rk4')
            
            ''' Fill Poisson RHS matrix. '''
            # Use boolean to control whether BC value added to RHS if at top or bottom boundary
            bool_bc = 1 if (j == start_z) or (j == end_z-1) else 0
            # Initialize BC value
            bc_value = 0
            # Assign BC value
            if bool_bc:
                if j == start_z:
                    bc_value = 0.0001 if n == 0 and (3*end_x//4-1 < i < 3*end_x//4+1) else 0
                elif j == end_z-1:
                    bc_value = 0.0001 if n == 0 and (2*end_x//4-1 < i < 2*end_x//4+1) else 0
                
            # print('({1}, {2}) | BC forcing value: {0}'.format(bool_bc*bc_value, j, i))
            
            # Populate Poisson RHS array
            b[len(z)*j+i] = p_rhs(data, j, i, n, len(z), len(x), constants) + bool_bc*bc_value
            # print('({0}, {1}) | Index: {2} | RHS: {3:.3e} | H/W = ({4}, {5})'.format(j, i, len(z)*j+i, b[len(z)*j+i], len(x), len(z)))
            
    A = poisson_ms(data['p'][n])
    
    p = np.linalg.solve(A, b).reshape(len(z), len(x))
    
    # Iterate over grid
    for l in range(start_z, end_z):
        for k in range(start_x, end_x):
            data['p'][n+1, l, k] = data['p'][n, l, k] + dt*temp_disc(p[l, k], dt, method='rk4')
    
    # data['b'][n] = data['p'][n, l, k]/np.diff(data['p'][1], axis=0)
    
    return data

def main():
    
    data, constants, [dt, times] = grid_setup()
    
    ''' Calculate dynamic fields (dynamic pressure, velocities). '''
    # Run the model
    for n in range(0, len(times)-1):
        data = run(data, constants, n, dt)
        
    return data, times
        
if __name__ == '__main__':
    data, times = main()