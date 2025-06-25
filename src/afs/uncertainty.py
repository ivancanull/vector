import numpy as np
from sympy import false
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

# Physics models for particle interaction
PHYSICS_MODEL = {
    'coulomb': {
        'force': lambda d, q1, q2: q1*q2/(d**2),  # Coulomb's law (1/r² repulsion)
        'decay': 0.95  # Velocity decay coefficient
    },
    'spring': {
        'force': lambda d, k, _: k*d,  # Hooke's law (linear repulsion)
        'decay': 0.98
    },
    'gravitational': {
        'force': lambda d, m, _: m/(d**2 + 1e-10),  # Gravitational attraction
        'decay': 0.92
    }
}

def uncertainty_aware_distribute(
    x: np.ndarray,
    u: np.ndarray,
    sample_num: int,
    lambda_param: float = 0.3
):
    """
    Uncertainty-aware distribution function for a given input and uncertainty.
    Args:
    - x (np.ndarray): Input data points.
    - u (np.ndarray): Uncertainty values corresponding to the input data points.
    - sample_num (int): Number of samples to generate from the distribution.

    Returns:
    - samples (np.ndarray): Samples drawn from the uncertainty-aware distribution.
    """

    pdf_u = u / (np.sum(u))
    pdf_mix = lambda_param * (1 / len(x)) + (1 - lambda_param) * pdf_u
    samples = np.random.choice(x, size=sample_num, replace=false, p=pdf_mix)
    return samples.reshape(-1)  # Ensure samples are in the correct shape

def partition_by_uncertainty(x, u, n_samples):
    
    # Create a dense grid for numerical integration
    x_idx = np.arange(len(x))
    
    # Compute cumulative uncertainty integral
    cumulative_u = cumtrapz(u, x_idx, initial=0)
    total_u = cumulative_u[-1]

    # Create normalized cumulative uncertainty (0 to 1)
    item_u = total_u / n_samples
    inv_cdf = interp1d(cumulative_u, x_idx)
    partition_points = []
    for i in range(n_samples):
        partition_points.append(int(inv_cdf(i * item_u))) 
    partition_points.append(len(x_idx) - 1)  # Ensure last point is the end of the range
    
    sample_idx = []
    for i in range(len(partition_points) - 1):
        left = partition_points[i]
        right = partition_points[i + 1]
        # Sample uniformly within each partition
        sample_idx.append(np.random.choice(x[left:right + 1], size=1)[0])
    sample_idx = np.array(sample_idx)
    return np.sort(sample_idx)

def uncertainty_integrated_sampling(a, b, n_samples, uncertainty_func):
    """
    Generate samples such that each subinterval has equal uncertainty mass
    
    Parameters:
    a, b - Interval bounds
    n_samples - Total number of samples to generate
    uncertainty_func - Uncertainty function U(x)
    
    Returns:
    samples - Array of sample positions
    """
    # Create partition points
    partition_points = partition_by_uncertainty(a, b, uncertainty_func, n_samples)
    
    # Generate one sample per subinterval at its midpoint
    samples = []
    for i in range(len(partition_points) - 1):
        left = partition_points[i]
        right = partition_points[i+1]
        samples.append((left + right) / 2)
    
    return np.array(samples)

# Advanced version: Generate multiple samples per interval with density proportional to uncertainty
def adaptive_uncertainty_sampling(a, b, n_samples, uncertainty_func, min_samples_per_interval=1):
    """
    Generate samples with density proportional to local uncertainty
    
    Parameters:
    a, b - Interval bounds
    n_samples - Total number of samples
    uncertainty_func - Uncertainty function U(x)
    min_samples_per_interval - Minimum samples per subinterval
    
    Returns:
    samples - Array of sample positions
    """
    # Determine optimal number of intervals
    n_intervals = max(min_samples_per_interval, int(np.sqrt(n_samples)))
    
    # Create partition points
    partition_points = partition_by_uncertainty(a, b, uncertainty_func, n_intervals)
    
    # Calculate uncertainty mass in each interval
    interval_masses = []
    for i in range(len(partition_points) - 1):
        left = partition_points[i]
        right = partition_points[i+1]
        # Simple approximation: uncertainty at midpoint times width
        interval_masses.append(uncertainty_func((left+right)/2) * (right - left))
    
    total_mass = sum(interval_masses)
    
    # Allocate samples proportional to interval uncertainty mass
    samples = []
    for i in range(len(partition_points) - 1):
        left = partition_points[i]
        right = partition_points[i+1]
        
        # Calculate number of samples for this interval
        interval_fraction = interval_masses[i] / total_mass
        n_interval_samples = max(min_samples_per_interval, 
                                int(round(n_samples * interval_fraction)))
        
        # Generate uniformly spaced samples within interval
        interval_samples = np.linspace(left, right, n_interval_samples + 2)[1:-1]
        samples.extend(interval_samples)
    
    # Handle possible rounding errors in sample count
    if len(samples) > n_samples:
        # Remove extra samples from longest intervals
        samples = sorted(samples)
        while len(samples) > n_samples:
            # Find largest gap
            gaps = np.diff(samples)
            max_gap_idx = np.argmax(gaps)
            # Remove sample after largest gap
            del samples[max_gap_idx + 1]
    
    elif len(samples) < n_samples:
        # Add samples to highest uncertainty regions
        n_needed = n_samples - len(samples)
        uncertainty_values = uncertainty_func(np.array(samples))
        sorted_indices = np.argsort(uncertainty_values)[::-1]  # Highest first
        
        for i in range(n_needed):
            idx = sorted_indices[i % len(sorted_indices)]
            # Create new sample near existing high-uncertainty sample
            perturbation = 0.01 * (b - a) * np.random.randn()
            new_sample = samples[idx] + perturbation
            # Ensure within bounds
            new_sample = max(a, min(b, new_sample))
            samples.append(new_sample)
    
    return np.sort(np.array(samples))

def uniform_distribute(
    x: np.ndarray,
    sample_num: int,
):
    """
    Uniform distribution function for a given input.
    Args:
    - x (np.ndarray): Input data points.
    - sample_num (int): Number of samples to generate from the distribution.

    Returns:
    - samples (np.ndarray): Samples drawn from the uniform distribution.
    """

    indices = np.linspace(0, len(x)-1, num=sample_num, dtype=np.float32)
    indices = np.round(indices).astype(np.int32)
    indices = np.clip(indices, 0, len(x)-1)
    return x[indices].reshape(-1)  # Ensure samples are in the correct shape

def random_distribute(
    x: np.ndarray,
    sample_num: int,
):
    """
    Random distribution function for a given input.
    Args:
    - x (np.ndarray): Input data points.
    - sample_num (int): Number of samples to generate from the distribution.

    Returns:
    - samples (np.ndarray): Samples drawn from the random distribution.
    """

    indices = np.random.choice(len(x), size=sample_num, replace=False)
    return x[indices].reshape(-1)  # Ensure samples are in the correct shape

def electro_grav_sampling(x, sample_num, u, lambda_gravity=0.3,
                          physics_model='coulomb', max_iter=500, 
                          init_strategy='uniform', verbose=True):
    """
    Adaptive sampling based on electrostatic/gravitational heuristics
    
    Args:
    - a, b: Sampling interval [a, b]
    - sample_num: Total number of samples
    - u: Uncertainty function U(x)
    - physics_model: Physics model: 'coulomb', 'spring', or 'gravitational'
    - max_iter: Maximum iterations for optimization
    - init_strategy: Initialization strategy: 'uniform', 'random', or 'uncertainty'
    - verbose: Whether to print optimization progress
    
    Returns:
    Array of optimized sample positions
    """
    x_ = x.copy()
    x = np.arange(len(x), dtype=np.float32)  # Ensure x is a float array
    # 1. Initialize particle system
    if init_strategy == 'uniform':
        # Uniformly spaced initial particles
        particles = uniform_distribute(x, sample_num)
    elif init_strategy == 'random':
        # Random initial positions
        particles = random_distribute(x, sample_num)
    else:  # uncertainty-driven initialization
        # Create inverse CDF based on uncertainty profile
        particles = uncertainty_aware_distribute(x, u, sample_num)
    particles = np.sort(particles)  # Ensure particles are sorted

    # 2. Configure physics model parameters
    k = 10 * len(x) / sample_num  # Adaptive force constant
    
    # 3. Calculate particle "charges/masses" based on uncertainty
    charges = 1 - (u - np.min(u)) / (np.max(u) - np.min(u))   # Normalized charges
    
    # 4. Particle system optimization loop
    velocity = np.zeros(sample_num)  # Particle velocities
    best_particles = particles.copy()  # Best configuration found
    min_energy = float('inf')  # Minimum system energy
    print(particles)
    for iter in range(max_iter):
        # Calculate inter-particle forces
        forces = np.zeros(sample_num)
        total_energy = 0  # Total system energy
        
        # Calculate pairwise forces
        for i in range(sample_num):
            for j in range(i+1, sample_num):
                dist = particles[j] - particles[i]  # Distance between particles
                
                force_mag = charges[i] * charges[j] / dist ** 2 * k
                forces[i] += force_mag  # Particle i repelled by j
                forces[j] -= force_mag  # Particle j repelled by i
                
                # Accumulate energy (|F| * |dx|)
                total_energy += abs(force_mag) * abs(dist)
        # Add boundary constraining forces
        boundary_stiffness = 10.0
        for i in range(sample_num):
            if particles[i] < 0:
                # Force pushing particle back into domain
                forces[i] += boundary_stiffness * (0 - particles[i])
            elif particles[i] > len(x) - 1:
                # Force pushing particle back into domain
                forces[i] += boundary_stiffness * (len(x) - 1 - particles[i])
        
        # Track best configuration (lowest energy state)
        if total_energy < min_energy:
            min_energy = total_energy
            best_particles = particles.copy()
        
        # Update particle positions (damped Euler integration)
        damping = 0.2 * 0.95 **iter  # Decaying damping coefficient
        velocity = damping * velocity + forces  # Update velocity
        
        particles += velocity  # Update position
        
        # Ensure particles stay within bounds
        particles = np.clip(particles, 0, len(x) - 1)
        
        # Maintain particle order (critical for 1D systems)
        particles.sort()
        print(particles)
    best_particles = np.unique(best_particles).astype(int)  # Remove duplicates
    best_particles = x_[best_particles]
    return best_particles
       
