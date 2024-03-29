# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:11:04 2019

@author: dn16018

Randomised benchmarking non-Clifford gates for one qubit
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
import sys
from scipy.optimize import curve_fit
import numba
from numba import prange
up_spin = np.array([[1, 0], [0, 0]], dtype=complex)
down_spin = np.array([[0, 0], [0, 1]], dtype=complex)

# Clifford group
II = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PPP = (-II + 1j*X + 1j*Y + 1j*Z)/2
PPM = (-II + 1j*X + 1j*Y - 1j*Z)/2
PMM = (-II + 1j*X - 1j*Y - 1j*Z)/2
MMM = (-II - 1j*X - 1j*Y - 1j*Z)/2
MMP = (-II - 1j*X - 1j*Y + 1j*Z)/2
MPP = (-II - 1j*X + 1j*Y + 1j*Z)/2
PMP = (-II + 1j*X - 1j*Y + 1j*Z)/2
MPM = (-II - 1j*X + 1j*Y - 1j*Z)/2
cliffords = np.array([II, X, Y, Z, PPP, PPM, PMM, MMM, MMP, MPP, PMP, MPM])
paulis = np.array([II, X, Y, Z])
B = (1/np.sqrt(6-2*np.sqrt(3)))*np.array([[1-1j, np.sqrt(3)-1],
                                         [np.sqrt(3)-1, -1-1j]], dtype=complex)


def c_matrix_vander(n):
    """
    C_n matrix used to generate the non-Clifford gate sets, generated as a
    Vandermonde matrix.
    for 4*n
    C_n := [[exp(+i*2pi/n), 0],
            [0, exp(-i*2pi/n)]]
    Args:
        n (int): A number

    Returns:
        ndarray: A block matrix of C_n matrices.
    """

    exp = np.exp(1j*np.pi/n)
    exp_n = np.array([[exp, 0], [0, exp.conj()]], dtype=complex)
    c_matrix = np.vander(exp_n.ravel(), n,
                         increasing=True)[:, 1:].swapaxes(0, 1).reshape(n-1,
                                                                        2, 2)
    return np.concatenate(([II], c_matrix))


def non_clifford_group(n):
    """
    4*n Non-Clifford construction (2-designs)
    e.g. n = 3
    12-Cliffords := {II, X, Y, Z} x {II, D, D^2} where D = B*C_3*B^{dagger}

    for 4*n
    non-Cliffords := {B^{dagger} x {II, X, Y, Z} x B x {II, C_{2n},...,
                     C_{2n}^{n-1}}}
    Args:
        n (int): A number

    Returns:
        ndarray: A block matrix of 2-designs gate set.
    """

    conjugated_pauli = (B.conj().T@paulis@B).reshape(4, 1, 2, 2)
    return (conjugated_pauli@c_matrix_vander(n)).reshape(4*n, 2, 2)


def depol_error(mean, std_dev, sample_size):
    """
    Depolarizing quantum error channel.
    D(rho) -> (1-p)*rho + (p/3)*(X*rho*X + Y*rho*Y + Z*rho*Z)

    Args:
        noise_mean (float): Mean of the Gaussian distribution
        std_dev (float): Standard deviation of the Gaussian distribution
        sample_size (int): Size of the random number generated.

    Returns:
        ndarray: A block matrix of depolarizing error matrices.
    """

    # Randomly drawing from a Gaussian distribution
    param = np.random.normal(mean, std_dev, (1, sample_size))

    # Redraw elements that are negative from the Gaussian distribution
    np.putmask(param, param < 0, np.random.normal(mean, std_dev))

    # Broadcast the drawn numbers to compute ndarrays of error matrices
    XYZ = ([np.sqrt(param/3)]*np.array([[X, Y, Z]]).T).T
    III = ([np.sqrt(1-param)]*np.array([[II]]).T).T
    return np.hstack((III, XYZ)).transpose(1, 0, 3, 2)


def frame_potential(designs, t):
    """
    Computes the frame potential for t-designs, defined as:
    frame_potential -> (1/K^t)*sum_{i=1, j=1}^n |Tr[U_{i}^{dagger}*U_{j}]|^{2t}

    Args:
        designs (2x2 matrices): A list of t-designs gate set
        t (int): A number

    Returns:
        number (float): The result of computing the frame potential.
    """

    # Computes hermitian tranpose of the t-designs
    designs_herm = designs.transpose(0, 2, 1).conj()

    # Computes the trace of sums of matrix multiplication
    traced_list = np.trace(designs_herm.reshape(len(designs), 1, 2, 2)@designs,
                           axis1=2, axis2=3)
    return np.sum(np.abs(traced_list)**(2*t))/((len(designs))**t)


def operator_groups(sample_size, n):
    """
    Generates the original Clifford gate set, or Clifford gate set by a basis
    change via matrix B or the non-Clifford gate sets generated by C_n matrix.

    Args:
        sample_size (int): Size of the ndarray/number of operations running in
                           parallel.
        n : n=None    -> Original Clifford gate set
            n="pauli" -> Just the Pauli gate set
            n=int     -> Non Clifford gate set
    Returns:
        ndarray: A block matrix of unitary error matrices.
    """

    if n is "pauli":
        return paulis[np.random.choice(paulis.shape[0], sample_size)]
    elif n is "og":
        return cliffords[np.random.choice(cliffords.shape[0], sample_size)]
    else:
        gate_set = non_clifford_group(n)
        return gate_set[np.random.choice(gate_set.shape[0], sample_size)]


def unitary_error(sd, sample_size):
    """
    Given 3 randomly Gaussian distributed parameters (a,b,c), the unitary noise
    is defined as:
    U(a,b,c) -> [[exp(-i(a+c)/2)*cos(b/2), exp(-i(a-c)/2)*sin(b/2)],
                 [-exp(i(a-c)/2)*sin(b/2), exp(+i(a+c)/2)*cos(b/2)]]
    Here an ndarray of parameters (a,b,c) is randomly generated from a Gaussian
    distribution, broadcasted to calculate the matrix elements and then
    reshaped to give an ndarray of unitary errors matrices for vectorization.

    Args:
        sd (float): Std. dev of the Gaussian distribution with mean = 0.
        sample_size (int): Size of the ndarray/number of operations running in
                           parallel.

    Returns:
        ndarray: A block matrix of unitary error matrices.
    """

    # Random sampling from Gaussian distribution
    params = np.random.normal(0, sd, (sample_size, 3))

    # Computes error matrix elements
    e_1 = np.exp(-1j*(params[:, 0]+params[:, 2])/2)*np.cos(params[:, 1]/2)
    e_2 = np.exp(-1j*(params[:, 0]-params[:, 2])/2)*np.sin(params[:, 1]/2)
    return np.array([[e_1, e_2], [-e_2.conj(), e_1.conj()]],
                    dtype=complex).transpose(2, 0, 1)


def pl_rep_operation(matrix):
    op_sum = np.zeros((4,4), dtype=complex)
    for i in range(len(matrix)):
        op_sum += np.kron(matrix[i], matrix[i].conj())
    return op_sum


def compute_channel_operation(rho, gates):
    """
    Given a quantum state's density function rho, the effect of the
    channel on this state is:
    rho -> sum_{i=1}^n E_i * rho * E_i^dagger

    Args:
        rho (2x2 matrix): A density function array of shape (2,2)
        gates (list): List of quantum gates(matrices)

    Returns:
        matrix: The result of applying the list of quantum gates
    """

    return np.sum(gates@rho@gates.transpose(0, 1, 3, 2).conj(), axis=0)


def init_tensor(input_state, sample_size):
    """
    Given a quantum state(density operator), this function generates an amount
    of it according to "sample_size" into a block matrix/tensor/ndarray, for
    vectorization purposes.

    Args:
        input_state (2x2 matrix): A density function array of shape (2,2)
        sample_size (int): Number of matrices within the block matrix
    Returns:
        ndarray: An ndarray with dim (sample_size, 2, 2) with input_state as
                 entries.
    """

    return np.broadcast_to(input_state, (sample_size,)+input_state.shape)

def operator_groups2(sample_size, n):
    """
    Generates the original Clifford gate set, or Clifford gate set by a basis
    change via matrix B or the non-Clifford gate sets generated by C_n matrix.

    Args:
        sample_size (int): Size of the ndarray/number of operations running in
                           parallel.
        n : n=None    -> Original Clifford gate set
            n="pauli" -> Just the Pauli gate set
            n=int     -> Non Clifford gate set
    Returns:
        ndarray: A block matrix of unitary error matrices.
    """

    if n is "pauli":
        return paulis[np.random.choice(paulis.shape[0], sample_size)]
    elif n is "og":
        return cliffords[np.random.choice(cliffords.shape[0], sample_size)]
    else:
        gate_set = non_clifford_group(n)
        gate_set = np.array([np.kron(gate_set[i], gate_set[i].conj())
                             for i in range(len(gate_set))])
        return gate_set[np.random.choice(gate_set.shape[0], sample_size)]


def gen_truncated(minimum, maximum, ave, sigma, max_size):
    #(0.05, 0.05, 0.985, 0.15) 0.9, 1 
    einval = np.random.normal(ave, sigma, max_size)
    index = (einval > minimum) & (einval < maximum)
    return einval[index]


def get_tau(einval1, einval2, einval3):
    max_tau = 1 - abs(np.array([einval1, einval2, einval3]))
    tau = max_tau*(2*np.random.uniform(0,1,(3,max_tau.shape[1]))-1)
    
    return tau

def q(e):
    return (1+e[0,:]+e[1,:]+e[2,:])*(1+e[0,:]-e[1,:]-e[2,:])*\
           (1-e[0,:]+e[1,:]-e[2,:])*(1-e[0,:]-e[1,:]+e[2,:])


def z_eta(t: np.ndarray, l: np.ndarray):
    norm_t = norm(t, axis=0)
    sum_term = [(l[i,:]**2)*(2*(t[i,:]**2) - norm_t**2) for i in range(3)]
    condition = (norm_t**4 - 2*norm_t**2 - 2*np.sum(sum_term, axis=0) + q(l))
    
    return condition


def gen_sigma(minimum, maximum, ave, sigma, max_size, samp_size):
    lambda1 = gen_truncated(minimum, maximum, ave, sigma, max_size)
    lambda2 = gen_truncated(minimum, maximum, ave, sigma, max_size)
    if len(lambda1) < len(lambda2):
        lambda2 = lambda2[0:len(lambda1)]
    else:
        lambda1 = lambda1[0:len(lambda2)]

    lambda3 = gen_truncated(minimum, maximum, ave, sigma, max_size)
        
    if len(lambda1) < len(lambda3):
        lambda3 = lambda3[0:len(lambda1)]
    else:
        lambda2 = lambda2[0:len(lambda3)]
        lambda1 = lambda1[0:len(lambda3)]
    
    index = (1+lambda3 > abs(lambda1+lambda2)) & \
            (1-lambda3 > abs(lambda2-lambda1))
    
    lambda1, lambda2, lambda3 = lambda1[index], lambda2[index], lambda3[index]
    tau = get_tau(lambda1, lambda2, lambda3)
    lambdas = np.array([lambda1, lambda2, lambda3])
    
    cond = norm(tau, axis=0)**2 < (1 - np.sum(np.square(lambdas), axis=0) + \
                                  2*lambda1*lambda2*lambda3)
    index = cond & (z_eta(tau, lambdas) > 0)

    tau = tau[:,index]
    lambda1, lambda2, lambda3 = lambda1[index], lambda2[index], lambda3[index]

    if len(lambda1) <= samp_size:
        raise ValueError("Need to generate a bigger max_size.")
    else:
        pass
    
    tau = tau[:, 0:samp_size]
    tau = np.vstack((np.ones(tau.shape[1]), tau))
    lambda1 = lambda1[0:samp_size]
    lambda2 = lambda2[0:samp_size]
    lambda3 = lambda3[0:samp_size]
    lambdas = np.array([lambda1, lambda2, lambda3])

    sigma_gen = np.zeros((samp_size, 4, 4))
    sigma_gen[:, range(1, 4), range(1, 4)] = lambdas.T
    sigma_gen[:, :, 0] = tau.T
    
    return sigma_gen


def create_rotation(samp_size, r) -> np.ndarray:
    
    angles = np.random.normal(0,1,(3,samp_size))*r
    sin_angles = np.sin(angles)
    cos_angles = np.cos(angles)
    
    left_angles = np.array([[cos_angles[0], -sin_angles[0]],
                            [sin_angles[0], cos_angles[0]]])
    mid_angles = np.array([[cos_angles[1], -sin_angles[1]],
                           [sin_angles[1], cos_angles[1]]])
    right_angles = np.array([[cos_angles[2], -sin_angles[2]],
                            [sin_angles[2], cos_angles[2]]])
    
    left = np.zeros((samp_size, 3, 3))
    left[:, 0:2, 0:2] = left_angles.T
    left[range(0, samp_size), 2:3, 2:3] = np.ones((samp_size, 1, 1))
    
    mid = np.zeros((samp_size, 3, 3))
    mid[:, 1:3, 1:3] = mid_angles.T
    mid[range(0, samp_size), 0:1, 0:1] = np.ones((samp_size, 1, 1))
    
    right = np.zeros((samp_size, 3, 3))
    right[:, 0:2, 0:2] = right_angles.T
    right[range(0, samp_size), 2:3, 2:3] = np.ones((samp_size, 1, 1))
    
    rotation = np.zeros((samp_size, 4, 4))
    rotation[:, 1:4, 1:4] = left@mid@right
    rotation[range(0, samp_size), 0:1, 0:1] = np.ones((samp_size, 1, 1))
    
    # rotation = init_tensor(np.eye(4), samp_size)
    return rotation


def gen_channel(r1, r2, ave, sigma, max_size, samp_size):
    
    channel = create_rotation(samp_size, r1)@\
              gen_sigma(0.9, 1, ave, sigma, max_size, samp_size)@\
              create_rotation(samp_size, r2)
    
    return channel


def randomized_benchmarking2(input_state: np.ndarray, seq_len: int,
                             samp_size: int, noise_mean: float,
                             noise_sd: float, n: int, input_state_2):
    seq = []
    # input state needs to be 1x4 matrix
    rho_tensor = init_tensor(input_state, samp_size)
    
    # if input_state_2 is not None:
    rho_tensor_2 = init_tensor(input_state_2, samp_size)
    
    channel1 = init_tensor(np.eye(4, dtype=complex), samp_size)
    channel2 = init_tensor(np.eye(4, dtype=complex), samp_size)
    
    for j in range(1, seq_len+1):
        q_gates = operator_groups2(samp_size, n)
        unit_noise = gen_channel(0.06, 0.06, 0.998, 0.04, 10000, samp_size)
        channel1 = unit_noise@q_gates@channel1

        unit_noise2 = gen_channel(0.06, 0.06, 0.998, 0.04, 10000, samp_size)        
        channel2 = unit_noise2@q_gates@channel2
        seq.append(q_gates)
            
    inverse_gate = np.array(seq[::-1])[0]
    for el in np.array(seq[::-1])[1:]:
        inverse_gate = inverse_gate@el
        
    
    seq_adjoint = inverse_gate.transpose(0, 2, 1).conj()
    
    prep_noise = gen_channel(0.05, 0.05, 0.985, 0.15, 10000, samp_size)
    prep_noise2 = gen_channel(0.05, 0.05, 0.985, 0.15, 10000, samp_size)
    
    output_state_up = seq_adjoint@channel1@prep_noise@rho_tensor
    output_state_down = seq_adjoint@channel2@prep_noise2@rho_tensor_2

    meas_noise = gen_channel(0.05, 0.05, 0.98, 0.15, 10000, samp_size)
    meas_noise2 = gen_channel(0.05, 0.05, 0.98, 0.15, 10000, samp_size)
    
    measure_up = init_tensor(np.array([Z.flatten()]),samp_size)
    measure_down = init_tensor(np.array([Z.flatten()]),samp_size)

    expected_val_up = measure_up@meas_noise@output_state_up
    expected_val_down = measure_down@meas_noise2@output_state_down
    ave_fid = (np.average(expected_val_up)-np.average(expected_val_down))/2
    
    return np.real(ave_fid)


def randomized_benchmarking(input_state, seq_length, sample_size,
                            noise_mean, noise_sd, noise2_sd, n,
                            input_state_2=None):
    sequence = []

    # State preparation noise map
    prep_noise = depol_error(noise_mean, noise_sd, sample_size)
    rho_tensor = init_tensor(input_state, sample_size)
    rho = compute_channel_operation(rho_tensor, prep_noise)

    if input_state_2 is not None:
        prep_noise_2 = depol_error(noise_mean, noise_sd, sample_size)
        rho_tensor_2 = init_tensor(input_state_2, sample_size)
        rho_2 = compute_channel_operation(rho_tensor_2, prep_noise_2)

    for j in range(1, seq_length+1):
        # Gate operation noise map
        depol_noise = depol_error(noise_mean, noise_sd, sample_size)
        unit_noise = unitary_error(noise2_sd, sample_size)
        q_gates = operator_groups(sample_size, n)
        sequence.append(q_gates)

        i_ideal_operator = unit_noise@depol_noise@q_gates
        rho = compute_channel_operation(rho, i_ideal_operator)

        if input_state_2 is not None:
            depol_noise_2 = depol_error(noise_mean, noise_sd,
                                        sample_size)
            unit_noise_2 = unitary_error(noise2_sd, sample_size)
            i_ideal_operator_2 = unit_noise_2@depol_noise_2@q_gates
            rho_2 = compute_channel_operation(rho_2, i_ideal_operator_2)
            

    # Final CPTP map of random noises
    depol_noise = depol_error(noise_mean, noise_sd, sample_size)
    unit_noise = unitary_error(noise2_sd, sample_size)

    # Computes the Hermitian of the inverse of the forward gates sequence
    inverse_gate = np.array(sequence[::-1])[0]

    # Computes multi dot products
    for el in np.array(sequence[::-1])[1:]:
        inverse_gate = inverse_gate@el

    # Final ideal&noisy density operator
    unitary_undo = inverse_gate.transpose(0, 2, 1).conj()
    f_ideal_operator = unit_noise@depol_noise@unitary_undo
    f_noisy_operator = compute_channel_operation(rho, f_ideal_operator)

    # Final measurement noise map
    spam_noise = depol_error(noise_mean, noise_sd, sample_size)

    rho_tensor = init_tensor(Z, sample_size)
#    rho_tensor = init_tensor(input_state, sample_size)

    noisy_input_state = compute_channel_operation(rho_tensor, spam_noise)

    # Measure observables and compute average fidelity
    traced = np.trace(noisy_input_state@f_noisy_operator, axis1=1, axis2=2)
    avg_fidelity = np.average(traced)
#    print(avg_fidelity)
    if input_state_2 is not None:
        depol_noise_2 = depol_error(noise_mean, noise_sd, sample_size)
        unit_noise_2 = unitary_error(noise2_sd, sample_size)
        f_ideal_operator_2 = unit_noise_2@depol_noise_2@unitary_undo
        f_noisy_operator_2 = compute_channel_operation(rho_2,
                                                       f_ideal_operator_2)
        spam_noise_2 = depol_error(noise_mean, noise_sd, sample_size)

        rho_tensor_2 = init_tensor(Z, sample_size)
#        rho_tensor_2 = init_tensor(input_state, sample_size)

        noisy_input_state_2 = compute_channel_operation(rho_tensor_2,
                                                        spam_noise_2)
        traced_2 = np.trace(noisy_input_state_2@f_noisy_operator_2,
                            axis1=1, axis2=2)
        avg_fidelity_2 = np.average(traced_2)
        # print(avg_fidelity_2)
#        print("\n")
#        print(avg_fidelity)
#        print("\n")
#        print(avg_fidelity_2)
#        print("\n")
#        print((avg_fidelity - avg_fidelity_2)/2)
        return np.real((avg_fidelity - avg_fidelity_2)/2)
    return np.real(avg_fidelity)


def exponential(s, a, b, c):
    """
    A scaled exponential function for curve fitting specifically for RB.
    y = a(2*b - 1)^s + c

    Args:
        s, a, b, c : array-like, dtype=float

    Returns:
         list: The result of passing the x-parameters through this function.

    """
    return a * (2 * b - 1)**s + c


def decay(s, a, f):
    """
    A scaled exponential function for curve fitting specifically for RB.
    y = a*f^s

    Args:
        s, a, f: array-like, dtype=float

    Returns:
         list: The result of passing the x-parameters through this function.

    """
    return a * (2*f - 1)**s


def estimate_ideal_fid(noise_mean, noise_sd, noise2_sd, sample_size):
    depol_noise = depol_error(noise_mean, noise_sd, sample_size)
    unit_noise = unitary_error(noise2_sd, sample_size)
    traced = np.abs(np.trace(unit_noise@depol_noise, axis1=2, axis2=3))**2
    return np.average((2 + np.sum(traced, axis=0))/6)

# @numba.njit()
def get_data(input_state, seq_length, sample_size, noise_mean,
             noise_sd, noise2_sd, data_ss, n, input_state_2=None):

    length, fidelity = [], []

    for s in range(1, seq_length+data_ss, data_ss):
        avg_fidelity = randomized_benchmarking(input_state,
                                               s,
                                               sample_size,
                                               noise_mean,
                                               noise_sd,
                                               noise2_sd,
                                               n,
                                               input_state_2)

        sys.stdout.write("\r" + "gate applied: " + str(s))
        length.append(s)
        fidelity.append(avg_fidelity)

#    print("\n\n[INFO] Input state\t\t: {}".format(str(input_state)))
#    print("[INFO] Input state\t\t: {}".format(str(input_state_2)))
#    print("[INFO] Sequence length\t\t: {}".format(seq_length))
#    print("[INFO] Sequence samples\t\t: {}".format(sample_size))
#    print("[INFO] Depolarizing noise mean\t: {}".format(noise_mean))
#    print("[INFO] Depolarizing std. dev\t: {}".format(noise_sd))
#    print("[INFO] Unitary noise std. dev\t: {}".format(noise2_sd))
#    print("[INFO] n\t\t\t: {}".format(n))
    ideal_fid = (0.5 + (1 - 4*noise_mean/3)*(1 + np.exp(-0.5*noise2_sd**2) +
                 np.exp(-noise2_sd**2))*(1/6)*np.exp(-0.5*noise2_sd**2))
    return length, fidelity, seq_length, ideal_fid


def get_data2(input_state, seq_len, samp_size, noise_mean,
             noise_sd, data_ss, n, input_state_2):

    length, fidelity = [], []

    for s in range(1, seq_len+data_ss, data_ss):
        avg_fidelity = randomized_benchmarking2(input_state, s, samp_size,
                                                1, noise_sd, n, input_state_2)

        sys.stdout.write("\r" + "gate applied: " + str(s))
        length.append(s)
        fidelity.append(avg_fidelity)

#    print("\n\n[INFO] Input state\t\t: {}".format(str(input_state)))
#    print("[INFO] Input state\t\t: {}".format(str(input_state_2)))
#    print("[INFO] Sequence length\t\t: {}".format(seq_length))
#    print("[INFO] Sequence samples\t\t: {}".format(sample_size))
#    print("[INFO] Depolarizing noise mean\t: {}".format(noise_mean))
#    print("[INFO] Depolarizing std. dev\t: {}".format(noise_sd))
#    print("[INFO] Unitary noise std. dev\t: {}".format(noise2_sd))
#    print("[INFO] n\t\t\t: {}".format(n))
    # ideal_fid = (0.5 + (1 - 4*noise_mean/3)*(1 + np.exp(-0.5*noise2_sd**2) +
    #              np.exp(-noise2_sd**2))*(1/6)*np.exp(-0.5*noise2_sd**2))
    return length, fidelity, seq_len


def fit_curve(x_data, y_data, seq_length, data_ss, plot):

    popt, pcov = curve_fit(decay, x_data, y_data, bounds=(0, [1., 1.]))

    if plot:
        plt.figure(figsize=(15, 7.5))
        t = np.arange(1, seq_length+data_ss, data_ss)
        error = np.sqrt(np.diag(pcov))
        variance = np.average(np.abs(decay(t, *popt) - np.array(y_data)))
        best_fit_vals = tuple(np.vstack((popt, error)).ravel('F'))

        fit_label = 'fit: a=%.10f$\\pm$%.10f\n   ' + '   f=%.10f$\\pm$%.10f'

        plt.plot(x_data, decay(t, *popt), 'r-',
                 label=fit_label % best_fit_vals)
        plt.plot(x_data, y_data, 'b-', label='data: $\\sigma^2$=%7.4f' %
                 (variance))

        plt.title("1 Qubit Randomized Benchmarking", fontsize=20)
        plt.xlabel("Clifford length", fontsize=20)
        plt.ylabel("Fidelity", fontsize=20)
        plt.xlim(0, seq_length)
        plt.ylim(0, 1)
        plt.legend(shadow=True, fontsize=20)
        plt.show()

#    print("[INFO] Parameter f \t\t\t: {}".format(popt[1]))
#    print("[INFO] Difference\t\t\t: {}\n".format(ideal_fid - popt[1]))
    return popt[1]

# here end_point could be ascending max sequence length or sequence sample
# depends where u put the j
# @numba.njit()
def ascending_plot(exps, end_point, stepsize, input_state, noise_mean,
                   noise_sd, noise2_sd, data_ss, n, input_state_2, plot):
    ave_fid_fit_plot = []
    ideal_fit = [] # np.arange(1, end_point+stepsize, stepsize)
    for j in iter([1000, 2000, 4000, 6000, 8000, 10000]):
        ave_fid_fit = []
        ideal_number = []
        for i in range(1, exps+1):
            length, fidelity, seq_length, ideal_fid = get_data(input_state,
                                                               200,
                                                               j,
                                                               noise_mean,
                                                               noise_sd,
                                                               noise2_sd,
                                                               data_ss,
                                                               n,
                                                               input_state_2)
    #            print("[INFO] Ideal gate fidelity\t: {}\n{}".format(ideal_fid,
    #                                                                    "-"*100))
    
            ideal_fid_fit = fit_curve(length, fidelity, seq_length, data_ss,
                                      plot)
#            string = "\nj = {}, i = {},\nIdeal fidelity fit\t: " + \
#                     "{}\nIdeal fidelity \t\t: {}\n"
#            print(string.format(j, i, ideal_fid_fit, ideal_fid))
            print("\ni = {}".format(i))
            ave_fid_fit.append(ideal_fid_fit)
            ideal_number.append(ideal_fid)
#    string = "j = {}, averaged over {} experiments gives: {}\n"
#    print(string.format(j, i, np.average(ave_fid_fit)))
        print("\nj = {}".format(j))

        ave_fid_fit_plot.append(ave_fid_fit)
        ideal_fit.append(np.average(ideal_number))
    print(ave_fid_fit_plot)
#    import pdb; pdb.set_trace()
    # remember to change this if u change get_dat(200) above
#    resources = resources * 200

    plt.figure(figsize=(15, 7.5))
#    diff_1 = np.array(ideal_fit) - ave_fid_fit_plot

#    # plot the ideal fidelity line
#    header = "Resource Number\t\t # Ave. Fidelity\t\t # Variance\t\t\t " + \
#             "# Std. dev\t\t\t # Analytical Ideal Fid\t\t " + \
#             "# (AIF - Ave. Fid)\t\t # Estimated Ideal Fid\t\t " + \
#             "# (EIF - Ave Fid)"
#    filename = repr(exps) + 'exps, 200gates, seq sample(1,1001,50) ' + \
#               'noise(' + repr(noise_mean) + ',' + repr(noise_sd) + ',' + \
#               repr(noise2_sd) + ') n=' + repr(n)
#    np.savetxt(filename+'.txt', np.stack((resources, ave_fid_fit_plot,
#                                          variance, np.sqrt(variance),
#                                          ideal_fit, diff_1), axis=1),
#               delimiter=',\t ', header=header)
#    fit_label = r'Ideal $\mathcal{F}$        : %.10f'
#    plt.plot(resources, ideal_fit, 'r-', label=fit_label % (ideal_fit[0]))
#    plt.errorbar(resources, ave_fid_fit_plot, yerr=np.sqrt(variance), fmt="x",
#                 ecolor='g', elinewidth=2, markersize=10)
#    plt.title(filename, fontsize=20)
    plt.xlabel("Maximum sequence length", fontsize=20)
#    plt.ticklabel_format(axis='x', style='sci', scilimits=(4, 3))
    plt.ylabel("Estimated Fidelity", fontsize=20)
#    plt.ylim(ideal_fit[0]-0.0005, ideal_fit[0]+0.0005)
    plt.boxplot(ave_fid_fit_plot)
    labels = ('1000', '2000', '4000', '6000', '8000')
    plt.xticks(np.arange(len(labels))+1,labels)
#    plt.xlim(0,800)
    plt.legend(shadow=True, fontsize=20, loc='best')
#    plt.savefig(filename+'.png')
    plt.show()


#for k in [100]:
#    for i in iter([5,6]):  # for (a, b) it starts at a ends at b-1
# starttime = time.time()
# ascending_plot(exps=100,
#                end_point=1001,
#                stepsize=50,
#                input_state=up_spin,
#                noise_mean=0.01,
#                noise_sd=0.001,
#                noise2_sd=0.1,
#                data_ss=10,
#                n=3,
#                input_state_2=down_spin,
#                plot=False)
# timeElapsed = time.time() - starttime
# print("Time elapsed: {} seconds".format(timeElapsed))
starttime = time.time()
# randomized_benchmarking2(np.array([[1],[0],[0],[0]]), 100, 10, 1, 0.1, 3,
#                           np.array([[0],[0],[0],[1]]))
length, fidelity, seq_len = get_data2(np.array([[1],[0],[0],[0]]), 100,
                                      50, 1, 0.1, 1, 3,
                                      np.array([[0],[0],[0],[1]]))
ideal_fid_fit = fit_curve(length, fidelity, seq_len, 1, plot=True)
# length, fidelity, seq_length, ideal_fid = get_data(up_spin,
#                                                   200,
#                                                   50,
#                                                   0,#0.01
#                                                   0,#0.001
#                                                   0.1,
#                                                   1,
#                                                   3,
#                                                   down_spin)
# ideal_fid_fit = fit_curve(length, fidelity, seq_length, 1, plot=True)
timeElapsed = time.time() - starttime
print("\nTime elapsed: {} seconds".format(timeElapsed))

