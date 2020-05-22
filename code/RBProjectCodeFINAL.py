# -*- coding: utf-8 -*-
"""
Created on Sat Sep 20 16:11:04 2019

@author: dn16018

Randomised benchmarking non-Clifford gates for one qubit
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
# plt.switch_backend("agg")#switch this on for bluecrystal
import time
from scipy.optimize import curve_fit
import math as m
from mpi4py import MPI

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
taskid = comm.Get_rank()
MASTER = 0
TAG1 = 1
TAG2 = 2
TAG3 = 3
TAG4 = 4
TAG5 = 5

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
pauli = m.sqrt(0.5)*paulis
B = (1/np.sqrt(6-2*np.sqrt(3)))*np.array([[1-1j, np.sqrt(3)-1],
                                         [np.sqrt(3)-1, -1-1j]], dtype=complex)


def c_matrix_vander(n: int) -> np.ndarray:
    """
    C_n matrix used to generate the non-Clifford gate sets, generated as a
    Vandermonde matrix.
    for 4*n
    C_n := [[exp(+i*2pi/n), 0],
            [0, exp(-i*2pi/n)]]

    Parameters
    ----------
    n : int
        A number.

    Returns
    -------
    np.concatenate(([II], c_matrix)) : np.ndarray
        A block matrix of C_n matrices.
    """

    exp = np.exp(1j*np.pi/n)
    exp_n = np.array([[exp, 0], [0, exp.conj()]], dtype=complex)
    c_matrix = np.vander(exp_n.ravel(), n,
                         increasing=True)[:, 1:].swapaxes(0, 1).reshape(n-1,
                                                                        2, 2)
    return np.concatenate(([II], c_matrix))


def non_clifford_group(n: int) -> np.ndarray:
    """
    4*n Non-Clifford construction (2-designs)
    e.g. n = 3
    12-Cliffords := {II, X, Y, Z} x {II, D, D^2} where D = B*C_3*B^{dagger}

    for 4*n
    non-Cliffords := {B^{dagger} x {II, X, Y, Z} x B x {II, C_{2n},...,
                     C_{2n}^{n-1}}}

    Parameters
    ----------
    n : int
        A number.

    Returns
    -------
    (conjugated_pauli@c_matrix_vander(n)).reshape(4*n, 2, 2) : ndarray
        A block matrix of 2-designs gate set.
    """

    conjugated_pauli = (B.conj().T@paulis@B).reshape(4, 1, 2, 2)
    return (conjugated_pauli@c_matrix_vander(n)).reshape(4*n, 2, 2)


def frame_potential(designs: np.ndarray, t: int) -> float:
    """
    Computes the frame potential for t-designs, defined as:
    frame_potential -> (1/K^t)*sum_{i=1, j=1}^n |Tr[U_{i}^{dagger}*U_{j}]|^{2t}

    Parameters
    ----------
    designs : np.ndarray
        (2x2 matrices) A list of t-designs gate set
    t : int
        A number.

    Returns
    -------
    np.sum(np.abs(traced_list)**(2*t))/((len(designs))**t) : float
        The result of computing the frame potential.
    """

    # Computes hermitian tranpose of the t-designs
    designs_herm = designs.transpose(0, 2, 1).conj()

    # Computes the trace of sums of matrix multiplication
    traced_list = np.trace(designs_herm.reshape(len(designs), 1, 2, 2)@designs,
                           axis1=2, axis2=3)
    return np.sum(np.abs(traced_list)**(2*t))/((len(designs))**t)


def pl_rep_operation(matrix: np.ndarray) -> np.ndarray:
    """
    Computes the sum of K_i \\otimes K_i^T for (2x2) matrices.

    Parameters
    ----------
    matrix : np.ndarray
        An array of (2x2) matrices.

    Returns
    -------
    op_sum : np.ndarray
        SUM_i K_i \\otimes K_i^T.
    """
    op_sum = np.zeros((4, 4), dtype=complex)
    for i in range(len(matrix)):
        op_sum += np.kron(matrix[i], matrix[i].conj())
    return op_sum


def init_tensor(input_state: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Given a quantum state(density operator), this function generates an amount
    of it according to "sample_size" into a block matrix/tensor/ndarray, for
    vectorization purposes.

    Parameters
    ----------
    input_state : np.ndarray
        A density matrix array of shape (2,2).
    sample_size : int
        Number of matrices within the block matrix.

    Returns
    -------
    np.broadcast_to(input_state, (sample_size,)+input_state.shape) : np.ndarray
        An ndarray with dim (sample_size, 2, 2) with input_state as entries.
    """

    return np.broadcast_to(input_state, (sample_size,)+input_state.shape)


def operator_groups2(sample_size: int, n) -> np.ndarray:
    """
    Generates the original Clifford gate set, or Clifford gate set by a basis
    change via matrix B or the non-Clifford gate sets generated by C_n matrix.

    Parameters
    ----------
    sample_size : int
        Size of the ndarray/number of operations running in parallel.
    n : None
        Original Clifford gate set.
    n : "pauli"
        Just the Pauli gate set.
    n : int
        Non Clifford gate set.

    Returns
    -------
    gateset : np.ndarray
        A block matrix of unitary error matrices.
    """

    if n == "pauli":
        return paulis[np.random.choice(paulis.shape[0], sample_size)]
    elif n == "og":
        return cliffords[np.random.choice(cliffords.shape[0], sample_size)]
    else:
        gateset = non_clifford_group(n)
        gateset = np.array([pl_rep_channel(pauli, gateset[i])
                            for i in range(len(gateset))])
        return gateset


def pl_channel(basis: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Convert a channel from 2x2 into 4x4 by a trace-orthonormal basis. When the
    basis is input as the normalised Pauli group, the channel converts into
    Pauli-Liouville representation. Using for loops. Quicker for small
    computations.

    Parameters
    ----------
    basis : np.ndarray
        An array of the (2x2) normalised Pauli matrices including the Identity.
    gate : np.ndarray
        The channel to be converted.

    Returns
    -------
    channel : np.ndarray
        A channel in Pauli-Liouville representation.
    """

    channel = np.zeros((4, 4), dtype=complex)
    for i in range(len(basis)):
        for j in range(len(basis)):
            channel[i][j] = np.trace(basis[i]@gate@basis[j]@gate.T.conj())
    return channel


def pl_rep_channel(basis: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """
    Convert a channel from 2x2 into 4x4 by a trace-orthonormal basis. When the
    basis is input as the normalised Pauli group, the channel converts into
    Pauli-Liouville representation. Using Numpy vectorisation. Quicker for
    large computations.

    Parameters
    ----------
    basis : np.ndarray
        An array of the (2x2) normalised Pauli matrices including the Identity.
    gate : np.ndarray
        The channel to be converted.

    Returns
    -------
    channel : np.ndarray
        A channel in any/Pauli-Liouville representation.
    """

    basis = np.array([basis, ]*len(basis))
    gate_herm = np.array([[gate.T.conj(), ]*len(basis), ]*len(basis))
    gate = np.array([[gate, ]*len(basis), ]*len(basis))
    return np.trace(basis.transpose(1, 0, 2, 3)@gate@basis@gate_herm, axis1=2, axis2=3)


def pl_ket(basis: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """
    Convert a quantum state in density operator formalism into representation
    of a given trace-orthonormal basis. When the basis is the normalised Pauli
    group, it becomes the Pauli-Liouville representation.

    Parameters
    ----------
    basis : np.ndarray
        An array of the (2x2) normalised Pauli matrices including the Identity.
    gate : np.ndarray
        The channel to be converted.

    Returns
    -------
    channel : np.ndarray
        A channel in any/Pauli-Liouville representation.
    """

    operator = np.array([operator, ]*len(basis))
    ket = np.trace(pauli@operator, axis1=1, axis2=2)
    return ket.reshape(4, 1)


def gen_truncated(minimum: float, maximum: float, ave: float, sigma: float,
                  max_size: int) -> np.ndarray:
    """
    Generate a bunch of values from a Gaussian distribution with a mean and
    standard deviation. Accept all values that lie within the minimum to
    maximum interval and discard the rest. This effectively makes it a
    truncated Gaussian distribution.

    Parameters
    ----------
    minimum : float
        Minimum value of the accepted interval.
    maximum : float
        Maximum value of the accepted interval.
    ave : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    max_size : int
        The maximum size of the values that were first drawn from the Gaussian
        distribution.

    Returns
    -------
    einval[index] : np.ndarray
        An array of values generated from the Gaussian that lie within the
        interval [minimum, maximum].

    """

    einval = np.random.normal(ave, sigma, max_size)
    index = (einval > minimum) & (einval < maximum)
    return einval[index]


def get_tau(einval1: float, einval2: float, einval3: float) -> np.ndarray:
    """
    Generate the non-unital part of the Pauli-Liouville noise map

    Parameters
    ----------
    einval1 : float
        Generated eigenvalue 1.
    einval2 : float
        Generated eigenvalue 2.
    einval3 : float
        Generated eigenvalue 3.

    Returns
    -------
    tau : np.ndarray
        An array of values for tau.
    """

    max_tau = 1 - abs(np.array([einval1, einval2, einval3]))
    tau = max_tau*(2*np.random.uniform(0, 1, (3, max_tau.shape[1]))-1)

    return tau


def q(e: np.ndarray) -> np.ndarray:
    """
    A function the checks part of the constraint that is needed to generate
    the non-unital part of a Pauli-Liouville noise map.

    Parameters
    ----------
    e : np.ndarray
        The previously generated array of eigenvalues.

    Returns
    -------
    np.ndarray
    """

    return (1+e[0, :]+e[1, :]+e[2, :])*(1+e[0, :]-e[1, :]-e[2, :])*\
           (1-e[0, :]+e[1, :]-e[2, :])*(1-e[0, :]-e[1, :]+e[2, :])


def z_eta(t: np.ndarray, l: np.ndarray) -> bool:
    """
    A function the checks part of the constraint that is needed to generate
    the non-unital part of a Pauli-Liouville noise map.

    Parameters
    ----------
    t : np.ndarray
        The previously generated array of tau values.
    l : np.ndarray
        The previously generated array of eigenvalues.

    Returns
    -------
    condition : bool
        The condition of satisfying the constraints.
    """

    norm_t = norm(t, axis=0)
    sum_term = [(l[i, :]**2)*(2*(t[i, :]**2) - norm_t**2) for i in range(3)]
    condition = (norm_t**4 - 2*norm_t**2 - 2*np.sum(sum_term, axis=0) + q(l))

    return condition


def gen_sigma(minimum: float, maximum: float, ave: float, sigma: float,
              max_size: int, samp_size: int) -> np.ndarray:
    """
    The main function that generates the middle matrix of the Pauli-Liouville
    noise map.

    Parameters
    ----------
    minimum : float
        Minimum value of the accepted interval.
    maximum : float
        Maximum value of the accepted interval.
    ave : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    max_size : int
        The maximum size of the values that were first drawn from the Gaussian
        distribution.
    samp_size : int
        The amount of values that satisfied all the previous constraints that
        were actually needed.

    Raises
    ------
    ValueError
        Stops working if the amount of values that satisfy the constraints are
        not enough for the amount that's needed.

    Returns
    -------
    sigma_gen : np.ndarray
        An array of many (4x4) such matrices.
    """

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

    cond = norm(tau, axis=0)**2 < (1 - np.sum(np.square(lambdas), axis=0) +
                                   2*lambda1*lambda2*lambda3)
    index = cond & (z_eta(tau, lambdas) > 0)

    tau = tau[:, index]
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


def create_rotation(samp_size: int, r: float) -> np.ndarray:
    """
    Create the rotation maps in Pauli-Liouville representation that is needed
    to realise a single qubit noise map.

    Parameters
    ----------
    samp_size : int
        The number of matrices needed.
    r : float
        The multiplier the for the small angle generation.

    Returns
    -------
    rotation : np.ndarray
        The (4x4) rotation matrices.
    """

    angles = np.random.normal(0, 1, (3, samp_size))*r
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

    return rotation


def gen_channel(r1: float, r2: float, ave: float, sigma: float, max_size: int,
                samp_size: int) -> np.ndarray:
    """
    Generate the single qubit random CPTP noise map in Pauli-Liouville
    representation where E = U Lambda V, where the U and V are rotations
    in Pauli-Liouville representation.

    Parameters
    ----------
    r1 : float
        The multiplier the for the small angle generation for U.
    r2 : float
        The multiplier the for the small angle generation for V.
    ave : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    max_size : int
        The maximum size of the values that were first drawn from the Gaussian
        distribution.
    samp_size : int
        The number of (4x4) noise maps needed.

    Returns
    -------
    channel : np.ndarray
        An array of the (4x4) noise maps.
    """

    channel = create_rotation(samp_size, r1)@\
              gen_sigma(0.9, 1, ave, sigma, max_size, samp_size)@\
              create_rotation(samp_size, r2)

    return channel


def fidelity(channel: np.ndarray) -> np.ndarray:
    """
    Returns the average gate fidelity of a channel in Pauli-Liouville
    representation.

    Parameters
    ----------
    channel : np.ndarray
        The channels.

    Returns
    -------
    TYPE
        An array of all the average gate fidelities.
    """

    diag = np.trace(channel, axis1=1, axis2=2) - 1
    d = m.sqrt(channel.shape[1])
    p = diag/(d**2-1)
    return ((d-1)*p + 1)/d


def unitarity(channel: np.ndarray) -> np.ndarray:
    """
    Returns the unitarity of a channel in Pauli-Liouville representation.

    Parameters
    ----------
    channel : np.ndarray
        The channels.

    Returns
    -------
    TYPE
        An array of all the unitarities.
    """

    d = m.sqrt(channel.shape[1])
    unitals = channel[:, 1:4, 1:4]
    trace = np.trace(unitals.transpose(0,2,1)@unitals, axis1=1, axis2=2)

    return (1/(d**2 - 1))*trace


def unitarity_percent(channel):
    fid = fidelity(channel)
    uni = unitarity(channel)
    d = m.sqrt(channel.shape[1])
    r = 1 - fid
    minU = (1-(d*r)/(d-1))**2
    return (uni-minU)/(1-minU)


def plotter(fidelity: np.ndarray, unitarity: np.ndarray, xlabel: str,
            ylabel: str):
    """
    Plots unitarity against average gate fidelity

    Parameters
    ----------
    fidelity : np.ndarray
        The average gate fidelities.
    unitarity : TYPE
        The unitarities.
    xlabel : str
        Label of x-axis.
    ylabel : str
        Label of y-axis.
    """

    plt.figure()
    plt.scatter(fidelity, unitarity, s=20, marker="x")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()


def noisespread():
    """
    Plots the spread of the fidelity of the noise maps as well as their
    estimated errors
    """

    noise = gen_channel(0.06, 0.06, 0.998, 0.04, 30000000, 1000000)
    plt.figure(figsize=(8, 6))
    plt.hist(fidelity(noise), bins = np.linspace(min(fidelity(noise)),
                                                 max(fidelity(noise)), 50),
             label="$10^7$ single qubit random noise maps $\\mathbf{\\Lambda}$")
    plt.ylabel("Number of channels generated", fontsize=20)
    plt.xlabel("Fidelity of channel to the Identity Channel", fontsize=20)
    plt.savefig("1", dpi=500)
    plt.legend(fontsize=15)

    plt.figure(figsize=(8, 6))
    plt.boxplot([(fidelity(noise)-np.average(fidelity(noise)))/np.average(fidelity(noise))], sym="x")
    plt.xlabel("$10^7$ single qubit random noise maps $\\mathbf{\\Lambda}$", fontsize=20)
    plt.ylabel("Estimate error as a fraction \n of average $\\bar{F}$", fontsize=20)
    plt.savefig("2", dpi=500)
    plt.show()


def rb_pauli_liouville(input_state: np.ndarray, seq_len: int,
                       samp_size: int, n: int, max_size: int,
                       input_state_2: np.ndarray) -> float:
    """
    Computes the sequence fidelity where the calculation of survival
    probabilities are done by Numpy vectorisation and broadcasting.

    Parameters
    ----------
    input_state : np.ndarray
        The first (up) input state in Pauli-Liouville representation.
    seq_len : int
        The sequence length of the RB curve.
    samp_size : int
        The amount of sample size to average the survival probabilties over
        for a particular sequence length.
    n : int
        The number that generates the size of the 4n Clifford and non-Clifford
        gate set.
    max_size : int
        The maximum size drawn from distributions such that samp_size amount
        of them fulfills the CPTP constraints, and generate the noise maps.
    input_state_2 : np.ndarray
        The second (down) input state in Pauli-Lioville representation.

    Returns
    -------
    float
        The sequence fidelity.
    float
        The standard deviation of the sequence fidelity that comes from
        averaging the survival probabilities.
    """

    seq = []

    gateset = operator_groups2(samp_size, n)
    gateset = gateset[np.random.choice(gateset.shape[0], (seq_len, samp_size))]

    prep_noise = gen_channel(0.05, 0.05, 0.955, 0.15, max_size, samp_size)
    prep_noise2 = gen_channel(0.05, 0.05, 0.955, 0.15, max_size, samp_size)

    # input state needs to be 1x4 matrix
    channel1 = prep_noise@init_tensor(input_state, samp_size)
    channel2 = prep_noise2@init_tensor(input_state_2, samp_size)

    unit_noise = np.array([gen_channel(0.06, 0.06, 0.998, 0.04, max_size, samp_size) for i in range(seq_len)])
    unit_noise2 = np.array([gen_channel(0.06, 0.06, 0.998, 0.04, max_size, samp_size) for i in range(seq_len)])

    for j in range(1, seq_len+1):
        q_gates = gateset[j-1, :]
        channel1 = unit_noise[j-1, :]@q_gates@channel1
        channel2 = unit_noise2[j-1, :]@q_gates@channel2

        seq.append(q_gates)

    inverse_gate = np.array(seq[::-1])[0]
    for el in np.array(seq[::-1])[1:]:
        inverse_gate = inverse_gate@el

    seq_adjoint = inverse_gate.transpose(0, 2, 1).conj()
    adj_noise1 = gen_channel(0.06, 0.06, 0.998, 0.04, max_size, samp_size)
    adj_noise2 = gen_channel(0.06, 0.06, 0.998, 0.04, max_size, samp_size)

    output_state_up = adj_noise1@seq_adjoint@channel1
    output_state_down = adj_noise2@seq_adjoint@channel2

    meas_noise = gen_channel(0.05, 0.05, 0.95, 0.15, max_size, samp_size)
    meas_noise2 = gen_channel(0.05, 0.05, 0.95, 0.15, max_size, samp_size)

    measure_up = init_tensor(m.sqrt(1/2)*np.array([[0, 0, 0, 2]]), samp_size)
    measure_down = init_tensor(m.sqrt(1/2)*np.array([[0, 0, 0, 2]]), samp_size)

    expected_val_up = np.real(measure_up@meas_noise@output_state_up)
    expected_val_down = np.real(measure_down@meas_noise2@output_state_down)

    ave_fid = (expected_val_up-expected_val_down)/2
    ave_fid_error = np.std(ave_fid)

    return np.average(ave_fid), ave_fid_error


def decay(s: float, a: float, f: float) -> float:
    """
    A scaled exponential function for curve fitting specifically for RB.
    y = a*f^s    

    Parameters
    ----------
    s : float
        Sequence length.
    a : float
        SPAM parameter.
    f : float
        Depolarising parameter.

    Returns
    -------
    float
        A value.
    """

    return a * (2*f - 1)**s


def get_data2(input_state: np.ndarray, seq_len: int, samp_size: int,
              data_ss: int, n: int, max_size: int, input_state_2: np.ndarray,
              plot=False):
    """
    Generate the sequence fidelity for many different value of sequence length.
    Parallellised by decomposing the for loops so that they can be ran under
    different cores. Assumed a scaling of sqrt(n).

    Parameters
    ----------
    input_state : np.ndarray
        The first (up) input state in Pauli-Liouville representation.
    seq_len : int
        The max sequence length of the RB curve.
    samp_size : int
        Number of sample size per sequence length for a particular sequence
        fidelity.
    data_ss : int
        The jump in plotted sequence length.
    n : int
        The number that generates the size of the 4n Clifford and non-Clifford
        gate set.
    max_size : int
        The maximum size of the values that were first drawn from the Gaussian
        distribution.
    input_state_2 : np.ndarray
        The second (down) input state in Pauli-Lioville representation.
    plot : bool, optional
        Plots the RB curve if True. The default is False.

    Returns
    -------
    length : np.ndarray
        An array of sequence length values.
    fidelity : np.ndarray
        An array of sequence fidelity for different sequence length values.
    seq_len : int
        Max seqence length.
    error : np.ndarray
        An array of the errors of the sequence fidelity array.
    """

    starttime = time.time()
    length = np.zeros(seq_len, dtype=np.float64)
    fidelity = np.zeros(seq_len, dtype=np.float64)
    error = np.zeros(seq_len, dtype=np.float64)

    if seq_len <= numtasks:
        istart = 1
        iend = istart+seq_len
    else:
        istart = int(seq_len*np.sqrt(taskid/numtasks)) + 1
        iend = int(seq_len*np.sqrt((taskid+1)/numtasks)) + 1
        if iend > seq_len:
            iend = seq_len+1
    print("Rank: ", taskid, " start: ", istart, " end: ", iend)
    for s in range(istart, iend, 1):
        avg_fidelity, fid_error = rb_pauli_liouville(input_state, s, samp_size,
                                                     n, max_size, input_state_2)
        length[s-1] = s
        fidelity[s-1] = avg_fidelity
        error[s-1] = fid_error

    if taskid != MASTER:
        if seq_len <= numtasks:
            pass
        else:
            offset = istart-1
            chunksize = iend - istart
            comm.send(offset, dest=MASTER, tag=TAG1)
            comm.send(chunksize, dest=MASTER, tag=TAG2)
            comm.Send(length[offset:offset+chunksize], dest=MASTER, tag=TAG3)
            comm.Send(fidelity[offset:offset+chunksize], dest=MASTER, tag=TAG4)
            comm.Send(error[offset:offset+chunksize], dest=MASTER, tag=TAG5)
            timeElapsed = time.time() - starttime
            print("Time elapsed for worker {}: {} seconds".format(taskid, timeElapsed))

    if taskid == MASTER:
        if seq_len <= numtasks:
            pass
        else:
            for source in range(1, numtasks):
                offset = comm.recv(source=source, tag=TAG1)
                chunksize = comm.recv(source=source, tag=TAG2)
                comm.Recv([length[offset:], chunksize, MPI.DOUBLE],
                          source=source, tag=TAG3)
                comm.Recv([fidelity[offset:], chunksize, MPI.DOUBLE],
                          source=source, tag=TAG4)
                comm.Recv([error[offset:], chunksize, MPI.DOUBLE],
                          source=source, tag=TAG5)

        timeElapsed = time.time() - starttime
        print("Time elapsed for Master: {} seconds".format(timeElapsed))

    return length, fidelity, seq_len, error


def fit_curve(x_data: np.ndarray, y_data: np.ndarray, seq_length: int,
              data_ss: int, error: np.ndarray, n: int, plot: bool):
    """
    Curve fitting using SciPy's non-linear least square analysis.

    Parameters
    ----------
    x_data : np.ndarray
        The x-data.
    y_data : np.ndarray
        The y-data.
    seq_length : int
        The max sequence length.
    data_ss : int
        The jump in plotted sequence length.
    error : np.ndarray
        DESCRIPTION.
    n : int
        The number that generates the size of the 4n Clifford and non-Clifford
    plot : bool, optional
        Plots the RB curve if True. The default is False.

    Returns
    -------
    None.

    """
    popt, pcov = curve_fit(decay, x_data, y_data, bounds=(0, [1., 1.]))

    if plot:
        plt.figure(figsize=(15, 7.5))
        t = np.arange(1, seq_length+data_ss, data_ss)
        fit_error = np.sqrt(np.diag(pcov))
        mean_res = np.average(np.abs(decay(t, *popt) - np.array(y_data)))
        best_fit_vals = tuple(np.vstack((popt, fit_error)).ravel('F'))
        fit_label = 'fit: A=%.10f$\\pm$%.10f\n   ' + '  $\\bar{F}$=%.10f$\\pm$%.10f\n'
        plt.plot(x_data, decay(t, *popt), 'r-',
                 label=fit_label % best_fit_vals)

        plt.errorbar(x_data, y_data, yerr=error, fmt="--x", ecolor='b',
                     elinewidth=1, linewidth=1, markersize=6, capsize=3,
                     label='mean residuals: %7.4f' % (mean_res))
        # plt.title("$n = 1000$, $K_s = 200$", fontsize=20)
        # plt.title("n = " + repr(n) , fontsize=20)
        plt.xlabel("Sequence length $s$", fontsize=20)
        plt.ylabel("Sequence fidelity $F_\\mathbf{G}$", fontsize=20)
        plt.xlim(0, seq_length+1)
        plt.ylim(0, 1)
        plt.legend(shadow=True, fontsize=20)
        plt.savefig("n = " + repr(n) + ".png".format(n))
        plt.show()

    return popt[1]


def variance_plot(exps: int, input_state: np.ndarray, seq_len: int,
                  samp_sizes: np.ndarray, data_ss: int, n: int, max_size: int,
                  input_state_2: np.ndarray, plot: bool):
    """
    Compare the estimates of many average gate fidelity from RB to the direct
    average gate fidelity calculations of many randomly generated noise maps,
    for increasing sequence samples.

    Parameters
    ----------
    exps : int
        The number of RB curves/estimates for a particular sequence sample.
    input_state : np.ndarray
        The first (up) input state in Pauli-Liouville representation.
    seq_len : int
        The max sequence length of the RB curve.
    samp_sizes : np.ndarray
        An array of different sample size for a particular sequence length.
    data_ss : int
        The jump in plotted sequence length.
    n : int
        The number that generates the size of the 4n Clifford and non-Clifford
        gate set.
    max_size : int
        The maximum size of the values that were first drawn from the Gaussian
        distribution in order to produce the sequence samples required.
    input_state_2 : np.ndarray
        The second (down) input state in Pauli-Lioville representation.
    plot : bool, optional
        Plots the RB curve if True. The default is False.
    """

    ave_fid_fit_plot = []
    variance = []
    for j in range(len(samp_sizes)):
        ave_fid_fit = []
        for i in range(1, exps+1):
            length, fidelity, seq_length, error = get_data2(input_state,
                                                            seq_len,
                                                            samp_sizes[j],
                                                            data_ss,
                                                            n,
                                                            max_size,
                                                            input_state_2)
            if taskid == MASTER:
                ideal_fid_fit = fit_curve(length, fidelity, seq_length,
                                          data_ss, error, n, plot)
                ave_fid_fit.append(ideal_fid_fit)

        if taskid == MASTER:
            ave_fid_fit_plot.append(np.average(ave_fid_fit))
            variance.append(np.var(ave_fid_fit))

    if taskid == MASTER:
        resources = seq_len * samp_sizes
        plt.figure(figsize=(15, 7.5))
        plt.errorbar(resources, ave_fid_fit_plot, yerr=np.sqrt(variance),
                     fmt="x", ecolor='g', elinewidth=2, capsize=5,
                     markersize=10, 
                     label = "RB Estimated $\\bar{F}(\\mathbf{\\mathcal{D}}_{p})$")
        noise = gen_channel(0.06, 0.06, 0.998, 0.04, 30000000, 1000000)
        diag = np.trace(noise, axis1=1, axis2=2) - 1
        d = m.sqrt(noise.shape[1])
        p = diag/(d**2-1)
        fid = np.average(((d-1)*p + 1)/d)
        avg_fid = fid*np.ones(len(resources))
        plt.plot(resources, avg_fid, 'b-', label="$\\bar{F}_{10^7}(\\mathbf{\\Lambda})$ = %.10f" % fid)
        print("Average Gate Fidelity: {}".format(ave_fid_fit_plot))

        plt.xlabel("Resource Number ($s \\times K_s$)", fontsize=20)
        plt.ylabel("Average Gate Fidelity $\\bar{F}$", fontsize=20)
        plt.legend(shadow=True, fontsize=20, loc='best')

        plt.savefig('variance.png')  # on for bluecrystal
        plt.show()  # off for bluecrystal

        plt.figure(figsize=(15, 7.5))
        plt.plot(resources, np.sqrt(variance), 'o', markersize=10)
        plt.xlabel("Resource number", fontsize=20)
        plt.ylabel("Standard deviation", fontsize=20)

        plt.savefig("std.png")  # on for bluecrystal
        plt.show()  # off for bl1uecrystal


starttime = time.time()
input_state = m.sqrt(1/2)*np.array([[1],[0],[0],[1]])
seq_len = 100
samp_size = 100
n = 3
max_size = 100000 # simulation speed scales with max_size, not samp_size
input_state_2 = m.sqrt(1/2)*np.array([[1],[0],[0],[-1]])
data_ss = 1
exps = 2

rb_pauli_liouville(input_state, seq_len, samp_size, n, max_size, input_state_2)
length, fid, seq_len, fid_error = get_data2(input_state, seq_len,
                                            samp_size, data_ss, n,
                                            max_size, input_state_2)

if taskid==MASTER:
    ideal_fid_fit = fit_curve(length, fid, seq_len, 1, fid_error, n, plot=True)

samp_sizes = np.array([100, 200, 300, 400])

variance_plot(exps, input_state, seq_len, samp_sizes, data_ss, n, max_size,
              input_state_2, plot=False)

timeElapsed = time.time() - starttime
print("\nTime elapsed: {} seconds".format(timeElapsed))
