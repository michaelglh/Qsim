import numpy as np
from matplotlib import pyplot as plt, patches
import imageio.v2 as imageio

from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.circuit.library import HamiltonianGate

def generate_sierpinski_triangle(points, ternaries, depth, unique_points):
    if depth == 0:
        for point, ternary in zip(points, ternaries):
            if point in unique_points:
                unique_points[point].append(ternary)
            else:
                unique_points[point] = [ternary]
    else:
        # Calculate midpoints and their ternary representations
        mid1 = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
        mid2 = ((points[1][0] + points[2][0]) / 2, (points[1][1] + points[2][1]) / 2)
        mid3 = ((points[0][0] + points[2][0]) / 2, (points[0][1] + points[2][1]) / 2)

        # Recurse for each sub-triangle
        generate_sierpinski_triangle([points[0], mid1, mid3], [ternaries[0]+'0', ternaries[0]+'1', ternaries[0]+'2'], depth - 1, unique_points)
        generate_sierpinski_triangle([mid1, points[1], mid2], [ternaries[1]+'0', ternaries[1]+'1', ternaries[1]+'2'], depth - 1, unique_points)
        generate_sierpinski_triangle([mid3, mid2, points[2]], [ternaries[2]+'0', ternaries[2]+'1', ternaries[2]+'2'], depth - 1, unique_points)

def ternary_differs_by_one(a, b):
    """Check if two ternary numbers differ by exactly one digit."""
    count_diff = 0
    for digit_a, digit_b in zip(a, b):
        if digit_a != digit_b:
            count_diff += 1
    return count_diff == 1

def is_adjacent_ternary(ternaries_a, ternaries_b):
    """Check if two points are adjacent."""
    for a in ternaries_a:
        for b in ternaries_b:
            if ternary_differs_by_one(a, b):
                return True
    return False

def is_adjacent_point(point_a, point_b, depth):
    """Check if two points are adjacent."""
    threshold = 1.1 / (2 ** depth)
    return np.linalg.norm(np.array(point_a) - np.array(point_b)) < threshold

# Initial triangle points and their ternary representations
initial_points = [(0, 0), (1, 0), (0.5, np.sqrt(0.75))]  # Vertices of the triangle
initial_ternaries = ['0', '1', '2']  # Ternary representations

# Generate points and ternaries
max_depth = 4  # Adjust the depth as needed
unique_points = {}
generate_sierpinski_triangle(initial_points, initial_ternaries, max_depth, unique_points)

# # Print unique points and their ternary representations
# for point, ternary in unique_points.items():
#     print(f"Point: {point}, Ternary: {ternary}")
    
# Using the unique_points dictionary from the previous step
point_list = list(unique_points.keys())
ternary_list = list(unique_points.values())
num_points = len(point_list)

# Initialize adjacency matrix
adjacency_matrix = np.zeros((num_points, num_points))

# Calculate the adjacency matrix
C = 0.5
for i in range(num_points):
    for j in range(i + 1, num_points):
        if is_adjacent_ternary(ternary_list[i], ternary_list[j]) and is_adjacent_point(point_list[i], point_list[j], max_depth):
            adjacency_matrix[i][j] = adjacency_matrix[j][i] = C

# Fill diagonal with beta
beta = 0.5
np.fill_diagonal(adjacency_matrix, beta)

# # Visualization
# def draw_connections(points, adjacency_matrix):
#     n = len(points)
#     for i in range(n):
#         for j in range(i + 1, n):
#             if adjacency_matrix[i, j] == 1:
#                 plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'b-', alpha=0.2)  # Line

# # Plotting the Sierpiński triangle and labeling the points with their ternary representations
# plt.figure(figsize=(10, 8))
# draw_connections(np.array(point_list), H)

# for point, ternaries in unique_points.items():
#     x, y = point
#     plt.scatter(x, y, color='blue')
#     # plt.text(x, y, ternaries, color='red', fontsize=8, ha='right', va='bottom')

# plt.gca().set_aspect('equal', adjustable='box')
# plt.title(f'Sierpiński Triangle (Depth {max_depth}) with Ternary Labels')
# plt.show()

# Calculate the number of qubits needed for the representation
npoint = adjacency_matrix.shape[0]
nqubit = np.ceil(np.log2(npoint))
nstate = int(2**nqubit)
print("Adjacency matrix:")
print(adjacency_matrix.shape)

diff_type = 'cls'  # 'qft' or 'cls'

def vizp(nodestate, points, depth, path=None):
    # calculate the possibility of each node
    pnode = nodestate / np.sum(nodestate)
    ampli_qubit = np.sqrt(pnode)

    radius = 1.0 / (2 ** depth) / 2
    npoint = len(points)

    fig, ax = plt.subplots(1, 1)
    # draw state vector
    for i in range(npoint):
        circleInt = patches.Circle((points[i][0], points[i][1]), ampli_qubit[i]*radius, color='r',alpha=1.0)
        ax.add_patch(circleInt)
    # draw connections
    for i in range(npoint):
        for j in range(i + 1, npoint):
            if adjacency_matrix[i, j] > 0:
                ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'b-', alpha=0.1)  # Line
    ax.set_aspect('equal')
    ax.axis('off')
    if path is not None:
        plt.savefig(path, dpi=100)
        plt.close()
    else:
        plt.show()

def viz2(statevector, points, depth, path):
    ampli_qubit = np.absolute(statevector)
    phase_qubit = np.angle(statevector)
    radius = 1.0 / (2 ** depth) / 2
    npoint = len(points)

    fig, ax = plt.subplots(1, 1)
    # draw state vector
    for i in range(npoint):
        circleExt = patches.Circle((points[i][0], points[i][1]), radius, color='gray',alpha=0.1)
        circleInt = patches.Circle((points[i][0], points[i][1]), ampli_qubit[i]*radius, color='b',alpha=0.3)
        ax.add_patch(circleExt)
        ax.add_patch(circleInt)
        xl = [points[i][0], points[i][0] + radius*ampli_qubit[i]*np.cos(phase_qubit[i] + np.pi/2)]
        yl = [points[i][1], points[i][1] + radius*ampli_qubit[i]*np.sin(phase_qubit[i] + np.pi/2)]
        ax.plot(xl,yl,'r')
    # draw connections
    for i in range(npoint):
        for j in range(i + 1, npoint):
            if adjacency_matrix[i, j] > 0:
                ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'b-', alpha=0.1)  # Line
    ax.set_aspect('equal')
    ax.axis('off')
    if path is not None:
        plt.savefig(path, dpi=100)
        plt.close()
    else:
        plt.show()

# Classical Diffusion
if diff_type == 'cls':
    p_stay = beta**2/(beta**2 + C**2)
    p_tran = C**2/(beta**2 + C**2)
    print("Stay probability:", p_stay)
    print("Transition probability:", p_tran)

    # Initialize particles
    nodestate = np.zeros(npoint, dtype=int)
    num_particles = 10000
    particles = np.zeros(num_particles, dtype=int)
    nodestate[0] = num_particles

    # Random walk function
    def random_walk(particles, adjacency_matrix, dt):
        r = np.random.rand(len(particles))
        move = r < (p_tran*dt)
        stay = ~move

        # get neighbors
        neighbors = adjacency_matrix[particles]
        new_particles = np.array([np.random.choice(np.nonzero(neighbors[i])[0]) if move[i] else particles[i] for i in range(len(particles))])

        # update nodestate
        np.add.at(nodestate, particles[stay], -1)
        np.add.at(nodestate, new_particles[move], 1)

        return new_particles

    # Run the simulation and generate a gif
    tranmat = adjacency_matrix
    np.fill_diagonal(tranmat, 0)
    T = 200
    dt = 1
    num_steps = int(T/dt)
    filenames = []
    for _ in range(num_steps):
        filename = './Figs/fractal_classical/%d.png' % _
        filenames.append(filename)
        vizp(nodestate, point_list, max_depth, filename)
        particles = random_walk(particles, tranmat, dt)
    filename = './Figs/fractal_classical/%d.png' % num_steps
    filenames.append(filename)
    vizp(nodestate, point_list, max_depth, filename)

    # Create a GIF
    with imageio.get_writer('./Figs/fractal_classical/classical.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

# Qunatum Diffusion
if diff_type == 'qft':
    # Construct the Hamiltonian
    H = np.zeros((nstate, nstate))
    H[np.ix_(np.arange(npoint), np.arange(npoint))] = adjacency_matrix
    assert np.allclose(H, np.conj(H.T))
    print("Hamiltonian H:")
    print(H.shape)

    # Time parameter for evolution
    filenames = []
    simulator = Aer.get_backend('statevector_simulator')
    for t in np.arange(0, 20.1, 0.1):
        # Qiskit simulation
        qc = QuantumCircuit(nqubit)
        U = HamiltonianGate(H, time=t)
        qc.append(U, qc.qubits)

        # Simulate the circuit
        job = execute(qc, simulator)
        result = job.result()

        filename = './Figs/fractal_quantum/%d.png' % int(t*10)
        filenames.append(filename)
        viz2(result.get_statevector(), point_list, max_depth, filename)

    # Create a GIF
    with imageio.get_writer('./Figs/fractal_quantum/quantum.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
