#!/usr/bin/env python
# coding: utf-8

# In[27]:


import sys
import numpy as np
from math import pi
import qiskit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.visualization import plot_histogram,plot_bloch_multivector, plot_histogram, array_to_latex
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import RYGate, MCXGate
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import Statevector
from qiskit_aer import Aer, AerSimulator
from qiskit_quantuminspire.qi_provider import QIProvider
from qiskit_aer.primitives import SamplerV2 as Sampler
import cudaq
import qutip
import matplotlib.pylab as plt
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer.primitives import SamplerV2
import warnings
warnings.filterwarnings('ignore')


from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from matplotlib import pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_ibm_runtime import SamplerV2 as Sampler2
from qiskit.visualization import plot_distribution    


# ## **Quantum Register and Classical Register**

# In[28]:


# set up Quantum Register and Classical Register for 3 qubits
q = QuantumRegister(3)
c = ClassicalRegister(3)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)
qc.h(q)
qc.measure(q, c)

qc.draw('mpl')


# In[29]:


circ = qiskit.QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure_all()
circ.draw('mpl')


# In[131]:


service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=12)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
qcirc = pm.run(circ)

sampler = Sampler(backend)
job = sampler.run([qcirc])

# Perform an ideal simulation
result_ideal = job.result()
counts_ideal = result_ideal[0].data.meas.get_counts()
print('Counts(ideal):', counts_ideal)


# In[31]:


sim = AerSimulator()

psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
theta3 = [1, 2, 3, 4, 5, 6]

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")



# In[32]:


# --------------------------
# Simulating using sampler
# --------------------------
from qiskit_aer.primitives import SamplerV2
from qiskit import QuantumCircuit

# create a Bell circuit
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")



# In[33]:


@cudaq.kernel
def kernel():
    A = cudaq.qubit()
    B = cudaq.qvector(3)
    C = cudaq.qvector(5)


# In[34]:


N = 2
@cudaq.kernel
def kernel(N: int):
    register = cudaq.qvector(N)


# Passing complex vectors as parameters
c = [.707 + 0j, 0 - .707j]

@cudaq.kernel
def kernel(vec: list[complex]):
    q = cudaq.qvector(vec)


# In[35]:


c = np.array([0.70710678 + 0j, 0., 0., 0.70710678], dtype=cudaq.complex())

@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)


# In[36]:


# Define as CUDA-Q amplitudes
c = cudaq.amplitudes([0.70710678 + 0j, 0., 0., 0.70710678])
@cudaq.kernel
def kernel():
    q = cudaq.qvector(c)



# In[ ]:





# In[37]:


## Retry the subsequent cells by setting the target to density matrix simulator.
cudaq.set_target("density-matrix-cpu")
get_ipython().run_line_magic('matplotlib', 'inline')

@cudaq.kernel
def kernel(angles: np.ndarray):
    qubit = cudaq.qubit()
    rz(angles[0], qubit)
    rx(angles[1], qubit)
    rz(angles[2], qubit)


# In[38]:


rng = np.random.default_rng(seed=11)
blochSphereList = []
for _ in range(4):
    angleList = rng.random(3) * 2 * np.pi
    sph = cudaq.add_to_bloch_sphere(cudaq.get_state(kernel, angleList))
    blochSphereList.append(sph)


# In[39]:


cudaq.show(blochSphereList[0])    
 


# In[40]:


cudaq.show(blochSphereList[:2], nrows=1, ncols=2)


# In[41]:


cudaq.show(blochSphereList[:2], nrows=2, ncols=1)


# In[42]:


cudaq.show(blochSphereList[:], nrows=2, ncols=2)


# In[43]:


rng = np.random.default_rng(seed=47)
blochSphere = qutip.Bloch()
for _ in range(10):
    angleList = rng.random(3) * 2 * np.pi
    sph = cudaq.add_to_bloch_sphere(cudaq.get_state(kernel, angleList), blochSphere)
blochSphere.show()    


# In[44]:


blochSphere.show()


# In[45]:


@cudaq.kernel
def kernel_to_draw():
    q = cudaq.qvector(4)
    h(q)
    x.ctrl(q[0], q[1])
    y.ctrl([q[0], q[1]], q[2])
    z(q[2])

    swap(q[0], q[1])
    swap(q[0], q[3])
    swap(q[1], q[2])

    r1(3.14159, q[0])
    tdg(q[1])
    s(q[2])


# In[46]:


#print(cudaq.draw('latex', kernel_to_draw))


# In[47]:


qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
qc.cx(0, 3)

qc.barrier()

qc.t(0)
qc.t(1)
qc.t(2)
qc.t(3)

qc.barrier()

qc.cx(0, 3)
qc.cx(0, 2)
qc.cx(0, 1)

qc.measure_all()
print(qc)
qc.draw()


# In[48]:


qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0,1)
qc.draw()


# In[49]:


qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0,1)
display(qc.draw())  # `display` is a command for Jupyter notebooks
                    # similar to `print`, but for rich content

# Let's see the result
svsim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = transpile(qc, svsim)
final_state = svsim.run(qobj).result().get_statevector()
display(array_to_latex(final_state, prefix="\\text{Statevector} = "))
plot_bloch_multivector(final_state)


# In[ ]:


bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=12)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
qbell = pm.run(bell)

sampler = Sampler(backend)
job = sampler.run([qbell])



print(f"Job ID is {job.job_id()}")
pub_result = job.result()[0]
print(f"Counts for the meas output register: {pub_result.data.meas.get_counts()}")
ncounts = pub_result.data.meas.get_counts()

histogram = plot_histogram(ncounts,title="IBM Quantum Simulator")


# In[51]:


sim = AerSimulator()


psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
theta3 = [1, 2, 3, 4, 5, 6]

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")

# create a Bell circuit
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")



# In[52]:


# 1. Create a Quantum Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()


# In[53]:


# 2. Initialize the AerSimulator
simulator = AerSimulator()

# 3. Transpile the circuit for the simulator
compiled_circuit = transpile(qc, simulator)


# In[54]:


# 4. Run the circuit and get results
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()

print("\nCounts:", counts)


# In[55]:


histogram = plot_histogram(counts,title="IBM Quantum Simulator")


# In[ ]:





# In[56]:


circuit = QuantumCircuit(2)
circuit.cx(0, 1)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.measure_all()
circuit.draw()


# In[57]:


initialization_circuit = QuantumCircuit(2)
initialization_circuit.id(0)
initialization_circuit.x(1)
initialization_circuit.barrier()


# In[58]:


new_circuit = initialization_circuit.compose(circuit)
new_circuit.measure_all()
new_circuit.draw()


# In[59]:


simulator = AerSimulator()
compiled_new_circuit = transpile(new_circuit, simulator)
newjob = simulator.run(compiled_new_circuit, shots=1000)
nresult = newjob.result()
ncounts = nresult.get_counts()

print("\nCounts:", ncounts)

histogram = plot_histogram(ncounts,title="IBM Quantum Simulator")


# In[60]:


provider = QIProvider()
print(provider.backends())
backend = provider.get_backend("QX emulator")


# In[61]:


circuit = QuantumCircuit(3, 10)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)


# In[62]:


backend = provider.get_backend(name="QX emulator")
job = backend.run(circuit, shots=1024)
result = job.result()
print(result)


# In[63]:


counts = result.get_counts()
print("\nCounts:", counts)


# In[64]:


histogram = plot_histogram(counts,title="IBM Quantum Simulator")


# In[65]:


x = QuantumRegister(1, 'x')
y = QuantumRegister(1, 'y')
z = QuantumRegister(1, 'z')
v = QuantumRegister(1, 'v')
aux = QuantumRegister(2,'aux')
r = QuantumRegister(1, 'result')
qc = QuantumCircuit(x,y,z,v,aux,r)

qc.x(x)
qc.x(y)
qc.x(z)
qc.x(v)
qc.barrier()
qc.ccx(x,y,aux[0])
qc.ccx(aux[0],z,aux[1])
qc.ccx(aux[1],z,r)
qc.barrier()
qc.ccx(x,y,aux[0])
qc.ccx(aux[0],z,aux[1])
qc.ccx(aux[1],z,r)
#un-computing of the aux registers
qc.ccx(aux[0],z,aux[1])
qc.ccx(x,y,aux[0])

qc.measure_all()
qc.draw()

# Let's see the result
svsim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = transpile(qc, svsim) 
final_state = svsim.run(qobj).result().get_statevector()
display(array_to_latex(final_state, prefix="\\text{Statevector} = "))
plot_bloch_multivector(final_state)
counts = result.get_counts()


print("\nTotal count are:",counts)
plot_histogram(counts)



# ## **Simulating Qiskit circuit with Aer**

# In[66]:


sim = AerSimulator()
psi1 = transpile(RealAmplitudes(num_qubits=2, reps=2), sim, optimization_level=0)
psi2 = transpile(RealAmplitudes(num_qubits=2, reps=3), sim, optimization_level=0)

H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
H2 = SparsePauliOp.from_list([("IZ", 1)])
H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
theta3 = [1, 2, 3, 4, 5, 6]

estimator = EstimatorV2()

# calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
#              <psi1(theta3)|H3|psi1(theta3)>],
#             [<psi2(theta2)|H2|psi2(theta2)>] ]
job = estimator.run(
    [
        (psi1, [H1, H3], [theta1, theta3]),
        (psi2, H2, theta2)
    ],
    precision=0.01
)
result = job.result()
print(f"expectation values : psi1 = {result[0].data.evs}, psi2 = {result[1].data.evs}")


# In[67]:


# create a Bell circuit
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()
bell.draw()
# create two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc = transpile(pqc, sim, optimization_level=0)
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
pqc2 = transpile(pqc2, sim, optimization_level=0)

theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

# initialization of the sampler
sampler = SamplerV2()

# collect 128 shots from the Bell circuit
job = sampler.run([bell], shots=128)
job_result = job.result()
print(f"counts for Bell circuit : {job_result[0].data.meas.get_counts()}")
 
# run a sampler job on the parameterized circuits
job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
job_result = job2.result()
print(f"counts for parameterized circuit : {job_result[0].data.meas.get_counts()}")


# ## **Quantum oracle**

# In[68]:


qc = QuantumCircuit(3)
#Difusor
qc.h([0,1,2])
qc.x([0,1,2])
qc.h(0)
qc.ccx(1,2,0)
qc.h(0)
qc.x([0,1,2])
qc.h([0,1,2])

#Multiply with (-1)
qc.rz(2*pi,0)
qc.rz(2*pi,1)
qc.rz(2*pi,2)

simulator = AerSimulator()
qc.save_unitary()
qobj = transpile(qc, simulator)
unitary = simulator.run(qobj).result().get_unitary()
print("\nSize of the unitary matrix:",np.asarray(unitary).shape)
array_to_latex(unitary, prefix="\\text{G3 = }\n")


# In[69]:


qc.measure_all()
qc.draw()


# ## **Grover’s amplification**

# In[70]:


qc = QuantumCircuit(4)
qc.h([0,1,2])
qc.x(3)
qc.h(3)
qc.barrier()
qc.x(0)
qc.x(2)
gate = MCXGate(3)
qc.append(gate, [0, 1, 2, 3])
qc.x(0)
qc.x(2)
qc.barrier()
qc.h(3)
qc.draw()


# In[71]:


simulator = Aer.get_backend('statevector_simulator')
final_state = simulator.run(qc).result().get_statevector()


# In[72]:


array_to_latex(final_state,max_size=16,prefix="\\text{Statevector} = ")


# In[73]:


plot_bloch_multivector(final_state)


# In[74]:


qc = QuantumCircuit(4)
qc.h([0,1,2])
#Preparation of Aux
qc.x(3)
qc.h(3)
#Oracle
qc.barrier()
qc.x(0)
qc.x(2)
gate = MCXGate(3)
qc.append(gate, [0, 1, 2, 3])
qc.x(0)
qc.x(2)

#Diffusor
qc.barrier()
qc.h(3)
qc.barrier()
qc.h([0,1,2])
qc.x([0,1,2])
qc.h(0)
qc.ccx(1,2,0)
qc.h(0)
qc.barrier()
qc.x([0,1,2])
qc.h([0,1,2])
#Corrrect the sign, not required...
qc.rz(2*pi,0)
qc.rz(2*pi,1)
qc.rz(2*pi,2)
qc.draw(fold=140)


# In[75]:


simulator = Aer.get_backend('statevector_simulator')
final_state = simulator.run(qc).result().get_statevector()
array_to_latex(final_state,max_size=16,prefix="\\text{Statevector} = ")


# In[76]:


plot_bloch_multivector(final_state)


# In[77]:


qc = QuantumCircuit(4,3)
qc.h([0,1,2])
#Preparation of Aux
qc.x(3)
qc.h(3)
#Oracle
qc.barrier()
qc.x(0)
qc.x(2)
gate = MCXGate(3)
qc.append(gate, [0, 1, 2, 3])
qc.x(0)
qc.x(2)
#Diffusor
qc.barrier()
qc.h(3)
qc.barrier()
qc.h([0,1,2])
qc.x([0,1,2])
qc.h(0)
qc.ccx(1,2,0)
qc.h(0)
qc.x([0,1,2])
qc.h([0,1,2])
qc.barrier()
qc.h(3)
#Oracle
qc.barrier()
qc.x(0)
qc.x(2)
gate = MCXGate(3)
qc.append(gate, [0, 1, 2, 3])
qc.x(0)
qc.x(2)
qc.barrier()
qc.h(3)
#Diffusor
qc.barrier()
qc.h([0,1,2])
qc.x([0,1,2])
qc.h(0)
qc.ccx(1,2,0)
qc.h(0)
qc.x([0,1,2])
qc.h([0,1,2])
qc.barrier()
qc.measure(0,0)
qc.measure(1,1)

qc.measure(2,2)
qc.draw(fold=100)


# In[78]:


simulator = Aer.get_backend('statevector_simulator')
final_state = simulator.run(qc).result().get_statevector()
array_to_latex(final_state,max_size=16,prefix="\\text{Statevector} = ")


# In[79]:


plot_bloch_multivector(final_state)


# In[ ]:





# In[80]:


qc = QuantumCircuit(16)
#Initial state b,a,c,d,c
#First Position 01 is b 01
qc.x(0)
#Second Position 23 is a 00
#Third Position 45 is c 10
qc.x(5)
#Fourth Position 67 is d 11
qc.x(6)
qc.x(7)
#Fifth Position 89 is c 10
qc.x(9)
#Path Descriptor
qc.h(11)
qc.h(12)
qc.barrier()

#First Rule
#Set flag 10 dependent on the path descriptor
qc.ccx(11,12,10)
# Move
qc.cswap(10,0,2)
qc.cswap(10,1,3)
#Reset flag
qc.ccx(11,12,10)
#Second Rule
#Set flag 10 dependent on the path descriptor
qc.x(11)
qc.ccx(11,12,10)
# Move
qc.cswap(10,2,4)
qc.cswap(10,3,5)
#Reset flag
qc.ccx(11,12,10)
qc.x(11)
#Third Rule
#Set flag 10 dependent on the path descriptor
qc.x(12)
qc.ccx(11,12,10)
# Move
qc.cswap(10,4,6)
qc.cswap(10,5,7)
#Reset flag
qc.ccx(11,12,10)
qc.x(12)
#Fourth Rule
#Set flag 10 dependent on the path descriptor
qc.x(11)
qc.x(12)
qc.ccx(11,12,10)
# Move
qc.cswap(10,6,8)
qc.cswap(10,7,9)
#Reset flag
qc.ccx(11,12,10)
qc.x(12)
qc.x(11)
qc.barrier()
#Depth two: 2th Path Descriptor
qc.h(13)
qc.h(14)
qc.barrier()
#First Rule
#Set flag 10 dependent on the path descriptor
qc.ccx(13,14,10)       
# Move
qc.cswap(10,0,2)
qc.cswap(10,1,3)
#Reset flag
qc.ccx(13,14,10)
#Second Rule
#Set flag 10 dependent on the path descriptor
qc.x(13)
qc.ccx(13,14,10)
# Move
qc.cswap(10,2,4)
qc.cswap(10,3,5)
#Reset flag
qc.ccx(13,14,10)
qc.x(13)
#Third Rule
#Set flag 10 dependent on the path descriptor
qc.x(14)
qc.ccx(13,14,10)    
# Move
qc.cswap(10,4,6)
qc.cswap(10,5,7)
#Reset flag
qc.ccx(13,14,10)
qc.x(14)
#Fourth Rule
#Set flag 10 dependent on the path descriptor
qc.x(13)
qc.x(14)
qc.ccx(13,14,10)
# Move
qc.cswap(10,6,8)
qc.cswap(10,7,9)
#Reset flag
qc.ccx(13,14,10)
qc.x(14)
qc.x(13)
qc.barrier()
#Oracle
gate = MCXGate(5)
#Mark the goal state
#Initial state a,b,c,c,d
qc.append(gate, [2,5,7,8,9,15])


# In[81]:


def rules(): 
    qc = QuantumCircuit(15)
    #First Rule
    #Set flag 10 dependent on the path descriptor
    qc.ccx(11,12,10)
    # Move
    qc.cswap(10,0,2)
    qc.cswap(10,1,3)
    #Reset flag
    qc.ccx(11,12,10)
    #Second Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(11)
    qc.ccx(11,12,10)
    # Move
    qc.cswap(10,2,4)
    qc.cswap(10,3,5)
    #Reset flag
    qc.ccx(11,12,10)
    qc.x(11)
    #Third Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(12)
    qc.ccx(11,12,10)
    # Move
    qc.cswap(10,4,6)
    qc.cswap(10,5,7)
    #Reset flag
    qc.ccx(11,12,10)
    qc.x(12)
    #Fourth Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(11)
    qc.x(12)
    qc.ccx(11,12,10)
    qc.cswap(10,6,8)
    qc.cswap(10,7,9)
    #Reset flag
    qc.ccx(11,12,10)
    qc.x(12)
    qc.x(11)
    #depth two
    #First Rule
    #Set flag 10 dependent on the path descriptor
    qc.ccx(13,14,10)
    # Move
    qc.cswap(10,0,2)
    qc.cswap(10,1,3)
    #Reset flag
    qc.ccx(13,14,10)
    #Second Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(13)
    qc.ccx(13,14,10)
    # Move
    qc.cswap(10,2,4)
    qc.cswap(10,3,5)
    #Reset flag
    qc.ccx(13,14,10)
    qc.x(13)
    #Third Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(14)
    qc.ccx(13,14,10) 
    # Move 
    qc.cswap(10,4,6)
    qc.cswap(10,5,7)
    #Reset flag
    qc.ccx(13,14,10)
    qc.x(14)
    #Fourth Rule
    #Set flag 10 dependent on the path descriptor
    qc.x(13)
    qc.x(14)
    qc.ccx(13,14,10)
    # Move
    qc.cswap(10,6,8)
    qc.cswap(10,7,9)
    #Reset flag
    qc.ccx(13,14,10)
    qc.x(14)
    qc.x(13)
    qc.name="RULES"
    
    return qc


# In[82]:


def rules_inv(): 
    qc=rules()
    qc_inv=qc.inverse()
    qc_inv.name="RULES_INV"
    
    return qc_inv


# In[83]:


def Grover():
    qc = QuantumCircuit(15)
    #Diffusor 11, 12, 13, 14
    qc.h([11,12,13,14])
    qc.x([11,12,13,14])
    qc.h(11)
    gate = MCXGate(3)
    qc.append(gate, [12,13,14,11])
    qc.h(11)
    qc.x([11,12,13,14])
    qc.h([11,12,13,14])
    qc.name="G"
    
    return qc


# In[84]:


qc = QuantumCircuit(16,4)
qc.x(0)
qc.x(5)
qc.x(6)
qc.x(7)
qc.x(9)
#Path Descriptor
qc.h(11)
qc.h(12)
qc.h(13)
qc.h(14)
#Preparation of Aux
qc.x(15)
qc.h(15)
qc.append(rules(),range(15))
#Oracle
gate = MCXGate(5)
#Mark the goal state
#Initial state a,b,c,c,d
qc.append(gate, [2,5,7,8,9,15])
qc.append(rules_inv(),range(15))
qc.barrier()
qc.h(15)
qc.barrier()
qc.append(Grover(),range(15))
qc.barrier()
qc.h(15)
qc.append(rules(),range(15))
#Oracle
gate = MCXGate(5)
#Mark the goal state
#Initial state a,b,c,c,d
qc.append(gate, [2,5,7,8,9,15])
qc.append(rules_inv(),range(15))
qc.barrier()
qc.h(15)
qc.barrier()
qc.append(Grover(),range(15))
qc.measure(11,0)
qc.measure(12,1)
qc.measure(13,2)
qc.measure(14,3)
qc.draw(fold=500)


# In[85]:


simulator = Aer.get_backend('qasm_simulator')
qobj = transpile(qc, simulator)
counts = simulator.run(qobj).result().get_counts()
print("\nTotal count are:",counts)
plot_histogram(counts)


# ## **ENTANGLEMENT OF BINARY PATTERNS**

# In[86]:


qc = QuantumCircuit(5)
#0-2 data
#Index
#3-4
qc.h(3)
qc.h(4)
#First patern
qc.ccx(3,4,0)
qc.ccx(3,4,2)
qc.barrier()
#Second patern
qc.x(3)
qc.ccx(3,4,0)
qc.ccx(3,4,1)
qc.x(3)
qc.barrier()
#Third patern
qc.x(4)
qc.ccx(3,4,0)
qc.ccx(3,4,1)
qc.ccx(3,4,2)
qc.x(4)
qc.barrier()
#Fourth patern
qc.x(3)
qc.x(4)
qc.ccx(3,4,1)
qc.x(4)
qc.x(3)

qc.draw()


# In[87]:


dag = circuit_to_dag(qc)
dag_drawer(dag)


# In[88]:


simulator = Aer.get_backend('statevector_simulator')
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nCounts:", counts)
plot_histogram(counts)


# ## **Bloch Shapre**

# In[89]:


simulator = Aer.get_backend('statevector_simulator')
final_state = simulator.run(qc).result().get_statevector()
array_to_latex(final_state,max_size=16,prefix="\\text{Statevector} = ")


# In[90]:


plot_bloch_multivector(final_state)


# In[91]:


qc = QuantumCircuit(7,1)
qc.h(6)
qc.barrier()
#ang = Sqrt[0.3]
#ArcCos[ang]*2
qc.ry(1.98231,0)
qc.ry(1.98231,1)
qc.ry(1.98231,2)
qc.h(3)
qc.h(4)
qc.h(5)
qc.barrier()
qc.cswap(6,0,3)
qc.cswap(6,1,4)
qc.cswap(6,2,5)
qc.h(6)
qc.measure(6,0)

qc.draw()


# In[92]:


simulator = Aer.get_backend('qasm_simulator')
job = simulator.run(qc, shots=1000)
result = job.result()
counts = result.get_counts()
print("\nCounts:", counts)
plot_histogram(counts)


# ## **DAG diagram**

# In[93]:


# Convert the circuit into a DAG
dag = circuit_to_dag(qc)
dag_drawer(dag)


# In[94]:


QiskitRuntimeService.save_account(channel="ibm_cloud",
token="IhVEYntFC-3medPLKnw1AgAYCtgoMBB7rXjHdjkQPhSg",set_as_default=True, overwrite=True)

#Run every time you need the service
service = QiskitRuntimeService()


# In[95]:


# Create a new circuit with two qubits
qc = QuantumCircuit(2)

# Add a Hadamard gate to qubit 0
qc.h(0)

# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)

# Return a drawing of the circuit using MatPlotLib ("mpl").
# These guides are written by using Jupyter notebooks, which
# display the output of the last line of each cell.
# If you're running this in a script, use `print(qc.draw())` to
# print a text drawing.
qc.draw("mpl")


# In[96]:


# Set up six different observables.

observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
observables = [SparsePauliOp(label) for label in observables_labels]


# In[97]:


service = QiskitRuntimeService()

backend = service.least_busy(simulator=False, operational=True)

# Convert to an ISA circuit and layout-mapped observables.
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

isa_circuit.draw("mpl", idle_wires=False)


# In[98]:


# Construct the Estimator instance.

estimator = Estimator(mode=backend)
estimator.options.resilience_level = 1
estimator.options.default_shots = 5000

mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

# One pub, with one circuit to run against five different observables.
job = estimator.run([(isa_circuit, mapped_observables)])

# Use the job ID to retrieve your job data later
print(f">>> Job ID: {job.job_id()}")


# In[99]:


# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).
job_result = job.result()

# This is the result from our single pub, which had six observables,
# so contains information on all six.
pub_result = job.result()[0]


# In[100]:


# Use the following code instead if you want to run on a simulator:

from qiskit_ibm_runtime.fake_provider import FakeBelemV2
backend = FakeBelemV2()
estimator = Estimator(backend)

# Convert to an ISA circuit and layout-mapped observables.

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

job = estimator.run([(isa_circuit, mapped_observables)])
result = job.result()

# This is the result of the entire submission.  You submitted one Pub,
# so this contains one inner result (and some metadata of its own).

job_result = job.result()

# This is the result from our single pub, which had five observables,
# so contains information on all five.

pub_result = job.result()[0]


# In[101]:


# Plot the result

values = pub_result.data.evs

errors = pub_result.data.stds

# plotting graph
plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Values")
plt.show()


# In[102]:


def get_qc_for_n_qubit_GHZ_state(n: int) -> QuantumCircuit:
    if isinstance(n, int) and n >= 2:
        qc = QuantumCircuit(n)
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
    else:
        raise Exception("n is not a valid input")
    return qc


# Create a new circuit with a hundred qubits in the GHZ state
n = 100
qc = get_qc_for_n_qubit_GHZ_state(n)


# In[103]:


# ZZII...II, ZIZI...II, ... , ZIII...IZ
operator_strings = [
    "Z" + "I" * i + "Z" + "I" * (n - 2 - i) for i in range(n - 1)
]
print(operator_strings)
print(len(operator_strings))

operators = [SparsePauliOp(operator) for operator in operator_strings]


# In[104]:


service = QiskitRuntimeService()

backend = service.least_busy(simulator=False, operational=True, min_num_qubits=100)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

isa_circuit = pm.run(qc)
isa_operators_list = [op.apply_layout(isa_circuit.layout) for op in operators]


# In[105]:


options = EstimatorOptions()
options.resilience_level = 1
options.dynamical_decoupling.enable = True
options.dynamical_decoupling.sequence_type = "XY4"

# Create an Estimator object
estimator = Estimator(backend, options=options)


# In[106]:


# Submit the circuit to Estimator
job = estimator.run([(isa_circuit, isa_operators_list)])
job_id = job.job_id()
print(job_id)


# In[107]:


# data
data = list(range(1, len(operators) + 1))  # Distance between the Z operators
result = job.result()[0]
values = result.data.evs  # Expectation value at each Z operator.
values = [
    v / values[0] for v in values
]  # Normalize the expectation values to evaluate how they decay with distance.

# plotting graph
plt.plot(data, values, marker="o", label="100-qubit GHZ state")
plt.xlabel("Distance between qubits $i$")
plt.ylabel(r"$\langle Z_i Z_0 \rangle / \langle Z_1 Z_0 \rangle $")
plt.legend()
plt.show()


# In[108]:


# Set the state we wish to search 
N = '110'
num_qubits = len(N)

# Create the quantum circuit 
qc = QuantumCircuit(num_qubits)

# Set all qubits in superposition
qc.h(range(num_qubits))
qc.barrier()

#Draw the circuit
qc.draw(output='mpl')


# In[109]:


# Reverse the state so it’s in proper qubit ordering
N = N[::-1]

# Encode N into our circuit
for idx in range(num_qubits):
    if N[idx] == '0':
        qc.x(idx)
qc.barrier()

# Draw the circuit
qc.draw(output='mpl')


# In[110]:


# Create the Grover oracle for our 3-qubit quantum circuit
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.barrier()

# Draw the circuit
qc.draw(output='mpl')


# In[111]:


# Reset the value after the oracle
for idx in range(num_qubits):
    if N[idx] == '0':
        qc.x(idx)
qc.barrier()

# Draw the circuit
qc.draw(output='mpl')


# In[112]:


# Set all qubits in superposition
qc.h(range(num_qubits))
qc.x(range(num_qubits))
qc.barrier()

# Draw the circuit
qc.draw(output='mpl')


# In[113]:


# Apply another oracle, same as the previous, 
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.barrier()

# Draw the circuit
qc.draw(output='mpl')


# In[114]:


# Reapply the X rotations on all qubits
qc.x(range(num_qubits))

qc.barrier()

# Reapply Hadamard gates to all qubits
qc.h(range(num_qubits))

# Draw the circuit
qc.draw(output='mpl')


# In[115]:


# Add measurement operators
qc.measure_all()

# Draw the circuit
qc.draw(output='mpl')


# In[116]:


def simulate_on_sampler(qc, backend, options):
    
    sampler = StatevectorSampler()
    # Transpile circuit
    pm = generate_preset_pass_manager(optimization_level=1)
    transpiled_qc = pm.run(qc)

    # Run using sampler
    state_vector_result = sampler.run([qc])
    
    result = state_vector_result.result()
    
    return transpiled_qc, result, state_vector_result


# In[117]:


def run_on_sampler(circuit, shots):
    fake_manila = FakeManilaV2()
    pass_manager = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
    transpiled_qc = pass_manager.run(circuit)
     
    # You can use a fixed seed to get fixed results.
    options = {"simulator": {"seed_simulator": 10258}}
    sampler = Sampler(mode=fake_manila, options=options)
     
    result = sampler.run([transpiled_qc], shots=shots).result()[0]
    return result


# In[118]:


result = run_on_sampler(qc, shots=4000)
counts = result.data.meas.get_counts()

# Print and plot results
print(counts)
plot_distribution(counts)


# In[119]:


# Run the circuit on the least busy quantum computer
backend = service.least_busy(min_num_qubits = num_qubits, simulator=False, operational=True)
print("Set backend: ", backend)


# In[120]:


def run_on_qc(circuit, shots):
    backend = service.least_busy(min_num_qubits=circuit.num_qubits, simulator=False, operational=True)
    #Print the least busy device
    print('The least busy device: {}'.format(backend))
    #result = {}

    # Transpile the circuit using the preset pass manager
    transpiler = generate_preset_pass_manager(backend=backend, optimization_level=3)
    transpiled_qc = transpiler.run(circuit)

    sampler = Sampler2(backend)
    job = sampler.run([transpiled_qc], shots=shots)
    job_result = job.result()

    # Extract the results 
    result = job_result[0]
        
    return result 


# In[121]:


# Run the circuit on the backend
shots = 1000
results = run_on_qc(qc, shots)
print(results)


# In[122]:


from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
 
# Bell Circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

shots = 1000
 
# Run the sampler job locally using FakeManilaV2
fake_manila = FakeManilaV2()
pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
isa_qc = pm.run(qc)
 
# You can use a fixed seed to get fixed results.
options = {"simulator": {"seed_simulator": 42}}
sampler = Sampler(mode=fake_manila, options=options)
 
result = sampler.run([isa_qc], shots=shots).result()
print(result[0].data.meas.get_counts())


# In[123]:


# Get the state vector simulator to view our final QFT state
#from qiskit.quantum_info import Statevector
#statevector = Statevector(isa_qc)
#plot_bloch_multivector(statevector)


# In[124]:


get_ipython().system('pip list | grep qiskit')


# In[ ]:




