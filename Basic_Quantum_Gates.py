#!/usr/bin/env python
# coding: utf-8

# In[92]:


from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import generate_preset_pass_manager 
import bluequbit
import numpy as np
import matplotlib.pylab as plt
import warnings
warnings.filterwarnings('ignore')
bq_client = bluequbit.init("EVWv3LhwF7bQSkRSHeMMwba6YJgB9Oi0")


# ## **Controlled-Hadamard**

# In[93]:


c=0
t=1
qucir=QuantumCircuit(2,2)
qucir.h(0)
qucir.cx(c,t)
qucir.measure([0,1], [0,1]) 
plt.figure(figsize=(10, 5))
qucir.draw('mpl')


# In[94]:


get_ipython().run_cell_magic('latex', '', 'CH=\\begin{pmatrix}\n1 & 0 & 0 & 0 \\\\\n0 & 1 & 0 & 0 \\\\\n0 & 0 & \\frac{1}{\\sqrt{2}} & \\frac{1}{\\sqrt{2}} \\\\ \n0 & 0 & \\frac{1}{\\sqrt{2}} & \\frac{-1}{\\sqrt{2}} \n\\end{pmatrix}\n')


# In[95]:


qucir.global_phase=np.pi/4
qucir.measure([0], [1])
print(qucir.data)
print('\n')
print(qucir.global_phase)


# In[96]:


qc = QuantumCircuit(2, 2)
qc.h(0)          
qc.cx(0, 1)        
qc.measure([0,1], [0,1])  

# Draw the circuit
qc.draw('mpl')


# In[98]:


job = bq_client.run(qc, job_name="test1", device="mps.cpu")
state_vector = job.get_counts()


# In[99]:


qr1 = QuantumRegister(2)
qr2 = QuantumRegister(1)
cr1 = ClassicalRegister(2)
cr2 = ClassicalRegister(1)
qcx = QuantumCircuit(qr1, qr2, cr1, cr2)
qcx.h(0)
qcx.h(1)
qcx.cx(0,1)
qcx.h(2)
qcx.cx(0,2)
qcx.measure_all()
# Draw the circuit
qcx.draw('mpl')


# In[100]:


print("List the qubits in this circuit:", qcx.qubits)
print('\n')
print("List the classical bits in this circuit:", qcx.clbits)


# ## **Layout**

# In[53]:


# Create circuit to test transpiler on
qc = QuantumCircuit(3, 3)
qc.h(0)
qc.cx(0, 1)
qc.swap(1, 2)
qc.cx(0, 1)

# Add measurements to the circuit
qc.measure([0, 1, 2], [0, 1, 2])

# Specify the QPU to target
backend = GenericBackendV2(3)

# Transpile the circuit
pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
transpiled = pass_manager.run(qc)

# Print the layout after transpilation
print(transpiled.layout.routing_permutation())
qc.draw('mpl')


# In[52]:


transpiled.draw('mpl')


# In[54]:


print("List the qubits in this circuit:", qc.qubits)
print('\n')
print("List the classical bits in this circuit:", qc.clbits)


# In[55]:


print("List the qubits in this circuit:", transpiled.qubits)
print('\n')
print("List the classical bits in this circuit:", transpiled.clbits)


# In[56]:


qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Print the original circuit
print("Original circuit:")
qc.draw('mpl')


# In[63]:


# Transpile the circuit with a specific basis gates 
backend = GenericBackendV2(2, basis_gates=['u1', 'u2', 'u3', 'cx'])
pm = generate_preset_pass_manager( optimization_level=1, backend=backend, scheduling_method="alap")
transpiled_qc = pm.run(qc)
transpiled_qc.draw('mpl')


# In[64]:


print("Transpiled circuit with basis gates ['u1', 'u2', 'u3', 'cx']:")
print("Start times of instructions in the transpiled circuit:")
for instruction, start_time in zip(transpiled_qc.data, transpiled_qc.op_start_times):
    print(f"{instruction.operation.name}: {start_time}")


# ## **Parameterized Circuit**

# In[72]:


circuit = QuantumCircuit(2)
params = [Parameter('A'), Parameter('B'), Parameter('C')]
circuit.ry(params[0], 0)
circuit.crx(params[1], 0, 1)
circuit.draw('mpl')


# In[73]:


circuit.assign_parameters({params[0]: params[2]}, inplace=True)
circuit.draw('mpl')


# In[78]:


circuit = QuantumCircuit(2)
params = ParameterVector('P', 2)
circuit.ry(params[0], 0)
circuit.crx(params[1], 0, 1)
circuit.draw('mpl')


# In[79]:


bound_circuit = circuit.assign_parameters([1, 2])
bound_circuit.draw('mpl')


# In[ ]:




