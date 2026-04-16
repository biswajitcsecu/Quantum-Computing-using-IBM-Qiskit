#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import quairkit as qkit
from quairkit import Circuit
from quairkit.database import *
from quairkit.operator import RZ, ChoiRepr, KrausRepr, StinespringRepr, Oracle
from quairkit.qinfo import *

import warnings
warnings.filterwarnings('ignore')


# In[4]:


num_qubits = 3  # number of qubits
cir = Circuit(num_qubits) 
cir.h(0)  # Add Hadamard gate to qubit 0
cir.x([0, 1])  # Add X gate to qubit 0 and 1
cir.cx([0, 1])  # Add CNOT gate to qubit 0 and 1
print("the diagram of a quantum circuit:")
cir.plot()
print(f"the total number of qubit of the circuit is {cir.num_qubits}")


# In[5]:


print(f"the unitary matrix of the circuit:\n{cir.matrix}")


# In[6]:


cir.swap([[1, 2], [0, 2]])  # Add SWAP gate to qubit 1 and 2, and qubit 0 and 2
cir.z()  # Add Z gate to all qubits
cir.cy()  # Add CY gate in cycle
print("the diagram of the quantum circuit with more gates:")
cir.plot()
print(f"the total number of qubit of the circuit is {cir.num_qubits}")


# In[7]:


cir.rx() # Add Rx gate to all qubits with random parameters
cir.rx([0, 1], param=np.arange(2)) # Add RX gate to the first two qubits with specified parameters

cir.u3([0]) # Add universal single qubit gate
cir.universal_two_qubits([0, 1]) # Add universal two qubits gate
print("the diagram of a quantum circuit with parametrized gates:")
cir.plot()


# In[8]:


qft_mat = qft_matrix(num_qubits=2)  # construct a 2-qubit quantum Fourier transformation (QFT) operator
qft_inverse = dagger(qft_mat)  # get the inverse QFT

cir.oracle(qft_inverse, qubits_idx=[0, 1], latex_name=r'\text{QFT}_{2}^{\dagger}')  # add custom gate
print("the diagram of a quantum circuit with a custom gate:")
cir.plot()


# In[9]:


qft_mat = qft_matrix(num_qubits=3)  # construct a 3-qubit quantum Fourier transformation (QFT) operator
qft_inverse = dagger(qft_mat)  # get the inverse QFT
custom_unitary = Oracle(qft_inverse, system_idx=list(range(3)), gate_info={"tex": r'QFT_{3}^{\dagger}'})
cir.append(custom_unitary)
print("the diagram of a quantum circuit with custom gates:")
cir.plot()


# In[10]:


cir.oracle(random_unitary(num_qubits=2), qubits_idx=[0, 1, 2], control_idx=0, gate_name="O")

print("the diagram of a quantum circuit with a controlled custom gate:")
cir.plot()


# In[11]:


cir.depolarizing(prob=0.1, qubits_idx=[0])  # add a depolarizing channel (with probability 0.1) to qubit 0
cir.bit_phase_flip(0.2)  # Add bit-phase flip channels (with probability 0.2) to all qubits
rho = random_state(num_qubits=3)
replacement_choi_repr = replacement_choi(rho)  # replacement channel in Choi representation
cir.choi_channel(replacement_choi_repr, system_idx=[0, 1, 2])  # add replacement channel in Choi representation

reset_kraus_repr = reset_kraus([0.1, 0.2])  # reset channel in Kraus representation
cir.kraus_channel(reset_kraus_repr, system_idx=2)  # add reset channel in Kraus representation

random_stin_repr = random_channel(num_systems=1, target="stinespring")  # random single-qubit channel in Stinespring representation
cir.stinespring_channel(random_stin_repr, system_idx=1)  # add random channel in Stinespring representation


# In[12]:


choi_op = random_channel(num_systems=2, target="choi")
choi_repr = ChoiRepr(choi_op, system_idx=[1, 2])
cir.append(choi_repr)

kraus_op = random_channel(num_systems=1, target="kraus")
kraus_repr = KrausRepr(kraus_op, system_idx=0)
cir.append(kraus_repr)

stine_op = random_channel(num_systems=3, target="stinespring")
stine_repr = StinespringRepr(stine_op, system_idx=[0, 1, 2])
cir.append(stine_repr)


# In[13]:


num_qubits = 3  # number of qubits
cir = Circuit(num_qubits)
cir.linear_entangled_layer(depth=1)  # add a linear entangled layer
cir.real_block_layer(depth=1)  # add real block layers in depth 2
cir.real_entangled_layer(depth=1)  # add a real entangled layer

cir.complex_block_layer(depth=1)  # add a complex block layer 
cir.complex_entangled_layer(depth=1)  # add a complex entangled layer

print("the diagram of a quantum circuit with different kinds of layers:")
cir.plot(style='compact')


# In[14]:


print("The unitary matrix of the first layer is\n", cir[0].matrix)


# In[15]:


cir = Circuit(num_qubits=3)
# Set parameters and customize quantum gates. Here we select Ry, Rx, Rz gates
param = np.random.rand(2)

# By default, randomly generates a set of parameters
rz_gate = RZ(param=param, qubits_idx=[1, 2])

# Add quantum gates
cir.ry([0, 2])
cir.rx([0, 1])
cir.insert(index=2, module=rz_gate)  # index where to insert

print("The quantum circuit after adding gates is: ")
cir.plot()

cir.pop(1)  # Remove Rx gate
print("The quantum circuit after removing gates is:")
cir.plot()


# In[16]:


output_state = cir()  # Run the circuit with initial state |0>
print("the output state for inputting zero state is:", output_state)

rho = random_state(num_qubits=3)
output_state = cir(rho)  # Run the circuit with initial state sigma
print("the output state for inputting state rho is:", output_state)


# In[17]:


print("the circuit depth is", cir.depth, "\n")

print("the gate history of the circuit is\n", cir.operator_history)


# In[18]:


print("the trainable parameters of entire circuit are", cir.param)

cir.update_param(torch.ones_like(cir.param))  # update the parameters of the circuit
print("the updated trainable parameters of entire circuit are", cir.param)


# In[19]:


qkit.print_info()


# In[ ]:




