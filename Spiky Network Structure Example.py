import numpy as np
from brian2 import *

# Model parametreleri
num_inputs = 2
input_rate = 10 * Hz
weight = 0.1
simulation_time = 100 * ms

# XOR girişleri ve hedef çıktıları
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Brian2 modeli
start_scope()

input_layer = PoissonGroup(num_inputs, rates=input_rate)
neuron_eqs = '''
dv/dt = (v_rest - v) / tau + (ge + gi) / tau : volt (unless refractory)
dge/dt = -ge / tau_e : volt
dgi/dt = -gi / tau_i : volt
v_rest : volt
tau : second
tau_e : second
tau_i : second
'''
neuron_layer = NeuronGroup(1, neuron_eqs, threshold='v > -50 * mV',
                           reset='v = -60 * mV', refractory=5 * ms,
                           method='euler')
neuron_layer.v = -60 * mV
neuron_layer.v_rest = -60 * mV
neuron_layer.tau = 20 * ms
neuron_layer.tau_e = 5 * ms
neuron_layer.tau_i = 10 * ms

excitatory_synapses = Synapses(input_layer, neuron_layer, on_pre='ge += weight * mV')
excitatory_synapses.connect(i=[0, 1], j=0)

inhibitory_synapses = Synapses(input_layer, neuron_layer, on_pre='gi -= weight * mV')
inhibitory_synapses.connect(i=[1, 0], j=0)

spike_monitor = SpikeMonitor(neuron_layer)

# XOR işlemi için girişlerin her birini test edelim
for i in range(len(X)):
    print(f"Input: {X[i]}")
    input_layer.rates = input_rate * X[i]
    run(simulation_time)
    output = spike_monitor.count[0]
    print(f"Output: {output} (Expected: {y[i]})")
    spike_monitor.count[:]=0