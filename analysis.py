import torch
import matplotlib.pyplot as plt

STATE_SAVE_RAVI_FILE = 'ravi_saved_state.tar'
STATE_SAVE_BAPU_FILE = 'bapu_saved_state.tar'
STATE_SAVE_COMBINED_FILE = 'combined_saved_state.tar'
STATE_SAVE_CIFAR10_FILE = 'cifar10_saved_state.tar'

state_ravi = torch.load(STATE_SAVE_RAVI_FILE)
state_bapu = torch.load(STATE_SAVE_BAPU_FILE)
state_combined = torch.load(STATE_SAVE_COMBINED_FILE)
state_cifar10 = torch.load(STATE_SAVE_CIFAR10_FILE)

ravi_iteration_time_list = state_ravi['i_time_list']
bapu_iteration_time = state_bapu['i_time_list']
combined_iteration_time = state_combined['i_time_list']
cifar10_iteration_time = state_cifar10['i_time_list']

cifar10_iteration_time = cifar10_iteration_time[:3500]
combined_iteration_time = combined_iteration_time[:3500]

iteration_list = list()
for i in range(len(ravi_iteration_time_list)):
    iteration_list.append(i)

plt.plot(iteration_list, cifar10_iteration_time, label='CIFAR 10')
plt.plot(iteration_list, bapu_iteration_time, label='BAPU')
plt.plot(iteration_list, ravi_iteration_time_list, label='RAVI')
plt.plot(iteration_list, combined_iteration_time, label='COMBINED')
plt.xlabel('Iteration Number')
plt.ylabel('Time Taken For Iteration')
plt.legend()
plt.grid(True)
plt.show()

ravi_generator_loss = state_ravi['g_loss_list']
ravi_discriminator_loss = state_ravi['d_loss_list']
plt.plot(iteration_list, ravi_generator_loss, label='RAVI Generator Loss')
plt.plot(iteration_list, ravi_discriminator_loss, label='RAVI Discrimator Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

bapu_generator_loss = state_bapu['g_loss_list']
bapu_discriminator_loss = state_bapu['d_loss_list']
plt.plot(iteration_list, bapu_generator_loss, label='Bapu Generator Loss')
plt.plot(iteration_list, bapu_discriminator_loss, label='Bapu Discrimator Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

combined_generator_loss = state_combined['g_loss_list']
combined_discriminator_loss = state_combined['d_loss_list']
iteration_list = list()
for i in range(len(combined_generator_loss)):
    iteration_list.append(i)
plt.plot(iteration_list, combined_generator_loss, label='Combined Generator Loss')
plt.plot(iteration_list, combined_discriminator_loss, label='Combined Discrimator Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


cifar10_generator_loss = state_cifar10['g_loss_list']
cifar10_discriminator_loss = state_cifar10['d_loss_list']
iteration_list = list()
for i in range(len(cifar10_generator_loss)):
    iteration_list.append(i)
plt.plot(iteration_list, cifar10_generator_loss, label='CIFAR10 Generator Loss')
plt.plot(iteration_list, cifar10_discriminator_loss, label='CIFAR10 Discrimator Loss')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
