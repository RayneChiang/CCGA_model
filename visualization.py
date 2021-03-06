import numpy as np
from matplotlib import pyplot as plt
import math
Rastrigin_ga_result = np.load('Rastrigin_ga.npy')
Rastrigin_ccga_result = np.load('Rastrigin_ccga.npy')
Griewank_ga_result = np.load('Griewank_ga.npy')
Griewank_ccga_result = np.load('Griewank_ccga.npy')
Schewefel_ga_result = np.load('Schewefel_ga.npy')
Schewefel_ccga_result = np.load('Schewefel_ccga.npy')
Ackley_ga_result = np.load('Ackley_ga.npy')
Ackley_ccga_result = np.load('Ackley_ccga.npy')

X_ga = np.load('X_ga_400.npy')
X_ccga = np.load('X_ccga_400.npy')
X_svg_ccga = np.load('X_svg_ccga_400.npy')

X_ga = np.load('X_ga_400.npy')
X_ccga_2 = np.load('X_ccga_2_400.npy')
X_svg_ccga_2 = np.load('X_svg_ccga_2_400.npy')
X_ccga_2_2 = np.load('X_ccga_2_20.npy')
X_svg_ccga_2_2 = np.load('X_svg_ccga_2_20.npy')

X_ccga_1 = np.load('X_ccga_1_400.npy')
X_svg_ccga_1 = np.load('X_svg_ccga_1_400.npy')
X_ccga_1_2 = np.load('X_ccga_1_20.npy')
X_svg_ccga_1_2 = np.load('X_svg_ccga_1_20.npy')



R_ccga = np.load('Rccga.npy')
R_s_ccga = np.load('R_s_ccga.npy')
S_ga = np.load('S_ga.npy')
S_s_ccga = np.load('S_s_ccga.npy')
G_ccga = np.load('G_ga.npy')
G_s_ccga = np.load('G_svg_ccga.npy')
A_ccga = np.load('A_ga.npy')
A_s_ccga = np.load('A_ccga.npy')

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
# print(np.unique(Rastrigin_ga_result))
# print(np.unique(Rastrigin_ccga_result))



# x_ccga_list = list(map(math.log10, X_ccga_2))
# x_svg_list = list(map(math.log10, X_svg_ccga_2))
# x_ccga_list_2 = list(map(math.log10, X_ccga_2_2))
# x_svg_list_2 = list(map(math.log10, X_svg_ccga_2_2))
# ax[0].plot(range(len(X_ccga_2)), X_ccga_1, label='1-D-ccga')
# ax[0].plot(range(len(X_svg_ccga_2)), X_svg_ccga_1, label='10*40-ccga')
# ax[0].set_ylabel('log10(best individual fitness)')
# ax[0].set_xlabel('epoches')
# ax[0].set_title('400 dimensions')
# ax[0].legend()
# ax[0].grid()
# ax[1].plot(range(len(X_ccga_2_2)), X_ccga_1_2, label='1-D-ccga')
# ax[1].plot(range(len(X_svg_ccga_2_2)), X_svg_ccga_1_2, label='10*40-ccga')
# ax[1].set_ylabel('log10(best individual fitness)')
# ax[1].set_xlabel('epoches')
# ax[1].set_title('20 dimensions')
# ax[1].legend()
# ax[1].grid()
ax[0][0].plot(range(len(R_ccga)), R_ccga,  label='1-D-ccga')
ax[0][0].plot(range(len(R_s_ccga)), R_s_ccga, label='5*4-ccga')
ax[0][0].set_title('Rastrigin')
ax[0][0].legend()
ax[0][0].grid()
ax[0][0].set_ylim(0, 4)
ax[1][0].plot(range(len(S_ga)), S_ga,  label='1-D-ccga')
ax[1][0].plot(range(len(S_s_ccga)), S_s_ccga, label='5*2-ccga')
ax[1][0].set_title('Schewefel')
ax[1][0].set_ylim(0, 4)
ax[1][0].legend()
ax[1][0].grid()
ax[0][1].plot(range(len(G_ccga)), G_ccga, label='1-D-ccga')
ax[0][1].plot(range(len(G_s_ccga)), G_s_ccga, label='5*2-ccga')
ax[0][1].set_title('Griewangk')
ax[0][1].set_ylim(0, 4)
ax[0][1].legend()
ax[0][1].grid()
ax[1][1].plot(range(len(A_ccga)), A_ccga, label='1-D-ccga')
ax[1][1].plot(range(len(A_s_ccga)), A_s_ccga, label='5*6-ccga')
ax[1][1].set_title('Ackley')
ax[1][1].set_ylim(0, 4)
plt.show()

# ax.set_xlabel('function evaluations')
# ax.set_ylabel('best individual')
# plt.savefig('static variable')
# plt.savefig('xinshenyang_1')


# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(range(len(G_ccga)), G_ccga, label='standard_ga')
# ax.plot(range(len(G_s_ccga)), G_s_ccga, label='ccga')
# ax.set_xlabel('function evaluations')
# ax.set_ylabel('best individual')
# ax.set_ylim(0, 50)
# ax.set_title('Schewefel function')
# plt.grid()
# plt.legend()
# plt.show()
# plt.savefig('Schewefel function')