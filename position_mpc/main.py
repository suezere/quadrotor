from acados_settings import acados_settings
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from plotFnc import *
from utils import *
from trajectory import *

def Euler2Quaternion( phi, theta, psi):
        """
        Converts an euler angle attitude to a quaternian attitude
        :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
        :return: Quaternian attitude in np.array(e0, e1, e2, e3)
        """

        e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
        e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
        e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
        e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

        return np.asfarray([e0, e1, e2, e3], dtype = np.float32)



Tf = 1       # prediction horizon
N = 100      # number of discretization steps
T = 20.00    # simulation time[s]
Ts = Tf / N  # sampling time[s]

# noise bool
noisy_measurement = False

# load model and acados_solver
model, acados_solver, acados_integrator = acados_settings(Ts, Tf, N)

# constant 
g = 9.81 # m/s^2


# dimensions
nx = model.x.size()[0]
nu = model.u.size()[0]
# nparam = model.p.size()[0]
ny = nx + nu
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.ndarray((Nsim+1, nx))
simU = np.ndarray((Nsim, nu))
# simP = np.ndarray((Nsim, nparam))
tot_comp_sum = 0
tcomp_max = 0

# creating a reference trajectory
traj = 0 # traj = 0: circular trajectory, traj = 1: spiral trajectory
show_ref_traj = True
N_steps, x, y, z = trajectory_generator(T, Nsim, traj, show_ref_traj)
ref_traj = np.stack((x,y,z),1)
params = np.array([0.76, 0.1, 0.1, 0.1])
# set initial condition for acados integrator
xcurrent = np.array([0.5, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0])
simX[0,:] = xcurrent

# closed loop
for i in range(Nsim):

    for j in range(N):
        yref = np.array([x[i], y[i], z[i], 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.76, 0.0, 0.0])
        acados_solver.set(j, "yref", yref)
        acados_solver.set(j, "p", params)
    yref_N = np.array([x[i], y[i], z[i], 0.0, 0.0, 0.0, 0.0, 0.0])
    acados_solver.set(N, "yref", yref_N)
    acados_solver.set(N, "p", params)

    # solve ocp for a fixed reference
    acados_solver.set(0, "lbx", xcurrent)
    acados_solver.set(0, "ubx", xcurrent)
    comp_time = time.time()
    status = acados_solver.solve()
    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, i))

    elapsed = time.time() - comp_time

    # manage timings
    tot_comp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution from acados_solver
    xcurrent_pred = acados_solver.get(1, "x")
    u0 = acados_solver.get(0, "u")

    # computed inputs
    Thrust = u0[0]
    Roll = u0[1]
    Pitch = u0[2]

    quaternion = Euler2Quaternion(Roll, Pitch, 0)

    # making sure that q is normalized
    # quaternion = unit_quat(quaternion)

    # stacking u0 again
    u0 = np.array([Thrust,Roll,Pitch])

    # storing inputs
    simU[i,:] = u0

    # simulate the system
    acados_integrator.set("x", xcurrent)
    acados_integrator.set("u", u0)
    status = acados_integrator.solve()
    if status != 0:
        raise Exception('acados integrator returned status {}. Exiting.'.format(status))
    
    # update state
    xcurrent = acados_integrator.get("x")
    
    # add measurement noise
    if noisy_measurement == True:
        xcurrent = add_measurement_noise(xcurrent)
    
    # store state
    simX[i+1,:] = xcurrent

# RMSE
rmse_x, rmse_y, rmse_z = rmseX(simX, ref_traj)

# print the computation times
print("Average computation time: {}".format(tot_comp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))

# print the RMSE on each axis
print("RMSE on x: {}".format(rmse_x))
print("RMSE on y: {}".format(rmse_y))
print("RMSE on z: {}".format(rmse_z))
print(ref_traj)

# simU_euler = np.zeros((simU.shape[0], 3))

# for i in range(simU.shape[0]):
#     simU_euler[i, :] = quaternion_to_euler(simU[i, 1:])

# simU_euler = R2D(simU_euler)



# Plot Results
t = np.linspace(0,T, Nsim)
plotSim_pos(t, simX, ref_traj, save=False)
plotSim_vel(t, simX, save=False)
# plotThrustInput(t, simU,save=True)
# plotAngleInputs(t,simU, simU_euler, save=True)
plotSim3D(simX, ref_traj, save=False)

plt.show()
