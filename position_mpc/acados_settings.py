from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver,AcadosSim, AcadosSimSolver
import numpy as np
from casadi import SX, vertcat, sin, cos
import math
from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

def quadrotor_model() -> AcadosModel:

    model_name = 'quadrotor'

    # system parameters
    g = 9.80665                        # gravity constant [m/s^2]
    
    # hover_thrust = 0.76
    # tau_roll = 0.01667             # Inner-loop controller time constants
    # tau_pitch = 0.01667
    # tau_yaw = 0.01667
    hover_thrust = SX.sym('hover_thrust')
    tau_roll = SX.sym('tau_roll')             
    tau_pitch = SX.sym('tau_pitch')
    yaw = SX.sym('yaw')
    sym_p = vertcat(hover_thrust,tau_roll,tau_pitch,yaw)

    # states
    x = SX.sym('x')                 # earth position x
    y = SX.sym('y')                 # earth position y
    z = SX.sym('z')                 # earth position z
    u = SX.sym('u')                 # earth velocity x
    v = SX.sym('v')                 # earth velocity y
    w = SX.sym('w')                 # earth velocity z
    roll = SX.sym('roll')             # roll angle
    pitch = SX.sym('pitch')         # pitch angle
    sym_x = vertcat(x,y,z,u,v,w,roll,pitch)

    # controls
    thrust = SX.sym('thrust')       # thrust command
    roll_cmd = SX.sym('roll_cmd')     # roll angle command
    pitch_cmd = SX.sym('pitch_cmd') # pitch angle command
    sym_u = vertcat(thrust,roll_cmd,pitch_cmd)

    # xdot for f_impl
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    z_dot = SX.sym('z_dot')
    u_dot = SX.sym('u_dot')
    v_dot = SX.sym('v_dot')
    w_dot = SX.sym('w_dot')
    roll_dot = SX.sym('roll_dot')
    pitch_dot = SX.sym('pitch_dot')
    sym_xdot = vertcat(x_dot,y_dot,z_dot,u_dot,v_dot,w_dot,roll_dot,pitch_dot)

    # dynamics
    dx = u
    dy = v
    dz = w
    du = sin(pitch) * cos(roll) * thrust/hover_thrust*g
    dv = -sin(roll) * thrust/hover_thrust*g
    dw = -g + cos(pitch) * cos(roll) * thrust/hover_thrust*g
    droll = (roll_cmd - roll) / tau_roll
    dpitch = (pitch_cmd - pitch) / tau_pitch
    f_expl = vertcat(dx,dy,dz,du,dv,dw,droll,dpitch)

    f_impl = sym_xdot - f_expl


    
    # constraints
    h_expr = sym_u
    
    # cost
    #W_x = np.diag([120, 120, 120, 10, 10, 10, 10, 10])
    #W_u = np.diag([5000, 2000, 2000])

    #expr_ext_cost_e = sym_x.transpose()* W_x * sym_x
    #expr_ext_cost = expr_ext_cost_e + sym_u.transpose() * W_u * sym_u
    

    # nonlinear least sqares
    cost_y_expr = vertcat(sym_x, sym_u)
    #W = block_diag(W_x, W_u)
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = sym_xdot
    model.u = sym_u
    model.p = sym_p
    model.cost_y_expr = cost_y_expr
    model.cost_y_expr_e = sym_x
    #model.con_h_expr = h_expr
    model.name = model_name
    #model.cost_expr_ext_cost = expr_ext_cost
    #model.cost_expr_ext_cost_e = expr_ext_cost_e 

    return model

def acados_settings(Ts, Tf, N):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = quadrotor_model()
    ocp.model = model

    
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    nparam = model.p.size()[0]

    # set dimensions
    ocp.dims.N = N

    ocp.parameter_values = np.zeros((nparam, ))

    # set cost
    W_x = np.diag([60, 60, 60, 30, 30, 30, 10, 10])       #Q_mat
    W_u = np.diag([5000, 2000, 2000])                     #R_mat
    W = block_diag(W_x, W_u)
    ocp.cost.W_e = W_x
    ocp.cost.W = W

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    #ocp.cost.cost_type = 'EXTERNAL'
    #ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_expr_ext_cost = model.x.T @ W_x @ model.x + model.u.T @ W_u @ model.u
    ocp.model.cost_expr_ext_cost_e = model.x.T @ W_x @ model.x
    

    # set constraints
    u_min = np.array([0.3, -math.pi/2, -math.pi/2])
    u_max = np.array([0.9, math.pi/2, math.pi/2])
    x_min = np.array([-math.pi/2,-math.pi/2])
    x_max = np.array([math.pi/2,math.pi/2])    
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = np.array([0,1,2])
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx = np.array([6,7])
    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # reference trajectory (will be overwritten later)
    x_ref = np.zeros(nx)
    ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0])))
    ocp.cost.yref_e = x_ref

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    #ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.qp_solver_cond_N = 5
    #ocp.solver_options.print_level = 0
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP


    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    ocp_integrator = quadrotor_integrator(Ts, model)
    # simX = np.ndarray((N+1, nx))
    # simU = np.ndarray((N, nu))

    # status = ocp_solver.solve()
    # ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    # if status != 0:
    #     raise Exception(f'acados returned status {status}.')

    # # get solution
    # for i in range(N):
    #     simX[i,:] = ocp_solver.get(i, "x")
    #     simU[i,:] = ocp_solver.get(i, "u")
    #     simX[N,:] = ocp_solver.get(N, "x")
    return model,ocp_solver,ocp_integrator

def quadrotor_integrator(Ts, model):
    sim = AcadosSim()
    
    # simulation time 
    Tsim = Ts

    # export model
    model_sim = model

    # set model
    sim.model = model_sim
    # sim.parameter_values = np.zeros((4, ))
    sim.parameter_values = np.array([0.76, 0.1, 0.1, 0.1])
    # solver options
    sim.solver_options.integrator_type = 'ERK'

    # set prediction horizon
    sim.solver_options.T = Tsim

    # create the acados integrator
    acados_integrator = AcadosSimSolver(sim, json_file = 'acados_sim.json')

    return acados_integrator    

