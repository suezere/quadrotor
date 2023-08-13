import numpy as np
from acados_template import AcadosSim, AcadosOcpSolver, AcadosSimSolver
from drone_model import drone_model

# create ocp object to formulate the simulation problem
sim = AcadosSim()

def export_drone_integrator(Ts, model_ac):
    # simulation time 
    Tsim = Ts

    # export model
    model_sim = model_ac

    # set model
    sim.model = model_sim


    # solver options
    sim.solver_options.integrator_type = 'IRK'

    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps  = 3

    # set prediction horizon
    sim.solver_options.T = Tsim

    # create the acados integrator
    acados_integrator = AcadosSimSolver(sim, json_file = 'acados_sim_' + model_sim.name + '.json')

    return acados_integrator
