from os.path import join

from numpy import arange
from pyomo.environ import ConcreteModel, Var, Binary, maximize
from pyomo.opt import ProblemFormat
from pypsa.opt import l_constraint, LConstraint, l_objective, LExpression

from tools import retrieve_index_dict


def build_ip_model(deployment_dict, coordinate_dict, critical_matrix, c, output_folder, write_lp=False):
    """Model build-up.

    Parameters:

    ------------


    Returns:

    -----------

    """

    no_windows = critical_matrix.shape[0]
    no_locations = critical_matrix.shape[1]

    n, dict_deployment, partitions, indices = retrieve_index_dict(deployment_dict, coordinate_dict)

    for item in partitions:
        if item in indices:
            if dict_deployment[item] > len(indices[item]):
                raise ValueError(' More nodes required than available for {}'.format(item))
        else:
            indices[item] = []
            print('Warning! {} not in keys of choice. Make sure there is no requirement here.'.format(item))

    model = ConcreteModel()

    model.W = arange(1, no_windows + 1)
    model.L = arange(1, no_locations + 1)

    model.x = Var(model.L, within=Binary)
    model.y = Var(model.W, within=Binary)

    activation_constraint = {}

    for w in model.W:
        lhs = LExpression([(critical_matrix[w - 1, l - 1], model.x[l]) for l in model.L])
        rhs = LExpression([(c, model.y[w])])

        activation_constraint[w] = LConstraint(lhs, ">=", rhs)

    l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

    cardinality_constraint = {}

    for item in partitions:
        lhs = LExpression([(1, model.x[l]) for l in indices[item]])
        rhs = LExpression(constant=dict_deployment[item])

        cardinality_constraint[item] = LConstraint(lhs, "==", rhs)

    l_constraint(model, "cardinality_constraint", cardinality_constraint, partitions)

    objective = LExpression([(1, model.y[w]) for w in model.W])
    l_objective(model, objective, sense=maximize)

    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model
