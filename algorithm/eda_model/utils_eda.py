import numpy as np
from pyDOE import lhs
from abc import ABC

class Solution(ABC):
    def __init__(self, variables_num: int, objectives_num: int, constraints_num: int = 0):
        self.variables_num = variables_num
        self.objectives_num = objectives_num
        self.constraints_num = constraints_num

        self.decs = []
        self.objs = []
        self.cons = []


class FloatSolution(Solution):
    def __init__(self, lb, ub, objectives_num, constraints_num: int = 0):
        super(FloatSolution, self).__init__(len(lb), objectives_num, constraints_num)
        self.lb = lb
        self.ub = ub


def get_data_from_population(pop_list, data_type: str):
    if data_type.lower() not in ['decs', 'objs']:
        raise Exception('data_type is not defined')

    if data_type.lower() == 'decs':
        return np.array([s.decs for s in pop_list])
    elif data_type.lower() == 'objs':
        return np.array([s.objs for s in pop_list])

def float_random_init(lb, ub, objectives_num, constraints_num, popsize):
    """
    随机采样 float solutions
    :param lb:
    :param ub:
    :param objectives_num:
    :param constraints_num:
    :param popsize:
    :return:
    """
    pop_list = []
    Decs = np.random.uniform(lb, ub, [popsize, len(lb)])
    for i in range(popsize):
        new_solution = FloatSolution(lb=lb, ub=ub, objectives_num=objectives_num, constraints_num=constraints_num)
        new_solution.decs = Decs[i, :]
        pop_list.append(new_solution)
    return pop_list


def gen_float_solution_via_matrix(lb, ub, objectives_num, constraints_num, Decs, Objs=None):
    """
    通过矩阵 产生 float solutions
    :param lb:
    :param ub:
    :param objectives_num:
    :param constraints_num:
    :param Decs:
    :return:
    """
    pop_list = []
    if Objs is None:
        for x in Decs:
            new_solution = FloatSolution(lb=lb, ub=ub, objectives_num=objectives_num, constraints_num=constraints_num)
            new_solution.decs = x
            pop_list.append(new_solution)
    else:
        for x, y in zip(Decs, Objs):
            new_solution = FloatSolution(lb=lb, ub=ub, objectives_num=objectives_num, constraints_num=constraints_num)
            new_solution.decs = x
            new_solution.objs = y
            pop_list.append(new_solution)

    return pop_list


def assign_fitness_float_solution_via_matrix(poplist, Objs):
    for pop, objs in zip(poplist, Objs):
        pop.objs = objs
    return poplist


def float_lhs_init(lb, ub, objectives_num, constraints_num, popsize):
    """
    通过拉丁方采样生成新解
    :param lb:
    :param ub:
    :param objectives_num:
    :param constraints_num:
    :param popsize:
    :return:
    """
    pop_list = []
    raw_sample_data = lhs(len(lb), popsize, criterion='cm')
    LB = np.array(lb)
    UB = np.array(ub)
    Decs = LB + raw_sample_data * (UB - LB)
    for i in range(popsize):
        new_solution = FloatSolution(lb=lb, ub=ub, objectives_num=objectives_num, constraints_num=constraints_num)
        new_solution.decs = Decs[i, :]
        pop_list.append(new_solution)
    return pop_list