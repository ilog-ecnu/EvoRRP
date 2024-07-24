import copy
import numpy as np

from algorithm.eda_model.edamodel import VWH, local_search
from algorithm.eda_model.utils_eda import float_lhs_init, gen_float_solution_via_matrix, get_data_from_population

# from pyemo.util.population_init import float_lhs_init
# from pyemo.problem.singleobjproblem.lzg import LZG03 as Ackley
# from pyemo.problem.singleobjproblem.lzg import LZG01

class EDALS(object):
    def __init__(self, v_num=None, lb=None, ub=None, Pb=0.2, Pc=0.2, M=10, population_size=50, max_fes=8000):
        self.population_size = population_size
        if v_num is None:
            raise ValueError('v_num is None')
        self.v_num = v_num

        if lb is None:
            raise ValueError('lb is None')

        self.lb = lb

        if ub is None:
            raise ValueError('ub is None')

        self.ub = ub

        self.Pc = Pc
        self.Pb = Pb
        self.M = M
        self.NL = int(np.floor(self.population_size * self.Pb))

        self.EDA = VWH(
            M=self.M,
            D=self.v_num,
            LB=np.array(self.lb),
            UB=np.array(self.ub)
        )

        self.pop_init = False
        self.alpha = self.population_size  # 初始化种群大小

        self.to_eva_list = []
        self.eva_num = 0
        self.max_fes = max_fes

    def is_stop(self):
        return self.eva_num > self.max_fes

    def population_init(self):
        population = float_lhs_init(self.lb, self.ub, self.v_num, 0, self.alpha)
        return population

    def ask(self):
        if not self.pop_init:
            self.to_eva_list = self.population_init()
            return [p.decs for p in self.to_eva_list]
        else:
            current_pop = self.current_pop
            # 产生新解
            new_decs = self.reproduction(current_pop)
            return [x for x in new_decs.tolist()]

    def save_react_chain(self, content, save_path):    
        with open(save_path, "a+") as f:
            f.write(content + "\n")

    def tell(self, Xs, ys):
        new_population = gen_float_solution_via_matrix(self.lb, self.ub, objectives_num=self.v_num, constraints_num=0,
                                                       Decs=Xs, Objs=ys)
        self.eva_num += len(new_population)

        if not self.pop_init:
            self.pop_init = True
            self.current_pop = new_population
        else:
            # self.save_react_chain('{}'.format(self.get_current_pop(self.current_pop)), 'ga/react_chain/demo_d_current_pop-03-02.txt')
            # self.save_react_chain('{}'.format(self.get_current_pop(new_population)), 'ga/react_chain/demo_d_new_population-03-02.txt')
            self.current_pop = self.selection(self.current_pop, new_population)

    def get_current_pop(self, population):
        cur_p = copy.deepcopy(population)
        cur_pop = []
        for i, content in enumerate(cur_p):
            cur_pop.append([cur_p[i].decs, cur_p[i].objs])
        return cur_pop

    def get_best(self):
        t_p = copy.deepcopy(self.current_pop)
        t_p.sort(key=lambda s: s.objs)
        print('t_p:', t_p)
        return t_p[0].decs, t_p[0].objs

    def get_mean(self):
        t_p = copy.deepcopy(self.current_pop)
        all_fx = []
        for line in t_p:
            all_fx.append(line.objs)
        return np.mean(all_fx)

    def selection(self, f_pop, son_pop):
        f_pop.extend(son_pop)
        f_pop.sort(key=lambda s: s.objs)
        return f_pop[:self.population_size]

    def selection_de(self, f_pop, son_pop):
        select_result = []
        for i in range(len(f_pop)):
            if f_pop[i].objs >= son_pop[i].objs:
                select_result.append(f_pop[i])
            else:
                select_result.append(son_pop[i])
        return select_result

    def get_name(self):
        return 'EDALS'

    def reproduction(self, pop):
        Xs = get_data_from_population(pop, 'decs')
        ys = get_data_from_population(pop, 'objs')

        I = np.argsort(ys)
        Xs = Xs[I]
        ys = ys[I]
        self.EDA.update(Xs, ys)

        Xs_new = self.EDA.sample(self.population_size)

        Xs_l = local_search(Xs[:self.NL, :], ys[:self.NL])
        I = np.floor(np.random.random((self.population_size, 1)) * (Xs_l.shape[0] - 2)).astype(int).flatten()
        xtmp = Xs_l[I, :]
        mask = np.random.random((self.population_size, self.v_num)) < self.Pc
        Xs_new[mask] = xtmp[mask]

        # boundary checking
        lb_matrix = self.lb * np.ones(shape=Xs_new.shape)
        ub_matrix = self.ub * np.ones(shape=Xs_new.shape)
        pos = Xs_new < self.lb
        Xs_new[pos] = 0.5 * (Xs[pos] + lb_matrix[pos])
        pos = Xs_new > self.ub
        Xs_new[pos] = 0.5 * (Xs[pos] + ub_matrix[pos])
 
        return Xs_new


# if __name__ == '__main__':
    # # problem
    # prob = LZG01(10)
    # v_num = prob.variables_num
    # lb = prob.lb
    # ub = prob.ub
    #
    # # lb = [-1e10 for _ in range(prob.variables_num)]
    # # ub = [1e10 for _ in range(prob.variables_num)]
    #
    # # algorithm
    # opt = EDALS(v_num=v_num, lb=lb, ub=ub, max_fes=8000)
    #
    # # opt
    # iter_num = 0
    # while not opt.is_stop():
    #     solutions = opt.ask()
    #     observe = [prob.obj_func(x) for x in solutions]
    #     opt.tell(solutions, observe)
    #     Xs, ys = opt.get_best()
    #     print('after {} itre the best value is {}'.format(iter_num, ys))
    #     iter_num += 1
