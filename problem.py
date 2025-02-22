from algorithm.eda_model.EDALS import EDALS
from multiprocessing import Pool
import numpy as np
import time
import torch
from algorithm.CUSOP import ContinuousUnconstrainedSingleOpt

class CUSOP(ContinuousUnconstrainedSingleOpt):
    def __init__(self, problem_name='DemoA', config_path=None,**kwargs):
        super().__init__(problem_name, config_path,**kwargs)


if __name__ == '__main__':
    num_worker = 1  # 线程池数量  42  6
    fx = 1  # 2,3  # fx评估函数
    epoch = 150  # 迭代次数  150  100
    pop_size = 30  # 通常和线程池数量num_worker成倍数关系  42  30
    torch.multiprocessing.set_start_method('spawn')
    problem = CUSOP(problem_name='DemoA', config_path='config.yaml')
    opt_edals = EDALS(v_num=problem.n_var, lb=problem.xl, ub=problem.xu, population_size=pop_size, max_fes=epoch*pop_size)

    # opt_edals
    iter_num = 0
    avg_iter_time = []
    while not opt_edals.is_stop():
        start_time = time.time()
        solutions = opt_edals.ask()

        score_cls_list = []  # 每个个体的分数，转换成最小值优化
        pool = Pool(num_worker)
        for i, indis in enumerate(solutions):  # 遍历种群中每个个体
            score_cls_list.append(pool.apply_async(func=problem._evaluate, args=(iter_num, i, indis)))
        pool.close()
        pool.join()

        observe = problem._f_x(score_cls_list, fx)
        opt_edals.tell(solutions, observe)

        # show info
        Xs, ys = opt_edals.get_best()
        elapsed_time = int(time.time()-start_time)
        avg_iter_time.append(elapsed_time)
        print('iter: {}, loss: {}, elapsed_time: {}s, avg_iter_time: {}s, total_time: {}s'.format(iter_num, round(ys,4), elapsed_time, int(np.mean(avg_iter_time)), sum(avg_iter_time)))
        iter_num += 1
