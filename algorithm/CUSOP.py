from single_step.infer import SingleInference
from pymoo.core.problem import Problem as pymooProblem
from functools import reduce
from loguru import logger
from rdkit import Chem
import numpy as np
import yaml
import torch

from algorithm.utils import cal_similarity_with_FP, get_main_part_from_smistr, route_save_condition, read_txt

class ContinuousUnconstrainedSingleOpt(pymooProblem):
    """
    连续变量-无约束-单目标-优化问题
    """
    def __init__(self, problem_name='DemoA', config_path=None, **kwargs): 
        self.problem_name = problem_name
        self.avg_calls = []

        if config_path is not None:
            with open(config_path, 'r') as f:
                configs= yaml.safe_load(f)
                self.config = configs[self.problem_name]
        else:
            raise ValueError('config_path is None')
        super().__init__(n_var=self.config['step_len'], n_obj=1, xl=0, xu=1, vtype=float)
        
    def cal_score(self, react_smile):
        react_smiles = react_smile.split('.')
        if len(react_smiles) > 2:  # 如果最后一个分子是由超过2个分子组成的，score直接返回一个较小的值
            return -10
        else:
            smile_list = read_txt(self.config['building_block_dataset'])
            output_score = []
            for smile in react_smiles:
                compare_scaff_list = []
                for line in smile_list:
                    smile_score = cal_similarity_with_FP(smile, line)
                    compare_scaff_list.append(smile_score)
                output_score.append(np.mean(compare_scaff_list))
            return np.mean(output_score)

    # 初始化（0-1）之间的值映射成 scope 中的值
    def _get_value(self, x):
        scope = np.arange(0, self.config['beam_size'], 1)  # 初始化 [0 1 2 3 4 5 6 7 8 9]  <class 'numpy.ndarray'>
        x_bin = np.linspace(self.xl[0], self.xu[0], len(scope) + 1)
        for i in range(self.config['beam_size']):
            if x_bin[i] <= x < x_bin[i+1]:
                return scope[i]
    
    def evaluate_indivisual(self, iter_num, indis):
        reactant_predictor = SingleInference(beam_size=self.config['beam_size'])  # 用于单步预测的类 ， 替换成自己的单步模型
        react_smi = [self.config['target_smi']]
        react_chain = [self.config['target_smi']]
        react_chain_p = []  # 单步候选者的概率
        react_chain_index = []  # 单步候选者的index
        
        for j, indi in enumerate(indis):  # 遍历个体的每个节点 ，indi为0-1之间的值，需要转换成react
            init_node = self._get_value(indi)
            if j == 0:
                smi_nodes_sorted, prob_nodes_sorted = reactant_predictor.api_model_pred_reactant(self.config['target_smi'])  # 替换成自己的单步模型
                if len(smi_nodes_sorted) <= init_node:
                    logger.debug('next_node: {}'.format(next_node))
                    logger.debug('smi_nodes_sorted: {}'.format(smi_nodes_sorted))
                    logger.debug('len(smi_nodes_sorted): {}  init_node: {}'.format(len(smi_nodes_sorted), init_node))
                    break
                else:
                    current_smile = smi_nodes_sorted[init_node].split(',')[-1]
                    next_node = get_main_part_from_smistr(current_smile)
                    react_smi.append(current_smile)
                    react_chain.append(smi_nodes_sorted[init_node])
                    react_chain_p.append(prob_nodes_sorted[init_node])
            else:
                smi_nodes_sorted, prob_nodes_sorted = reactant_predictor.api_model_pred_reactant(next_node)  # 替换成自己的单步模型
                if len(smi_nodes_sorted) <= init_node:
                    logger.debug('next_node: {}'.format(next_node))
                    logger.debug('smi_nodes_sorted: {}'.format(smi_nodes_sorted))
                    logger.debug('len(smi_nodes_sorted): {}  init_node: {}'.format(len(smi_nodes_sorted), init_node))
                    break
                else:
                    current_smile = smi_nodes_sorted[init_node].split(',')[-1]
                    next_node = get_main_part_from_smistr(current_smile)
                    react_smi.append(current_smile)  # 不带有反应类型的反应路线
                    react_chain.append(smi_nodes_sorted[init_node])  # 带有反应类型的反应路线
                    react_chain_p.append(prob_nodes_sorted[init_node])
            
            react_chain_index.append(init_node)
            if route_save_condition(current_smile, self.config['building_block_dataset']):
                print('call_list---: {}'.format(' --> '.join(react_chain)))
        # 获取单步的概率，返回乘积作为权重，*score
        react_chain_p = np.round(np.multiply(10, react_chain_p), 4)
        probability_weight = reduce(lambda x,y : x*y, react_chain_p)

        reaction_smi = ".".join(react_smi)

        try:
            m = Chem.MolFromSmiles(reaction_smi) 
        except:
            m = None
        if m != None:
            sim_score = self.cal_score(react_smi[-1])  # react_smi[-1]为10个new_reaction反应中，每个反应路径的最后一个节点
        else:
            sim_score = 0
            print('Chem.MolFromSmiles返回空, 导致sim_score异常')
        
        self.avg_calls.append(reactant_predictor.calls)
        print('iter: {}, sum_calls: {}, call_list: {}'.format(iter_num, int(np.sum(self.avg_calls)),self.avg_calls))  
        # print('iter: {}'.format(iter_num))

        return sim_score * probability_weight

    def _evaluate(self, iter_num, i, indis):  # indis是个体
        available_GPU = [0,1,2,3]  # [0,1,2,3]
        GPU_ID = i % len(available_GPU)
        torch.cuda.set_device(available_GPU[GPU_ID])
        indis = torch.tensor(indis).cuda()
        return self.evaluate_indivisual(iter_num, indis)
    
    def _f_x(self, score_cls_list, eval_method):
        observe = []
        for score_cls in score_cls_list:
            observe.append(score_cls.get())
        if eval_method == 1:
            return (-1) * np.array(observe)
        elif eval_method == 2:
            score = np.exp(np.array(observe)) 
        elif eval_method == 3:
            score = np.random.choice(observe, size=len(observe), replace=True, p=np.array(observe)/sum(observe))
        return (-1) * np.array(score)