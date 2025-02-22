from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import subprocess
import numpy as np

def cal_similarity_with_FP(source_seq, target_seq):
	try:
		mol1 = Chem.MolFromSmiles(source_seq)  # 读取单个smiles字符串
		mol2 = Chem.MolFromSmiles(target_seq)
		mol1_fp = AllChem.GetMorganFingerprint(mol1, 2)  # 计算摩根指纹，radius=2 代表原子环境半径为2。 摩根型指纹是一种圆形指纹。每个原子的环境和连通性被分析到给定的半径，并且每种可能性都被编码。通常使用散列算法将很多可能性压缩到预定长度，例如1024。因此，圆形指纹是原子类型和分子连通性的系统探索
		mol2_fp = AllChem.GetMorganFingerprint(mol2, 2)
		score = DataStructs.DiceSimilarity(mol1_fp, mol2_fp)  # 比较分子之间的相似性
		return score
	except:
		return 0
	
def get_main_part_from_smistr(tmpseqs):
    """选择smile表达式最长的作为主要部分输出"""
    seq_list = tmpseqs.split(".")

    if len(seq_list) == 1:
        main_smi = tmpseqs
    else:
        main_smi = max(seq_list, key=len)
    return main_smi

def cano(smiles):
    """canonicalize smiles by MolToSmiles function"""
    try:
        canosmi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except:
        canosmi = ""
    return canosmi

def route_save_condition(react_chain, filename):
    substrings = react_chain.split('.')
    all_matched = True

    for substring in substrings:
        grep_command = ['grep', '-qxF', substring, filename]
        result = subprocess.run(grep_command)
        
        if result.returncode != 0:
            all_matched = False
            break
    return all_matched

def read_txt(path):
    with open(path, "r", encoding='utf-8') as f:  # 打开文件
        data = f.readlines()  # 读取文件
    return data