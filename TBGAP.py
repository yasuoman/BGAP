# project : BGAP
# file   : TBGAP.py
# author:yasuoman
# datetime:2024/3/27 11:31
# software: PyCharm

"""
description：
说明：
"""
# 参考的文献Mazzola J B, Neebe A W. Bottleneck generalized assignment problems[J].
# Engineering Costs and Production Economics, 1988, 14(1): 61-65.
#且实现的是TBGAP

# 实现的总体思路：
# 1初始化相关的输入数据
# 2 根据cij与ck的关系建立新的TGBAP(K)问题
# 3 找到Z的下限，从这个下限开始往更大的数方向寻找
# 4 TGBAP(K)是否存在可行解，如果不存在的话，继续往下个数找，直到找到一个可行的TGBAP(K)
# 5 输出这个可行方案和对应的最小最大时间
import numpy as np

#这里是相关的数据集,输出相关的数据和变量
def construct_dataset():
    m, n = 5, 10
    # 运行成本矩阵
    cost_matrix = np.array(
        [[36,102,35,31,18,25,30,76,108,82],
         [61,75,69,19,45,97,117,74,35,85],
         [34,79,26,114,27,44,25,76,93,89],
         [17,97,65,51,81,82,89,40,21,95],
         [70,7,74,79,74,44,52,94,107,108]])
    #
    # cost_matrix = np.array(
    #     [[36, 102, 35, 31, 18, 25, 30, 76, 108, 65],
    #      [61, 75, 69, 19, 45, 97, 117, 74, 35, 85],
    #      [34, 79, 26, 114, 27, 44, 25, 76, 93, 76],
    #      [17, 97, 69, 51, 81, 82, 89, 40, 21, 95],
    #      [70, 7, 74, 79, 74, 44, 52, 94, 107, 108]])
    # 资源需求矩阵
    resource_matrix = np.array(
        [[78,14,82,70,87,93,78,34,7,36],
        [59,28,40,89,69,21,3,32,70,33],
        [72,40,95,6,85,60,94,25,9,29],
        [96,16,34,57,39,29,20,62,95,16],
        [39,98,33,24,45,61,59,7,12,12]])

    # resource_matrix = np.array(
    #     [[78, 14, 82, 70, 87, 93, 78, 34, 7, 36],
    #      [59, 28, 40, 89, 69, 21, 3, 32, 70, 33],
    #      [72, 40, 95, 6, 85, 60, 94, 25, 9, 29],
    #      [96, 16, 34, 57, 39, 29, 20, 62, 95, 16],
    #      [39, 98, 33, 24, 45, 61, 59, 7, 12, 12]])

    # 机器资源容量向量
    capacity_vector = np.array([93,71,82,74,62])

    return m,n,cost_matrix,resource_matrix,capacity_vector
#这里是对http://www.al.cm.is.nagoya-u.ac.jp/~yagiura/gap/ 的a20100数据集进行简单的测试
#目前没有优化这组数据集的读取，只是写了个示例。有需要可以自行写这里的代码
# def construct_dataset():
#     with open('Data/gap_a/a20100', 'r') as file:
#         #先随便写着
#         import re
#         # 读取文件内容
#         content = file.read()
#         # words = content.split(' ')
#         words= re.split(r'[ ,\n]+', content)
#
#         m,n = int(words[1]),int(words[2])
#         c_list = words[3:2003]
#         r_list = words[2003:4003]
#         cap_list = words[4003:4023]
#         c_int_list = [int(item) for item in c_list]
#         r_int_list = [int(item) for item in r_list]
#         cap_int_list = [int(item) for item in cap_list]
#         cost_matrix = np.array(c_int_list).reshape(m, n)
#         resource_matrix = np.array(r_int_list).reshape(m, n)
#         capacity_vector =np.array(cap_int_list)
#         return m, n, cost_matrix, resource_matrix, capacity_vector



#输入resource_matrix、cost_matrix、capacity_vector和初始的k，输出新的resource_matrix矩阵
def reconstruct_resource_matrix(resource_matrix, cost_matrix, capacity_vector,k):
    import copy
    copy_resource_matrix =copy.deepcopy(resource_matrix)
    # 使用fancy indexing来更新矩阵
    mask = k < cost_matrix
    # resource_matrix[mask] = 9999
    copy_resource_matrix[mask] = max(capacity_vector)
    return copy_resource_matrix

#输入新的resource_matrix矩阵和capacity_vector，输出一组可行解或输出FALSE，借用Pulp包求解
def find_soulution(resource_matrix,capacity_vector,m,n):
    import pulp
    # 创建问题实例
    prob = pulp.LpProblem("Machine_Assignment", pulp.LpMinimize)
    # 二元决策变量
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n)),
                              cat=pulp.LpBinary)
    # 目标函数：这里我们只需要找到可行解，因此可以设置一个任意的目标函数
    prob += 0
    # 约束条件
    # 1每个工件只能在一个机器上运行
    for j in range(n):
        prob += sum(x[(i, j)] for i in range(m)) == 1

    # 2每个机器的资源需求之和不能大于资源容量
    for i in range(m):
        prob += sum(resource_matrix[i][j] * x[(i, j)] for j in range(n)) <= capacity_vector[i]
    # 求解问题
    # status = prob.solve()
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 输出结果
    if pulp.LpStatus[status] == 'Optimal':
        solution = [0] * n
        for i in range(m):
            for j in range(n):
                if pulp.value(x[(i, j)]) == 1:
                    solution[j] = i + 1  # 机器编号从1开始
        return solution
    else:
        # 无法找到可行解
        return False

def main():
    #得到数据
    m, n, cost_matrix, resource_matrix, capacity_vector = construct_dataset()
    # 构建待遍历的数组
    # 得到目标函数的下界 max min(cij), 找到每一列的最小值
    min_values = np.min(cost_matrix, axis=0)
    # 从最小值中找到最大值
    max_of_mins = np.max(min_values)
    #对矩阵进行排序并去重，得到一维数组ck
    ck= np.unique(np.sort(cost_matrix, axis=None))
    # 截取从下界开始到数组ck末尾的数据，保存到新的数组new_ck中
    new_ck = ck[ck >= max_of_mins]

    # object_value = new_ck[0]
    #遍历去找
    for i,k in np.ndenumerate(new_ck):
        new_resource_matrix =reconstruct_resource_matrix(resource_matrix, cost_matrix, capacity_vector, k)
        solution = find_soulution(new_resource_matrix,capacity_vector,m,n)

        if solution != False:
            print("第"+str(i)+"次,分配方案为:",solution,"最优运行时间为：",k)

            break
        else:
            print("第"+str(i)+"次,最优运行时间的分配方案"+str(k)+"不存在")


if __name__ == "__main__":
    main()

