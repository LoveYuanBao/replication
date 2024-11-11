import numpy as np
import math
import random
from collections import deque
import bisect
from PrefixTree import PrefixTreeNode

# 根据原始数据集构造前缀树,dataset[[(timestamp, location)]]
def build_initial_prefix_tree(prefix_tree, dataset):
    # Step 1: Build initial prefix tree by counting each sub-trajectory
    for trajectory in dataset:
        current_node = prefix_tree.root
        for point in trajectory:
            temp_node = current_node.children.get(point)
            if  temp_node is None:
                current_node.children[point] = PrefixTreeNode(point[0], point[1], count = 1)
                current_node = current_node.children[point]
            else:
                temp_node.count += 1
                current_node = temp_node

# 添加拉普拉斯噪音                
def add_noise(prefix_node, budget):
    # Add Laplace noise to the count
    noise = np.random.laplace(scale = 1/budget)
    prefix_node.count += noise

# 根据层级计算隐私预算
def calculate_budget_allocation(prefix_tree, total_budget, max_height):
    # Calculate incremental privacy budget for each level
    level_budget_list = [0] * (max_height + 1)
    sum = 0
    for i in range(1, max_height + 1):
        level_budget_list[i] = np.log(i + prefix_tree.o)
        sum += level_budget_list[i]
    for i in range(1, max_height + 1):
        level_budget_list[i] = level_budget_list[i]/sum * total_budget
    return level_budget_list

# 根据层级计算阈值
def calculate_threshold(prefix_tree, max_height):
    # Calculate threshold based on level
    level_threshold_list = [0] * (max_height + 1)
    for i in range(1, max_height + 1):
        level_threshold_list[i] = prefix_tree.k / i + prefix_tree.b
    return level_threshold_list

# 根据初始前缀树构建噪音前缀树，算法主函数
def build_noisy_prefix_tree(prefix_tree, total_budget, max_height, loc_domain, time_domain, time_unit):
    # 隐私预算数组和阈值数组
    level_budget_list = calculate_budget_allocation(prefix_tree, total_budget, max_height)
    level_threshold_list = calculate_threshold(prefix_tree, max_height)
    # 可达矩阵(单位为1分钟)
    KL_matrix = create_KL_matrix(loc_domain)
    # 开始清理前缀树
    parent_count = 0
    parent_list = deque([])
    parent_list.append(prefix_tree.root)
    # 为第一层节点添加噪音
    for first_level_node in prefix_tree.root.children.values():
        add_noise(parent_node, level_budget_list[1])
        parent_list.append(first_level_node)
        parent_count += 1
    # 筛选每一层的子层(即下一层节点)
    for level in range(1, max_height + 1):
        level_budget = level_budget_list[level + 1]
        level_threshold = level_threshold_list[level + 1]
        handled_node_count = 0
        while parent_count > 0:
            parent_node = parent_list.popleft()
            handled_node_count += handle_subtree(parent_node, loc_domain, time_domain, KL_matrix, level_budget, level_threshold, parent_list, time_unit)
        parent_count = handled_node_count    
    # 噪音前缀树构造完成
    sanitized_trajectories = generate_sanitized_trajectories(prefix_tree)
    return sanitized_trajectories

# 计算可达矩阵
def create_KL_matrix(loc_domain):
    pass

# 根据可达矩阵，上一个位置的位置-时间戳和当前时间戳，返回候选的位置集合
def filter_location_domain(parent_node, child_timestamp, location_domain, location_knowledge_matrix, time_unit):
    # Filter reachable locations based on minimum travel time
    reachable_locations = []
    time_interval = child_timestamp - parent_node.loc_time[0]
    index = bisect.bisect_left(location_domain, parent_node.loc_time[1])
    location_index = location_knowledge_matrix[index][location_knowledge_matrix[index] <= time_interval * time_unit]
    for i in location_index:
        reachable_locations.append(location_domain[i])
    return reachable_locations

# 构建噪音前缀树的子层
def handle_subtree(parent_node, location_domain, time_domain, location_knowledge_matrix, level_budget, level_threshold, parent_list, time_unit):
    # Add child nodes with noise and filter based on location constraints
    sum_noisy_count = 0
    # 这里选择子节点的顺序有没有讲究,先处理原有子节点
    for child_node in parent_node.children.values:
        add_noise(child_node, level_budget)
        if child_node.count >= level_threshold:
            sum_noisy_count += child_node.count
            sum_noisy_count += 1
            parent_list.append(child_node)
            if sum_noisy_count >= parent_node.count:
                break
        else:
            parent_node.children.pop(child_node.loc_time, None)
    # 如果处理完原节点后，所有子节点noisy_count之和小于父节点，则添加空节点
    while sum_noisy_count < parent_node.count:
        # 随机选择时间戳，要求大于父节点的时间戳
        child_timestamp = random.randint(parent_node.loc_time[0] + 1, len(time_domain) - 1)
        available_location_domain = filter_location_domain(parent_node, child_timestamp, location_domain, location_knowledge_matrix, time_unit)
        for location in available_location_domain:
            noise = np.random.laplace(scale = 1/level_budget)
            if noise >= level_threshold:
                parent_node.children[(child_timestamp, location)] = PrefixTreeNode(child_timestamp, location, count = noise)
                sum_noisy_count += noise
                if sum_noisy_count >= parent_node.count:
                    break
        if sum_noisy_count >= parent_node.count:
                    break



# 根据噪音前缀树还原清理后的轨迹数据集
def generate_sanitized_trajectories(prefix_tree):
    """
    生成差分隐私的轨迹数据集，基于文献中描述的噪音前缀树。
    通过遍历噪音前缀树生成符合差分隐私要求的轨迹数据。

    参数:
    prefix_tree - 构建好的噪音前缀树

    返回:
    sanitized_trajectories - 经过差分隐私处理的轨迹数据集
    """
    sanitized_trajectories = []
    # 一致性处理+取整处理(level(1)的节点向上取整，其他层次的节点向下取整)
    def enhance_consistency(prefix_tree):
        parent_list = deque([])
        for value in prefix_tree.root.childen:
            math.ceil(value.count)
            parent_list.append(value)
        while parent_list:
            parent = parent_list.popleft()
            child_nodes = parent.children
            if child_nodes:
                child_count = 0
                child_sum = 0
                for value in child_nodes:
                    child_sum += value.count
                    child_count += 1
                d = min(0, (parent.count - child_sum) / child_count)
                for value in child_nodes:
                    value.count += d
                    math.floor(value.count)
                    parent_list.append(value)
    
    enhance_consistency(prefix_tree)
    def dfs_restore(node, current_path, trajectories):
        """
        深度优先搜索递归恢复轨迹。
        
        参数：
            node (TrieNode): 当前节点。
            current_path (list): 当前路径。
            trajectories (list): 保存恢复的轨迹数据集。
        """
        # 遍历子节点，优先处理长路径
        total_child_count = 0  # 记录所有子节点的计数总和
        for location_time, child in node.children.items():
            current_path.append(location_time)  # 加入路径
            dfs_restore(child, current_path, trajectories)  # 递归到下一层
            current_path.pop()  # 回溯，移除当前路径节点
            total_child_count += child.count  # 累加子节点计数

        # 检查是否存在以当前节点为终点的轨迹
        missing_count = node.count - total_child_count
        if missing_count > 0:
            for _ in range(missing_count):
                trajectories.append(current_path.copy())  # 添加以当前节点为终点的轨迹
    
    def restore_trajectories(prefix_tree):
        """
        从前缀树中还原轨迹数据集，支持任意层级终点。
        
        返回：
            list of list: 还原的轨迹数据集，每个轨迹是一个位置-时间戳对列表。
        """
        restored_trajectories = []
        dfs_restore(prefix_tree.root, [], restored_trajectories)
        return restored_trajectories
    
    sanitized_trajectories = restore_trajectories(prefix_tree)
    
    return sanitized_trajectories




