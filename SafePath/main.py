from prefixTree import PrefixTreeNode, PrefixTree, constructTreeByDataset
from taxonomyTree import TaxonomyTreeNode, TaxonomyTree
from collections import deque
from rtree import index
import math
import numpy as np



def laplace_mechanism(value, sensitivity, epsilon):
    """
    拉普拉斯机制：为输入值添加拉普拉斯噪音，保证差分隐私。
    
    参数:
    value - 原始值 (如轨迹计数)
    sensitivity - 灵敏度 (通常为1)
    epsilon - 隐私预算
    
    返回:
    加噪后的值
    """
    scale = sensitivity / epsilon  # 计算噪音的尺度
    noise = np.random.laplace(0, scale)  # 使用拉普拉斯分布生成噪音
    return value + noise  # 返回加噪后的值

def laplace_noise_probability_ge_threshold(epsilon, threshold, sensitivity = 1):
    scale = sensitivity / epsilon
    probability = 0.5 * np.exp(-threshold / scale)
    return probability

def cal_empty_count(epsilon, threshold, total_count):
    p = laplace_noise_probability_ge_threshold(epsilon, threshold)
    number_of_successes = np.random.binomial(1, p, total_count).sum()
    return number_of_successes

def threshold(epsilon):
    """
    阈值函数，用于节点筛选。
    
    参数:
    epsilon - 隐私预算
    
    返回:
    计算出的阈值
    """
    return 4 * math.sqrt(2) / epsilon  # 使用文献中的公式计算阈值


# def cal_trajectorys(taxonomy_node, level_prefix_node_list):
#     taxonomy_node.count = 0
#     for e in taxonomy_node.values:
#         node = level_prefix_node_list.get(e)
#         if node is not None:
#             taxonomy_node.count += node.count
#     return taxonomy_node.count

def loc_cal_trajectorys(r_tree, taxonomy_leaf_list, prefix_node):
    dict = prefix_node.children
    for value in dict.values:
        index = r_tree.intersection(value.prefix)
        taxonomy_leaf_list[index].count += 1

def time_cal_trajectorys(r_tree, taxonomy_leaf_list, prefix_node):
    dict = prefix_node.children
    for value in dict.values:
        index = r_tree.intersection(value.prefix)
        taxonomy_leaf_list[index].count += 1

def traverse_taxonomy(taxonomy_tree):
    stack = [taxonomy_tree.root]
    child_count = []
    result = []
    while stack:
        node = stack.pop()  # 弹出栈顶节点
        result.append(node)  # 访问当前节点
        if len(node.children) != 0:
            child_count.append(len(node.children))
            # 将子节点逆序加入栈中，这样栈中第一个弹出的就是原来的第一个子节点
            for child in reversed(node.children):
                stack.append(child)
    return result, child_count

# result分类树节点数组
def cal_general_node(result, child_count):
    leaf_count = len(child_count)- 1# child_count大小为非叶节点数
    index = len(result) - 1
    # leaf_count<0 分类树只有一个根节点
    # if(leaf_count < 0):
    #     result[0].count = cal_trajectorys(result[0], level_prefix_node_list)
    while leaf_count>= 0:
        sum = 0
        count = child_count[leaf_count]
        j = count
        while j > 0:
            node = result[index]
            sum += node.count
            j -= 1
            index -= 1
        result[index].count = sum
        index -= 1
        leaf_count -= 1

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


def set_zero(result):
    for i in range(len(result)):
        result[i].count = 0

def BuildTaxonomy(taxonomy_tree, epsilon_g, threshold_g, result):
    """
    基于分类树构建通用节点。构建每一层的位置或时间节点，并添加拉普拉斯噪音。

    参数:
    taxonomy_tree - 位置或时间分类树的节点层级
    epsilon_g - 通用节点的隐私预算

    返回:
    general_nodes - 构建的通用节点列表
    """
    general_nodes = []  # 存储通用节点
    if taxonomy_tree.level_count == 0:
        general_nodes.append(taxonomy_tree.root)
        return general_nodes
    
    queue = deque([])
    for node in taxonomy_tree.root.children:
        queue.append(node)
    while queue:
        taxonomy_node = queue.popleft()
        # 计算通用节点的轨迹计数，并添加拉普拉斯噪音
        trajectory_count = laplace_mechanism(taxonomy_node.count, 1, epsilon_g[taxonomy_node.level])
        # 比较计数值是否超过阈值，决定是否保留通用节点
        if trajectory_count >= threshold_g:
            if len(taxonomy_node.children) == 0:
                general_nodes.append(taxonomy_node)  # 保留叶子通用节点
            else:
                for node in taxonomy_node.children:
                    queue.append(node)
    set_zero(result)
    return general_nodes # 返回筛选后的通用节点


def BuildLocSubLevel(level_prefix_node_list, result, child_count, taxonomy_tree, epsilon_g, epsilon_ng, threshold_g, threshold_ng):
    """
    构建位置或时间子层。该函数基于分类树构建节点，并根据拉普拉斯机制筛选节点。

    参数:
    taxonomy_tree - 位置或时间的分类树 (例如 [loc_1, loc_2, ...] 或 [2024, 2025, ...])
    epsilon_g - 分配给通用节点的隐私预算
    epsilon_ng - 分配给非通用节点的隐私预算

    返回:
    sublevel - 经过筛选的子层节点
    """
    sublevel = []  # 用于存储通过筛选的节点（前缀树节点）
    cal_general_node(result, child_count)
    # 构建该层的通用节点
    general_nodes= BuildTaxonomy(taxonomy_tree, epsilon_g, threshold_g, result)
    
    # 对通用节点的非通用节点进行筛选，并应用拉普拉斯机制
    for node in general_nodes:
        # value是前缀树的prefix属性
        for value in node.values:
            temp = level_prefix_node_list.get(value)
            # 计算非通用节点的轨迹计数，加上拉普拉斯噪音
            if temp is None:
                trajectory_count = laplace_mechanism(0, 1, epsilon_ng)
            else:
                trajectory_count = laplace_mechanism(temp.count, 1, epsilon_ng)
        
            # 比较计数值是否超过阈值，决定是否保留节点
            if trajectory_count >= threshold_ng:
                if temp is None:
                    temp = PrefixTreeNode(prefix = value, count = trajectory_count)
                else:
                    temp.count = trajectory_count
                sublevel.append(temp)  # 将节点加入子层
    
    return sublevel # 返回筛选后的子层

def BuildTimeSubLevel(level_prefix_node_list, result, child_count, taxonomy_tree, epsilon_g, epsilon_ng, threshold_g, threshold_ng, time_threshold):
    """
    构建位置或时间子层。该函数基于分类树构建节点，并根据拉普拉斯机制筛选节点。

    参数:
    taxonomy_tree - 位置或时间的分类树 (例如 [loc_1, loc_2, ...] 或 [2024, 2025, ...])
    epsilon_g - 分配给通用节点的隐私预算
    epsilon_ng - 分配给非通用节点的隐私预算

    返回:
    sublevel - 经过筛选的子层节点
    """
    sublevel = []  # 用于存储通过筛选的节点（前缀树节点）
    cal_general_node(result, child_count)
    # 构建该层的通用节点
    general_nodes= BuildTaxonomy(taxonomy_tree, epsilon_g, threshold_g, result)
    
    # 对通用节点的非通用节点进行筛选，并应用拉普拉斯机制
    for node in general_nodes:
        if node.values[1] < time_threshold:
            continue
        start = node.values[0]
        if node.values[0] < time_threshold and node.values[1] > time_threshold:
            start = time_threshold
        # value是前缀树的prefix属性
        for value in range(start,node.values[1]+1):
            temp = level_prefix_node_list.get(value)
            # 计算非通用节点的轨迹计数，加上拉普拉斯噪音
            if temp is None:
                trajectory_count = laplace_mechanism(0, 1, epsilon_ng)
            else:
                trajectory_count = laplace_mechanism(temp.count, 1, epsilon_ng)
        
            # 比较计数值是否超过阈值，决定是否保留节点
            if trajectory_count >= threshold_ng:
                if temp is None:
                    temp = PrefixTreeNode(prefix = value, count = trajectory_count)
                else:
                    temp.count = trajectory_count
                sublevel.append(temp)  # 将节点加入子层
    
    return sublevel # 返回筛选后的子层

def create_loc_rtree(location_taxonomy, loc_rtree):
    pass

def create_time_rtree(time_taxonomy, time_rtree):
    pass


def SafePath(dataset, epsilon, location_taxonomy, time_taxonomy, height):
    """
    SafePath算法的主函数，用于生成差分隐私的轨迹数据集。

    参数:
    dataset - 原始轨迹数据集 (例如 [[{loc, time}, {loc, time}], ...])
    epsilon - 差分隐私的总预算
    location_taxonomy - 位置的分类树 (例如 [loc_1, loc_2, ...])
    time_taxonomy - 时间的分类树 (例如 [2024, 2025, ...])
    height - 噪音前缀树的高度 (层数)

    返回:
    D_hat - 经过差分隐私处理的轨迹数据集
    """
    # 初始化根节点，即噪音前缀树的根
    prefix_tree = constructTreeByDataset(dataset)

    # 将总隐私预算epsilon划分给每一层，假设每一层都均匀分配
    epsilon_s = epsilon / (2*height)  # 每一个子层的隐私预算
    loc_epsilon_u = 2*epsilon_s / len(location_taxonomy.root.values)
    time_epsilon_u = 2*epsilon_s / len(time_taxonomy.root.values)
    loc_epsilon_g = [0]
    loc_epsilon_ng = epsilon_s
    for i in range(0, location_taxonomy.level_count):
        loc_epsilon_g.append(loc_epsilon_u * (i+1))
        loc_epsilon_ng -= loc_epsilon_g[i+1]

    time_epsilon_g = [0]
    time_epsilon_ng = epsilon_s
    for i in range(0, time_taxonomy.level_count):
        time_epsilon_g.append(time_epsilon_u * (i+1))
        time_epsilon_ng -= time_epsilon_g[i+1]

    # 计算阈值
    threshold_g = threshold(epsilon / height)
    threshold_ng = threshold_g / 2

    loc_result, loc_child_count = traverse_taxonomy(location_taxonomy)
    time_result, time_child_count = traverse_taxonomy(location_taxonomy)

    # 清理前缀树前打印
    prefix_tree.print_tree()
    # 构建rtree
    loc_rtree = index.Index()
    time_rtree = index.Index()
    create_loc_rtree(location_taxonomy, loc_rtree)
    create_time_rtree(time_taxonomy, time_rtree)

    parent_list = deque([])
    parent_list.append(prefix_tree.root)
    parent_count = 1
    # 遍历每一层并构建噪音前缀树
    for _ in range(1, height + 1):
        parent_count = len(parent_list)
        # list为位置节点列表,字典类型
        while parent_count>0:
            parent_node = parent_list.popleft()
            list = parent_node.children
            parent_count -= 1
            non_general_list = {}
            # 处理位置树
            # general_list为通过阈值的通用节点列表(如位置通用节点a,b,c)
            general_list = BuildLocSubLevel(list, loc_result, loc_child_count, location_taxonomy, loc_epsilon_g, loc_epsilon_ng, threshold_g = 5, threshold_ng = 5)
            # node为每一个叶通用节点
            for node in general_list:
                # 处理时间树
                # temp为通过阈值的非通用节点列表
                temp = BuildTimeSubLevel(node.children, time_result, time_child_count, time_taxonomy,time_epsilon_g, time_epsilon_ng, threshold_g, threshold_ng, parent_node.parent)
                for loc_time in temp:
                    if loc_time.parent is None:
                        loc_time.parent = node.prefix
                    time = loc_time.prefix
                    loc_time.prefix = (loc_time.parent, loc_time.prefix)
                    loc_time.parent = time
                    non_general_list[loc_time.prefix] = loc_time
                    parent_list.append(loc_time)
            # 将筛选后的非通用节点挂载在上一层节点
            parent_node.children = non_general_list

    # 生成差分隐私的轨迹数据集
    # D_hat = generate_sanitized_trajectories(prefix_tree)
    # return D_hat
    # 输出整个前缀树
    prefix_tree.print_tree()

# if __name__ == '__main__':
#     dataset = [[{"loc":"a", "time":1},{"loc":"c", "time":2}],
#            [{"loc":"c", "time":2},{"loc":"b", "time":4}],
#            [{"loc":"a", "time":2},{"loc":"b", "time":3},{"loc":"c", "time":4}],
#            [{"loc":"c", "time":3},{"loc":"a", "time":4}],
#            [{"loc":"a", "time":1},{"loc":"b", "time":2},{"loc":"c", "time":3}],
#            [{"loc":"a", "time":3},{"loc":"c", "time":4}],
#            [{"loc":"a", "time":3},{"loc":"b", "time":4}]]
#     location_taxonomy = TaxonomyTree(["a", "b", "c"])
#     time_taxonomy = TaxonomyTree([1,2,3,4])
#     time_taxonomy.root.add_child(TaxonomyTreeNode([1,2],level = 1))
#     time_taxonomy.root.add_child(TaxonomyTreeNode([3,4], level = 1))
#     time_taxonomy.level_count += 1
#     SafePath(dataset, 1, location_taxonomy, time_taxonomy, 3)