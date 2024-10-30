from prefixTree import PrefixTreeNode, PrefixTree, constructTreeByDataset
from taxonomyTree import TaxonomyTreeNode, TaxonomyTree
from collections import deque

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

def threshold(epsilon):
    """
    阈值函数，用于节点筛选。
    
    参数:
    epsilon - 隐私预算
    
    返回:
    计算出的阈值
    """
    return 4 * math.sqrt(2) / epsilon  # 使用文献中的公式计算阈值


def cal_trajectorys(taxonomy_node, level_prefix_node_list):
    taxonomy_node.count = 0
    for e in taxonomy_node.values:
        node = level_prefix_node_list.get(e)
        if node is not None:
            taxonomy_node.count += node.count
    return taxonomy_node.count

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
def cal_general_node(result, child_count, level_prefix_node_list):
    leaf_count = len(child_count)- 1# child_count大小为非叶节点数
    index = len(result) - 1
    # leaf_count<0 分类树只有一个根节点
    if(leaf_count < 0):
        result[0].count = cal_trajectorys(result[0], level_prefix_node_list)
    while leaf_count>= 0:
        sum = 0
        count = child_count[leaf_count]
        j = count
        while j > 0:
            node = result[index]
            sum += cal_trajectorys(node, level_prefix_node_list)
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
    
    def traverse_tree(node, current_trajectory):
        """
        递归遍历前缀树的节点，并生成轨迹。
        
        参数:
        node - 当前节点
        current_trajectory - 当前轨迹，表示从根节点到该节点的路径
        """
        # 如果该节点的计数大于 0，说明此节点被保留
        if node.count > 0:
            sanitized_trajectories.append(list(current_trajectory))  # 复制当前轨迹并保存
            
        # 遍历子节点，继续生成轨迹
        for (location, timestamp), child in node.children.items():
            # 添加位置和时间到当前轨迹
            current_trajectory.append({'loc': location, 'time': timestamp})
            
            # 递归遍历子节点
            traverse_tree(child, current_trajectory)
            
            # 回溯，移除最后一个节点
            current_trajectory.pop()
    
    # 从根节点开始递归生成轨迹
    traverse_tree(prefix_tree, [])
    
    return sanitized_trajectories


def BuildTaxonomy(taxonomy_tree, epsilon_g, threshold_g):
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
    return general_nodes # 返回筛选后的通用节点


def BuildSubLevel(level_prefix_node_list, result, child_count, taxonomy_tree, epsilon_g, epsilon_ng, threshold_g, threshold_ng):
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
    cal_general_node(result, child_count, level_prefix_node_list)
    # 构建该层的通用节点
    general_nodes= BuildTaxonomy(taxonomy_tree, epsilon_g, threshold_g)
    
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
            # general_list为通过阈值的通用节点列表(如位置通用节点a,b,c)
            general_list = BuildSubLevel(list, loc_result, loc_child_count, location_taxonomy, loc_epsilon_g, loc_epsilon_ng, threshold_g = 5, threshold_ng = 5)
            # node为每一个叶通用节点
            for node in general_list:
                # temp为通过阈值的非通用节点列表
                temp = BuildSubLevel(node.children, time_result, time_child_count, time_taxonomy,time_epsilon_g, time_epsilon_ng, threshold_g, threshold_ng)
                for loc_time in temp:
                    if loc_time.parent is None:
                        loc_time.parent = node.prefix
                    loc_time.prefix = f"{loc_time.parent}-{loc_time.prefix}"
                    non_general_list[loc_time.prefix] = loc_time
                    parent_list.append(loc_time)
            # 将筛选后的非通用节点挂载在上一层节点
            parent_node.children = non_general_list

        
                
    # 生成差分隐私的轨迹数据集
    # D_hat = generate_sanitized_trajectories(prefix_tree)
    # return D_hat
    # 输出整个前缀树
    prefix_tree.print_tree()

if __name__ == '__main__':
    dataset = [[{"loc":"a", "time":1},{"loc":"c", "time":2}],
           [{"loc":"c", "time":2},{"loc":"b", "time":4}],
           [{"loc":"a", "time":2},{"loc":"b", "time":3},{"loc":"c", "time":4}],
           [{"loc":"c", "time":3},{"loc":"a", "time":4}],
           [{"loc":"a", "time":1},{"loc":"b", "time":2},{"loc":"c", "time":3}],
           [{"loc":"a", "time":3},{"loc":"c", "time":4}],
           [{"loc":"a", "time":3},{"loc":"b", "time":4}]]
    location_taxonomy = TaxonomyTree(["a", "b", "c"])
    time_taxonomy = TaxonomyTree([1,2,3,4])
    time_taxonomy.root.add_child(TaxonomyTreeNode([1,2],level = 1))
    time_taxonomy.root.add_child(TaxonomyTreeNode([3,4], level = 1))
    time_taxonomy.level_count += 1
    SafePath(dataset, 1, location_taxonomy, time_taxonomy, 3)