class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点
        self.count = 0      # 经过该节点的轨迹数

class TrajectoryPrefixTree:
    def __init__(self):
        self.root = TrieNode()  # 虚拟根节点

    def insert(self, trajectory):
        """
        插入一条轨迹到前缀树中，并更新节点计数。
        
        参数：
            trajectory (list of tuple): 轨迹列表，每个元素是位置-时间戳对，例如 ("A", 1)。
        """
        node = self.root
        for location_time in trajectory:
            if location_time not in node.children:
                node.children[location_time] = TrieNode()
            node = node.children[location_time]
            node.count += 1  # 增加该位置-时间戳节点的计数

    def restore_trajectories(self):
        """
        从前缀树中还原轨迹数据集，支持任意层级终点。
        
        返回：
            list of list: 还原的轨迹数据集，每个轨迹是一个位置-时间戳对列表。
        """
        restored_trajectories = []
        self._dfs_restore(self.root, [], restored_trajectories)
        return restored_trajectories

    def _dfs_restore(self, node, current_path, trajectories):
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
            self._dfs_restore(child, current_path, trajectories)  # 递归到下一层
            current_path.pop()  # 回溯，移除当前路径节点
            total_child_count += child.count  # 累加子节点计数

        # 检查是否存在以当前节点为终点的轨迹
        missing_count = node.count - total_child_count
        if missing_count > 0:
            for _ in range(missing_count):
                trajectories.append(current_path.copy())  # 添加以当前节点为终点的轨迹

    def display(self, node=None, level=0):
        """
        显示前缀树结构，用于查看每个节点的原始值和计数。
        
        参数：
            node (TrieNode): 当前节点。
            level (int): 当前节点的层级，用于打印缩进。
        """
        if node is None:
            node = self.root
        for location_time, child in node.children.items():
            print("  " * level + f"{location_time}({child.count})")
            self.display(child, level + 1)

# 示例使用
# 构建轨迹前缀树
tree = TrajectoryPrefixTree()
tree.insert([("A", 1), ("B", 2)])
tree.insert([("A", 1), ("B", 2), ("C", 3)])
tree.insert([("A", 1), ("B", 2), ("D", 3)])
tree.insert([("A", 1), ("E", 2)])
tree.insert([("A", 1)])

# 还原轨迹数据集
restored_trajectories = tree.restore_trajectories()
print("Restored Trajectories:")
for traj in restored_trajectories:
    print(traj)


