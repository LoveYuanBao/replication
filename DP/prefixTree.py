class PrefixTreeNode:
    def __init__(self, prefix, parent = None, count=0, general=False):
        """
        前缀树节点类的初始化
        
        参数:
        prefix - 该节点的前缀 (例如位置或时间)
        count - 经过该节点的轨迹数量 (默认值为0)
        general - 标记是否为通用节点，默认为False
        """
        self.parent = parent
        self.prefix = prefix  # 该节点表示的前缀信息 (位置或时间)
        self.count = count  # 节点的轨迹计数
        self.general = general  # 节点是否为通用节点
        self.children = {}  # 使用 hashMap (dict) 存储子节点

    def add_child(self, child_node):
        """
        添加子节点，使用节点的前缀作为键
        
        参数:
        child_node - 要添加的前缀树子节点
        """
        self.children[child_node.prefix] = child_node

    def get_child(self, prefix):
        """
        根据前缀获取子节点
        
        参数:
        prefix - 子节点的前缀
        
        返回:
        如果存在该前缀的子节点，返回对应的节点，否则返回None
        """
        return self.children.get(prefix)

    def is_leaf(self):
        """
        判断节点是否为叶子节点 (没有子节点)
        
        返回:
        如果没有子节点，返回True，否则返回False
        """
        return len(self.children) == 0

    def __repr__(self):
        """
        返回节点的字符串表示 (用于调试和打印)
        """
        return f"PrefixTreeNode(prefix={self.prefix}, count={self.count}, general={self.general})"

class PrefixTree:
    def __init__(self):
        """
        前缀树类的初始化
        """
        self.root = PrefixTreeNode(prefix="root", parent = None)  # 初始化根节点

    def insert(self, trajectory):
        """
        在前缀树中插入一条轨迹
        
        参数:
        trajectory - 包含 {loc, time} 对的轨迹列表 (例如 [{"loc": "loc_1", "time": "2024-01-01 08:00"}])
        """
        current_node = self.root
        for entry in trajectory:
            loc = entry['loc']
            time = entry['time']

            # 先插入位置节点
            if current_node.get_child(loc) is None:
                loc_node = PrefixTreeNode(prefix = loc, general = True)
                current_node.add_child(loc_node)
            current_node = current_node.get_child(loc)
            current_node.count += 1

            # 然后插入时间节点
            if current_node.get_child(time) is None:
                time_node = PrefixTreeNode(prefix = time, parent = current_node.prefix, general = False)
                current_node.add_child(time_node)
            current_node = current_node.get_child(time)

            # 更新节点的计数
            current_node.count += 1

    def print_tree(self, node=None, level=0):
        """
        递归打印前缀树的每一层
        
        参数:
        node - 当前节点，默认为根节点
        level - 当前层级，初始值为0
        """
        if node is None:
            node = self.root  # 从根节点开始

        indent = " " * (level * 4)  # 根据当前层级设置缩进
        print(f"{indent}{node.prefix} (count: {node.count})")

        # 递归打印子节点
        for child in node.children.values():
            self.print_tree(child, level + 1)

    def __repr__(self):
        """
        返回前缀树的字符串表示 (用于调试和打印)
        """
        return f"PrefixTree(root={self.root})"

def constructTreeByDataset(dataset):
    prefix_tree = PrefixTree()
    for trajectory in dataset:
        prefix_tree.insert(trajectory)
    return prefix_tree

# if __name__ == '__main__':
#     # 创建前缀树
#     dataset = [[{"loc":"a", "time":1},{"loc":"c", "time":2}],
#            [{"loc":"c", "time":2},{"loc":"b", "time":4}],
#            [{"loc":"a", "time":2},{"loc":"b", "time":3},{"loc":"c", "time":4}],
#            [{"loc":"c", "time":3},{"loc":"a", "time":4}],
#            [{"loc":"a", "time":1},{"loc":"b", "time":2},{"loc":"c", "time":3}],
#            [{"loc":"a", "time":3},{"loc":"c", "time":4}],
#            [{"loc":"a", "time":3},{"loc":"b", "time":4}]]
#     prefix_tree = constructTreeByDataset(dataset)



#     # 输出整个前缀树
#     prefix_tree.print_tree()

