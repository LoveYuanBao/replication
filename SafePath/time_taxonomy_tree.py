class TimeTaxonomyTreeNode:
    def __init__(self, values, level = 0,count = 0):
        """
        分类树节点类，表示位置列表或时间列表。
        
        参数:
        values - 节点的属性数组 (例如 [loc1, loc2] 或 [2024, 2025])
        level - 节点在树中的层级，根节点为0
        """
        self.values = values  # 时间列表,总时间段长度暂时定为为28小时，时间段格式：元组(开始时刻时间戳，结束时刻时间戳)，单位暂时定为分钟
        self.level = level  # 节点在树中的层级
        self.count = count # 轨迹计数
        self.children = []  # 存储子节点的列表

    def add_child(self, child_node):
        """
        添加子节点
        
        参数:
        child_node - 要添加的子节点
        """
        self.children.append(child_node)

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
        return f"TaxonomyTreeNode(values={self.values}, level={self.level})"


class TimeTaxonomyTree:
    def __init__(self, root_values, height, child_count):
        """
        分类树类的初始化
        
        参数:
        root_values - 根节点的属性数组 (例如 ["root"] 或 ["2024", "2025"])
        """
        self.root = TimeTaxonomyTreeNode(values=root_values)  # 创建根节点
        self.level_count = 0
        self.height = height
        self.child_count = child_count


    def print_tree(self, node=None, level=0):
        """
        递归打印分类树的每一层
        
        参数:
        node - 当前节点，默认为根节点
        level - 当前层级，初始值为0
        """
        if node is None:
            node = self.root  # 从根节点开始

        indent = " " * (level * 4)  # 根据当前层级设置缩进
        print(f"{indent}{node.values} (level: {node.level})")

        # 递归打印子节点
        for child in node.children:
            self.print_tree(child, level + 1)

    def __repr__(self):
        """
        返回分类树的字符串表示 (用于调试和打印)
        """
        return f"TaxonomyTree(root={self.root})"