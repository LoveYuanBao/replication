class PrefixTreeNode:
    def __init__(self, timestamp, location, count=0):
        """
        前缀树节点初始化
        :param timestamp: 时间戳，表示在该节点的时间
        :param location: 位置，表示该节点的地点
        :param count: 计数器，表示该子轨迹的出现次数
        """
        self.loc_time = (timestamp, location)
        self.count = count
        self.children = {}  # 存储子节点，键为(timestamp, location)对，值为 PrefixTreeNode 对象

    # def increment_count(self):
    #     """
    #     增加节点的计数
    #     """
    #     self.count += 1

    # def add_child(self, timestamp, location):
    #     """
    #     添加子节点
    #     :param timestamp: 子节点的时间戳
    #     :param location: 子节点的位置
    #     :return: 新增的子节点对象
    #     """
    #     if (timestamp, location) not in self.children:
    #         self.children[(timestamp, location)] = PrefixTreeNode(timestamp, location)
    #     return self.children[(timestamp, location)]


class PrefixTree:
    def __init__(self, o = 0, k = 1.5, b = 1):
        """
        前缀树初始化
        """
        self.root = PrefixTreeNode(None, None)  # 根节点的时间戳和位置为空，表示起始节点
        self.o = o
        self.k = k
        self.b = b

    # def add_trajectory(self, trajectory):
    #     """
    #     将一个轨迹添加到前缀树中
    #     :param trajectory: 轨迹列表，格式为 [(timestamp, location), ...]
    #     """
    #     current_node = self.root
    #     for timestamp, location in trajectory:
    #         current_node = current_node.add_child(timestamp, location)
    #         current_node.increment_count()

    # def get_subtree(self, node=None):
    #     """
    #     获取以指定节点为根的子树，用于调试或展示
    #     :param node: 起始节点，默认为根节点
    #     :return: 子树的结构（字典形式）
    #     """
    #     if node is None:
    #         node = self.root

    #     subtree = {'count': node.count, 'children': {}}
    #     for (timestamp, location), child in node.children.items():
    #         subtree['children'][(timestamp, location)] = self.get_subtree(child)
        
    #     return subtree
