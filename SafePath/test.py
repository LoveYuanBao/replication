
a = [1,2,3]

def fname(arr):
    arr[0] = 9

# fname(a)

# print(a[0])

class node(object):
    """docstring for node."""
    def __init__(self, arr):
        super(node, self).__init__()
        self.children = arr
    
b = node(a)
# print(b.children[0])
c = [8,8,8]
b.children = c
# print(b.children[0])
c[0] = 9
# print(b.children[0])
    
d = {"name": "Lily", "age": 18}

b.children = d
# print(b.children.get("home") == None)


import random
import time
from multiprocessing import Pool

class TreeNode:
    def __init__(self, value=0):
        self.value = value
        self.children = []

    def add_child(self, child):
        if len(self.children) < 10:
            self.children.append(child)

    def remove_child(self):
        if self.children:
            self.children.pop(random.randint(0, len(self.children) - 1))

class Tree:
    def __init__(self, root=None):
        self.root = root

    def generate_large_tree(self, depth, max_depth, max_children=10, max_value=9):
        if depth == 0 or depth >= max_depth:
            return None
        self.root = TreeNode(random.randint(0, max_value))
        queue = [(self.root, 1)]
        while queue:
            current, cur_depth = queue.pop(0)
            if cur_depth < depth:
                num_children = random.randint(0, min(max_children, max_depth - cur_depth))
                for _ in range(num_children):
                    child = TreeNode(random.randint(0, max_value))
                    current.add_child(child)
                    queue.append((child, cur_depth + 1))

def random_modify_node(args):
    node, depth, max_depth = args
    num_modifications = random.randint(1, 5)
    for _ in range(num_modifications):
        if random.random() > 0.5:
            if len(node.children) < 10 and depth + 1 < max_depth:
                new_value = random.randint(0, 9)
                node.add_child(TreeNode(new_value))
        elif node.children:
            node.remove_child()
    return node

def modify_tree_single_thread(tree, max_depth):
    def modify(node, depth):
        random_modify_node((node, depth, max_depth))
        for child in node.children:
            modify(child, depth + 1)
    modify(tree.root, 1)

def modify_tree_multi_process(tree, max_processes=4, max_depth=10):
    def process_nodes(nodes, depth):
        with Pool(processes=max_processes) as pool:
            pool.map(random_modify_node, [(node, depth, max_depth) for node in nodes])

    queue = [(tree.root, 1)]
    while queue:
        next_queue = []
        for node, depth in queue:
            if node.children:
                next_queue.extend([(child, depth + 1) for child in node.children])
            if depth < max_depth:
                process_nodes([node], depth)
        queue = next_queue

if __name__ == '__main__':
    # 更大的树和更深的修改
    depth = 7
    max_depth = 10
    max_children = 10
    max_value = 9
    tree = Tree()
    tree.generate_large_tree(depth, max_depth, max_children, max_value)

    # 单线程修改
    start_time = time.time()
    modify_tree_single_thread(tree, max_depth)
    end_time = time.time()
    print("Single-threaded modification time:", end_time - start_time)

    # 多进程修改，进程数与CPU核心数匹配
    # import multiprocessing
    # num_cores = multiprocessing.cpu_count()
    start_time = time.time()
    modify_tree_multi_process(tree, max_processes= 8, max_depth=max_depth)
    end_time = time.time()
    print("Multi-threaded modification time:", end_time - start_time)










