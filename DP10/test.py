from concurrent.futures import ThreadPoolExecutor
import hashlib

def hash_trajectory(traj):
    """
    将轨迹转换为哈希值字符串，以便进行快速比较。
    """
    return hashlib.md5(str(traj).encode()).hexdigest()

def is_sub_trajectory_hash(sub_traj, traj):
    """
    使用哈希值和滑动窗口检查子轨迹是否在给定轨迹中。
    """
    sub_hash = hash_trajectory(sub_traj)
    len_sub = len(sub_traj)

    for i in range(len(traj) - len_sub + 1):
        if hash_trajectory(traj[i:i + len_sub]) == sub_hash:
            return True
    return False

def count_query(dataset, sub_traj):
    """
    并行化计算包含特定子轨迹的轨迹数量。
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda traj: is_sub_trajectory_hash(sub_traj, traj), dataset))
    return sum(results)

# 示例数据集（模拟大型数据集）
trajectory_dataset = [
    [(1, 'A'), (2, 'B'), (3, 'C')],
    [(1, 'A'), (2, 'B'), (3, 'D')],
    [(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D')],
    # 生成更多轨迹以模拟大数据集...
] * 100000  # 假设重复生成 100,000 条记录

# 子轨迹查询
sub_trajectory = [(2, 'B'), (3, 'C')]

# 计算结果
result = count_query(trajectory_dataset, sub_trajectory)
print(f"包含子轨迹的轨迹数量: {result}")
