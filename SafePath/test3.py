import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def random_point_in_box(min_lat, max_lat, min_lon, max_lon):
    return random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon)

def create_trajectory(lat, lon, current_time, avg_length, max_length, interval, speed_range):
    trajectory = []
    length = min(max_length, max(avg_length, np.random.poisson(avg_length)))
    speed = random.uniform(*speed_range)
    angle = random.uniform(0, 360)
    for point in range(length):
        if point > 0:
            speed += random.uniform(-5, 5)
            angle += random.uniform(-30, 30)
            speed = max(min(speed, speed_range[1]), speed_range[0])
            distance = (speed / 3600) * interval
            rad_angle = np.deg2rad(angle)
            delta_lat = np.cos(rad_angle) * distance / 111
            delta_lon = np.sin(rad_angle) * distance / (111 * np.cos(np.deg2rad(lat)))
            lat += delta_lat
            lon += delta_lon
        time = current_time + timedelta(minutes=point * interval)
        trajectory.append((lat, lon, time))
    return trajectory

def generate_smooth_trajectories(n, box, peak_times, avg_length=5, max_length=20, interval=6, speed_range=(20, 60)):
    trajectories = []
    hourly_counts = [0] * 24  # 初始化一个长度为24的列表，用于计数每小时的轨迹数量
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    time_interval_per_trajectory = 24 * 60 / n

    for i in range(n):
        current_time = base_time + timedelta(minutes=i * time_interval_per_trajectory)
        hour = current_time.hour

        if 2 <= hour < 6:
            if random.random() >= 0.1:
                continue
        elif 8 <= hour < 10 or 18 <= hour < 22:
            trajectories_to_add = 2
            for _ in range(trajectories_to_add):
                lat, lon = random_point_in_box(box[0][0], box[1][0], box[0][1], box[1][1])
                trajectory = create_trajectory(lat, lon, current_time, avg_length, max_length, interval, speed_range)
                trajectories.append((trajectory, current_time))
                hourly_counts[hour] += 1

        lat, lon = random_point_in_box(box[0][0], box[1][0], box[0][1], box[1][1])
        trajectory = create_trajectory(lat, lon, current_time, avg_length, max_length, interval, speed_range)
        trajectories.append((trajectory, current_time))
        hourly_counts[hour] += 1

    return trajectories, hourly_counts

def plot_trajectories_between_times(trajectories, start_time, end_time):
    plt.figure(figsize=(10, 8))
    for trajectory, time in trajectories:
        if start_time <= time.hour < end_time:
            lats, lons, _ = zip(*trajectory)
            plt.plot(lons, lats, marker='o', linestyle='-', alpha=0.6)  # 设置轨迹透明度以便于观看重叠部分

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Trajectories between {start_time}:00 and {end_time}:00')
    plt.grid(True)
    plt.show()

# 配置和生成数据
bounding_box = [(37.7749, -122.4194), (37.8049, -122.3894)]
peak_hours = {
    'morning_start': 8, 'morning_end': 10,
    'evening_start': 18, 'evening_end': 22,
    'night_start': 2, 'night_end': 6
}


import csv

def write_trajectories_to_csv(trajectories, filename="trajectories.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Trajectory_ID', 'Latitude', 'Longitude', 'Timestamp'])  # 添加了Trajectory_ID
        trajectory_id = 1  # 初始化轨迹ID计数器
        for trajectory, _ in trajectories:
            for lat, lon, time in trajectory:
                writer.writerow([trajectory_id, lat, lon, time.isoformat()])
            trajectory_id += 1  # 每完成一条轨迹的写入，增加轨迹ID


trajectories, hourly_trajectory_counts = generate_smooth_trajectories(1000, bounding_box, peak_hours)

# write_trajectories_to_csv(trajectories)

# 打印每小时的轨迹数量
for hour in range(24):
    print(f"Hour {hour}: {hourly_trajectory_counts[hour]} trajectories")

# 可视化8点到9点的轨迹
plot_trajectories_between_times(trajectories, 8, 9)


















