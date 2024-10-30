import multiprocessing
import time

def test_process(number):
    print(f"Process {number} is running")
    time.sleep(1)
    print(f"Process {number} has finished")

if __name__ == '__main__':
    print("Starting multiprocessing test...")
    processes = []
    for i in range(5):  # 创建5个进程
        process = multiprocessing.Process(target=test_process, args=(i,))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    print("All processes have completed.")
