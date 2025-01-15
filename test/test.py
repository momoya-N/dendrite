
import threading
import time

def worker(task_id):
    print(f"Task {task_id} is starting.")
    time.sleep(2)  # Simulate a time-consuming task
    print("happy happy!!"+str(task_id))
    print(f"Task {task_id} is finished.")

threads = []
for i in range(4):  # Create 4 threads
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All tasks are done.")
