import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class PolicyChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and not event.src_path.endswith('~'):
            logging.info(f"检测到文件变动: {event.src_path}，将自动更新知识库...")
            subprocess.run(["python", "update_knowledge_base.py"])
            logging.info("知识库更新完毕。守护进程继续监控中...")

if __name__ == "__main__":
    path = "./policy_source"
    logging.info("首次启动，执行初始知识库构建...")
    subprocess.run(["python", "update_knowledge_base.py"])
    
    logging.info(f"🚀 [守护进程启动] 正在监控文件夹: {path}")
    event_handler = PolicyChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()