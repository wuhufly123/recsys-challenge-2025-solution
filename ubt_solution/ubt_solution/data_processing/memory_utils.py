import os
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

def report_memory_usage(step_name: str) -> float:
    """报告当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # 转换为MB
    rss_mb = memory_info.rss / 1024 / 1024
    vms_mb = memory_info.vms / 1024 / 1024
    
    logger.info(f"内存使用情况 [{step_name}] - RSS: {rss_mb:.2f} MB, VMS: {vms_mb:.2f} MB")
    
    # 尝试主动触发垃圾回收
    collected = gc.collect()
    logger.debug(f"垃圾回收：释放了 {collected} 个对象")
    
    return rss_mb 