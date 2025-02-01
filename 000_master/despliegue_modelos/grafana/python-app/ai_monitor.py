from flask import Flask
import random
import time
import logging
from prometheus_client import start_http_server, Gauge

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Configuración de métricas de Prometheus
cpu_usage = Gauge('ai_model_cpu_usage', 'CPU Usage of AI Model')
memory_usage = Gauge('ai_model_memory_usage', 'Memory Usage of AI Model')
disk_usage = Gauge('ai_model_disk_usage', 'Disk Usage of AI Model')
network_usage = Gauge('ai_model_network_usage', 'Network Usage of AI Model')

app = Flask(__name__)

@app.route("/")
def health_check():
    return "AI Model Monitoring Running", 200

def simulate_metrics():
    """Simula métricas de uso de recursos"""
    while True:
        cpu = random.uniform(10, 90)
        mem = random.uniform(500, 3000)  # MB
        disk = random.uniform(10, 100)  # GB
        net = random.uniform(0.1, 10)  # Mbps

        cpu_usage.set(cpu)
        memory_usage.set(mem)
        disk_usage.set(disk)
        network_usage.set(net)

        logger.info(f"Inferencia realizada: CPU={cpu}%, MEM={mem}MB, DISK={disk}GB, NET={net}Mbps")
        time.sleep(2)

if __name__ == "__main__":
    start_http_server(8000)
    simulate_metrics()

