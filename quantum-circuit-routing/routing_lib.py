import re
from dataclasses import dataclass, field
from typing import Any, List, Dict
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph, DiGraph, floyd_warshall_numpy
from IPython.display import display

from qiskit import qasm3,transpile,qasm2
from qiskit.transpiler import CouplingMap

# Define symbols to simplify writing
H,X,Y,Z,S,T,M,U,R = 'HXYZSTMUR'
CNOT,CPHASE,CZ,CX,TOFFOLI,SWAP,NOP = 'CNOT','CPHASE','CZ','CX','TOFFOLI','SWAP','NOP'

c_black = (0,0,0)
c_white = (255, 255, 255)
c_yellow = (255, 255, 0)
c_lightyellow = (255,222,179)
c_gold = (255, 215, 0)
c_gray = (100, 100, 100)
c_gray_light = (150, 150, 150)
c_green = (0, 255, 0)
c_red = (255, 0, 0)
c_blue = (0, 0, 255)
c_lightblue = (0, 245, 255)
c_lightgreen = (193, 255, 193)
c_pink = (255, 105, 180)


class CN_Qubits:
    def __init__(self, qtype: int = 0):
        self.set_type(qtype)
        
    def set_type(self, qtype: int):
        """Configure qubit parameters based on type."""
        self.qtype = qtype
        if qtype == 0:
            # superconducting
            self.single_gate_t = 30         # ns
            self.double_gate_t = 60         # ns
            self.single_gate_fd = 0.9998    # 99.98%
            self.double_gate_fd = 0.9995    # 99.95%
            self.t1 = 300e3                 # ns
            self.t2 = 200e3                 # ns
        elif qtype == 1:
            # neutral atom
            self.single_gate_t = 1000       # ns
            self.double_gate_t = 400        # ns
            self.single_gate_fd = 0.999     # 99.9%
            self.double_gate_fd = 0.997     # 99.7%
            self.t1 = 7e9                   # ns  # https://www.nature.com/articles/s41586-022-04592-6/figures/7
            self.t2 = 1e9                   # ns
        else:
            raise ValueError(f"qtype out of range: {qtype}")
        
    def change(self):
        self.set_type(1 - self.qtype)
        return

@dataclass
class Gate:
    number: int = 0                     # 门编号
    gate_type: str = U                  # 门类别
    parameters: Any = None              # 门参数
    num_qubits: int = 1                 # 门比特数
    phys_qubits: List[int] = field(default_factory=list)        # 作用的物理比特
    logi_qubits: List[int] = field(default_factory=list)        # 作用的逻辑比特
    qubits_type: List[CN_Qubits] = field(default_factory=list)  # 作用的比特类型
    trans:bool = False                  # 是否跨体系
    goto_mem: int = 0                   # 传去哪个memory （传回/非mem门 = 0）
    start_time: int = 0                 # 门开始的绝对时间（ns）
    gate_time: int = 0                  # 门运行时间（ns）
    idle_aft: List[int] = field(default_factory=list)           # 门运行后idle（ns）

class Q_circuit:
    def __init__(self,num = 0, trans_fd = 0.997, trans_time = 90):
        self.source_file = None
        self.gate_list: List[Gate] = []
        self.logical_n = num
        self.physical_n = num
        self.max_time = 0
        # self._logi_time: List[int] = []
        # self.logi_idle_list = [[] for i in range(self.logical_n)]
        # self.phys_idle_list = [[] for i in range(self.physical_n)]
        self.mapping: List[int] = None
        self.cs_coupling_graph = None
        self.cs_D = None
        self.trans_gate_time = trans_time
        self.trans_gate_fd = trans_fd
        self.max_idle = self._get_idle()
        self.logi_fed = [1 for i in range(num)]
        self.mem_port = [0 for i in range(num)] # 当前物理比特指向哪个
        self.mem = None
        self.dag = None
        self.online = [0 for i in range(num)] # logical qubits 是否在中性原子比特上、在哪个原子接口上。
        self.cs_empty = None # 超导比特 是否 空
        self.swap_coupling_graph = None
        self.swap_D = None
        self.dq_list = []
        self.logi_fed_list = None
        self.phys_fed_list = None
        self.gate_layer_list = []
        self.layers= []
        # self.qq_layers = []
        self.last_logi_gates = [ 0 for i in range(num)]
        self.set_mem()
        
    def reset_with_qubit_num(self,num):
        self.__init__(num,self.trans_gate_fd,self.trans_gate_time)

    def _get_idle(self):
        na = CN_Qubits(1)
        cs = CN_Qubits(0)
        t_start = 2*self.trans_gate_time
        t = t_start
        fed1 = np.exp(-t/(cs.t1))
        fed2 = (np.exp(-(t - self.trans_gate_time)/(na.t1))*
                np.exp(-self.trans_gate_time/(cs.t1))*
                (self.trans_gate_fd**2))
        while fed1 > fed2:
            t+=cs.single_gate_t
            fed1 = np.exp(-t/(cs.t1))
            fed2 = (np.exp(-(t - self.trans_gate_time)/(na.t1))*
                    np.exp(-self.trans_gate_time/(cs.t1))*
                    (self.trans_gate_fd**2))
        return t
        
    def set_mem(self,port_num = 1):
        size = self.physical_n // port_num
        
        # mem_port 从1开始
        for i in range(self.physical_n):
            self.mem_port[i] = (i // size)+1
           
    def load_qasm(self, qasm_file: str = "output.qasm", with_memory: bool = False):
        self.source_file = qasm_file
        # 读取所有行
        with open(qasm_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]

        # 提取 qubit 声明，构建 name->物理索引 映射
        logi_q_list: Dict[str,int] = {}
        for ln in lines:
            m = re.match(r"qubit\[(\d+)\]\s+([A-Za-z_0-9]+);", ln)
            if m:
                count, base = int(m.group(1)), m.group(2)
                self.logical_n = count
                for i in range(count):
                    logi_q_list[f"{base}[{i}]"] = len(logi_q_list)
        # 初始化逻辑映射（identity）
        self.mapping = list(range(min(self.logical_n, self.physical_n))) + [-1] * max(0, self.physical_n - self.logical_n)
        self.cs_empty = np.array([0 if i< self.logical_n else 1 for i in range(self.physical_n)])
        # 生成 Gate 
        self.gate_list = []
        measure_count = 0
        for idx, ln in enumerate(lines):
            # U(theta,phi,lambda) q[i];
            m = re.match(r"U\(([^)]+)\)\s+([A-Za-z_0-9\[\]]+);", ln)
            if m:
                params = m.group(1)
                lq = logi_q_list[m.group(2)]
                pq = self.mapping[lq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="U",
                    parameters=params,
                    num_qubits = 1,
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                continue

            # cz q[i],q[j];
            m = re.match(r"cz\s+([A-Za-z_0-9\[\]]+),\s*([A-Za-z_0-9\[\]]+);", ln)
            if m:
                l0 = logi_q_list[m.group(1)]
                l1 = logi_q_list[m.group(2)]
                p0 = self.mapping[l0]
                p1 = self.mapping[l1]
                cnq1 = CN_Qubits()
                cnq2 = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="CZ",
                    parameters=[],
                    num_qubits = 2,
                    trans=False,
                    phys_qubits=[p0, p1],
                    logi_qubits=[l0, l1],
                    qubits_type=[cnq1,cnq2],
                    gate_time=cnq1.double_gate_t
                )
                self.gate_list.append(gate)
                continue

            # swap q[i],q[j];
            m = re.match(r"swap\s+([A-Za-z_0-9\[\]]+),\s*([A-Za-z_0-9\[\]]+);", ln)
            if m:
                l0 = logi_q_list[m.group(1)]
                l1 = logi_q_list[m.group(2)]
                self.mapping[l0],self.mapping[l1] = self.mapping[l1],self.mapping[l0]
                p0 = self.mapping[l0]
                p1 = self.mapping[l1]
                cnq1 = CN_Qubits()
                cnq2 = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="SWAP",
                    parameters=[],
                    num_qubits = 2,
                    trans=False,
                    phys_qubits=[p0, p1],
                    logi_qubits=[l0, l1],
                    qubits_type=[cnq1,cnq2],
                    gate_time=cnq1.double_gate_t*3
                )
                self.gate_list.append(gate)
                continue

            # measure c = measure q[i];
            m = re.match(r"([A-Za-z_0-9\[\]]+)\s*=\s*measure\s+([A-Za-z_0-9\[\]]+);", ln)
            if m:
                lq = logi_q_list[m.group(2)]
                pq = self.mapping[lq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="M",
                    parameters=[int(measure_count)],
                    num_qubits = 1,
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                measure_count += 1
                continue

            # reset q[i];
            m = re.match(r"reset\s+([A-Za-z_0-9\[\]]+);", ln)
            if m:
                lq = logi_q_list[m.group(1)]
                pq = self.mapping[lq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="R",
                    num_qubits = 1,
                    parameters=[],
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                continue
            
        return
    
    def load_qasm_qiskit(self, qasm_file: str = "output.qasm", with_memory: bool = False):
        # 读取所有行
        with open(qasm_file, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]

        # qiskit 的结果已经是物理比特了，我们需要反追踪
        re_mapping = [i for i in range(self.physical_n)]
        
        # 生成 Gate 
        self.gate_list = []
        measure_count = 0
        for idx, ln in enumerate(lines):
            # U(theta,phi,lambda) q[i];
            m = re.match(r"U\(([^)]+)\)\s+\$(\d+);", ln)
            if m:
                params = m.group(1)
                pq = int(m.group(2))
                lq = re_mapping[pq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="U",
                    parameters=params,
                    num_qubits = 1,
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                continue

            # cz q[i],q[j];
            m = re.match(r"cz\s+\$(\d+),\s*\$(\d+);", ln)
            if m:
                p0 = int(m.group(1))
                p1 = int(m.group(2))
                l0 = re_mapping[p0]
                l1 = re_mapping[p1]
                cnq1 = CN_Qubits()
                cnq2 = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="CZ",
                    parameters=[],
                    num_qubits = 2,
                    trans=False,
                    phys_qubits=[p0, p1],
                    logi_qubits=[l0, l1],
                    qubits_type=[cnq1,cnq2],
                    gate_time=cnq1.double_gate_t
                )
                self.gate_list.append(gate)
                continue

            # swap q[i],q[j];
            m = re.match(r"swap\s+\$(\d+),\s*\$(\d+);", ln)
            if m:
                p0 = int(m.group(1))
                p1 = int(m.group(2))
                re_mapping[p0],re_mapping[p1] = re_mapping[p1],re_mapping[p0]
                l0 = re_mapping[p0]
                l1 = re_mapping[p1]
                cnq1 = CN_Qubits()
                cnq2 = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="SWAP",
                    parameters=[],
                    num_qubits = 2,
                    trans=False,
                    phys_qubits=[p0, p1],
                    logi_qubits=[l0, l1],
                    qubits_type=[cnq1,cnq2],
                    gate_time=cnq1.double_gate_t*3
                )
                self.gate_list.append(gate)
                continue

            # measure c = measure q[i];
            m = re.match(r"([a-zA-Z_0-9\[\]]+)\s*=\s*measure\s+\$(\d+);", ln)
            if m:
                pq = int(m.group(2))
                lq = re_mapping[pq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="M",
                    parameters=[int(measure_count)],
                    num_qubits = 1,
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                measure_count += 1
                continue

            # reset q[i];
            m = re.match(r"reset\s+\$(\d+);", ln)
            if m:
                pq = int(m.group(1))
                lq = re_mapping[pq]
                cnq = CN_Qubits()
                gate = Gate(
                    number=len(self.gate_list),
                    gate_type="R",
                    num_qubits = 1,
                    parameters=[],
                    trans=False,
                    phys_qubits=[pq],
                    logi_qubits=[lq],
                    qubits_type=[cnq],
                    gate_time=cnq.single_gate_t
                )
                self.gate_list.append(gate)
                continue
        
        # for pq,lq in enumerate(re_mapping):
        #     self.mapping[lq] = pq
        
        return
    
    def schedule_gates(self,get_layer = 1) -> None:
        # 从0开始
        num = self.physical_n
        last_gates = [-1 for i in range(num)]
        last_logi_gates = [-1 for i in range(num)]
        phys_last_time = [ 0 for i in range(num)]
        last_layer = [-1 for i in range(num)]
        gate_layer_list = [0 for i in range(len(self.gate_list))]
        edges = []
        for gate in self.gate_list:
            gate.idle_aft = [] #初始化
            # 最晚可用时间
            gate_start = max(phys_last_time[q] for q in gate.phys_qubits)
            # 开始和结束的绝对时间
            gate.start_time = gate_start
            finish = gate_start + gate.gate_time   
            # 更新相关比特时间轴
            for q,lq in zip(gate.phys_qubits,gate.logi_qubits):
                # idle时间初始化为0
                gate.idle_aft.append(0)
                # 上一个门到这个门之间有多长idle
                idle = gate_start - phys_last_time[q]
                # 如果有上一个门：
                if last_gates[q] != -1:
                    # 找到上一个门
                    last_gate = self.gate_list[last_gates[q]]
                    edges.append((last_gate.number,gate.number))
                    # 更新对应门qubit的idle
                    for i,last_pq in enumerate(last_gate.phys_qubits):
                        if last_pq == q:
                            last_gate.idle_aft[i] = idle
                # 将时间轴和门轴更新
                phys_last_time[q] = finish   
                last_gates[q] = gate.number
                last_logi_gates[lq] = gate.number
        if get_layer: 
            for gate in self.gate_list:
                # 当前门层 (-1 -> 0)
                current_layer = max(*(last_layer[q] for q in gate.logi_qubits),0)
                # 如果是双比特门
                if len(gate.logi_qubits) == 2:
                    current_layer = max(*(last_layer[q]+1 for q in gate.logi_qubits),0)
                    last_layer[gate.logi_qubits[0]]=current_layer
                    last_layer[gate.logi_qubits[1]]=current_layer
                    self.dq_list.append(gate.number)
                gate_layer_list[gate.number] = current_layer

        self.max_time = max(phys_last_time)
        self.last_logi_gates = last_logi_gates
        return

    def schedule_gates_aft_routing(self,get_layer = 1) -> None:
        gate_list = self.gate_list
        # 上传/下载门，上传门mem = mem_port，下载门mem = 0; 
        # back_time = 0表示当前时间为start, = 1表示为end
        def _mem(lq,pq,start_time,mem = 0, back_time = 0, idle = 0):
            return Gate(
                        number=len(gate_list),
                        gate_type="mem",
                        parameters=[],
                        num_qubits = 1,
                        trans=True,
                        goto_mem = mem,
                        phys_qubits=[pq],
                        logi_qubits=[lq],
                        start_time = start_time - back_time*self.trans_gate_time,
                        gate_time=self.trans_gate_time,
                        idle_aft = [idle]
                    )
        
        # 从0开始
        num = self.physical_n
        last_gates = [-1 for i in range(num)]
        phys_last_time = [ 0 for i in range(num)]
        for gate in gate_list[:]:
            if gate.goto_mem == 0:
                end_time = gate.start_time + gate.gate_time
                for i,lq in enumerate(gate.logi_qubits):
                    if lq != -1:
                        # print(i)
                        # print(gate.logi_qubits)
                        # print(gate.idle_aft)
                        # print(gate)
                        idle = gate.idle_aft[i]
                        pq = gate.phys_qubits[i]
                        mem_port = self.mem_port[pq]
                        if idle >= self.max_idle:
                            print("do mem")
                            new_idle = idle - 2* self.trans_gate_time
                            gate.idle_aft[i] = 0
                            self.gate_list.append(_mem(lq,pq,end_time,mem=mem_port,idle = new_idle))
                            new_end = end_time + idle - self.trans_gate_time
                            self.gate_list.append(_mem(lq,pq,new_end))
        
        self.gate_list = gate_list
                # 将时间轴和门轴更新
                # phys_last_time[pq] = finish   
                # last_gates[pq] = gate.number
            # if gate.gate_type == "mem" and gate.logi_qubits[0] == 11:
            #     print(gate)
        # if get_layer: 
        #     for gate in self.gate_list:
        #         # 当前门层 (-1 -> 0)
        #         current_layer = max(*(last_layer[q] for q in gate.logi_qubits),0)
        #         # 如果是双比特门
        #         if len(gate.logi_qubits) == 2:
        #             current_layer = max(*(last_layer[q]+1 for q in gate.logi_qubits),0)
        #             last_layer[gate.logi_qubits[0]]=current_layer
        #             last_layer[gate.logi_qubits[1]]=current_layer
        #             self.dq_list.append(gate.number)
        #         gate_layer_list[gate.number] = current_layer

        #     # 生产layer 数组
        #     layers = [[] for _ in range(max(gate_layer_list)+1)]
        #     for idx, val in enumerate(gate_layer_list):
        #         layers[val].append(idx)
            
        #     self.gate_layer_list = gate_layer_list
        #     self.layers = layers
        #     self._get_gate_dag(edges)

        # self.max_time = max(phys_last_time)
        return

    def blocks(self, rows: int, cols: int):
        """
        生成一个 rows × cols 的网格状拓扑
        """
        # Create graph and node list
        coupling_graph = nx.Graph()
        num_nodes = rows * cols
        self.reset_with_qubit_num(num_nodes)
        coupling_graph.add_nodes_from(range(num_nodes))

        # Build edges between neighbors
        edges = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                # horizontal neighbor (right)
                if c < cols - 1:
                    edges.append((idx, r * cols + (c + 1)))
                # vertical neighbor (down)
                if r < rows - 1:
                    edges.append((idx, (r + 1) * cols + c))

        coupling_graph.add_edges_from(edges)

        # Store in self
        self.cs_coupling_graph = coupling_graph
        self.cs_D = np.array(floyd_warshall_numpy(coupling_graph))
        self.physical_n = num_nodes
        self.set_mem()
        return

    # 用于绘制电路概览,ex_in是缩放
    def plot_circuit(self, ex_in = 5):
        
        def _plot_range(pixels, width, num, start, color,length=1):
            x_start = int(start[0]*ex_in)
            y_start = int(start[1]*ex_in*2)
            for x in range(x_start, x_start + ex_in*length):
                for y in range(y_start, y_start + ex_in*2):
                    pixels[x%width, 2*(num+1)*ex_in*(x//width)+y] = color 
        
        def _plot_gate(pixels, width, num, gate:Gate, s_t, color):
            x0 = int(gate.start_time // s_t * ex_in)
            y0 = int(gate.phys_qubits[0] * ex_in * 2)
            length = int(gate.gate_time // s_t)
            for x in range(x0, x0 + ex_in * length):
                for y in range(y0, y0 + ex_in * 2):
                    pixels[x % width, 2*(num+1)*ex_in*(x//width) + y] = color
            if len(gate.phys_qubits) > 1:
                y1 = gate.phys_qubits[1] * ex_in * 2
                for x in range(x0, x0 + ex_in * length):
                    for y in range(y1, y1 + ex_in * 2):
                        pixels[x % width, 2*(num+1)*ex_in*(x//width) + y] = color

        # gate data
        CS = CN_Qubits()
        s_t = CS.single_gate_t # 单位时间
        num   = self.physical_n
        D     = self.cs_D
        total_step = int(self.max_time//s_t)
        # image size
        width = total_step * ex_in
        height = (num + 1) * ex_in * 2
        times = 1
        if width > 2000:
            times = math.ceil(width / 2000)
            width = 2000
            height *= times
        image = Image.new('RGB', (width, height), c_white)  # 背景白色
        pixels = image.load()
        for g in self.gate_list: 
            color = c_green
            if g.gate_type == CZ:
                dist = D[g.phys_qubits[0]][g.phys_qubits[1]]
                if dist == 1:
                    color = c_blue 
                else:
                    color = c_red
                    
            elif g.gate_type == SWAP:
                color = c_gray
            elif g.gate_type == "mem":
                color = c_pink
            _plot_gate(pixels, width, num, g, s_t, color)
        
        exh = ex_in*2
        for i in range(times):
            for x in range(0, width):
                for y in range(((i+1)*(num+1)-1)*exh, ((i+1)*(num+1))*exh):
                    pixels[x, y] = (0,0,0)
        
        display(image)
        return image

    def compute_fed(self):
        l_cs_time = [0 for i in range(self.logical_n)]
        l_cs_single = [0 for i in range(self.logical_n)]
        l_cs_double = [0 for i in range(self.logical_n)]        
        l_na_time = [0 for i in range(self.logical_n)]
        
        # p_cs_time = [0 for i in range(self.logical_n)]
        # p_cs_single = [0 for i in range(self.logical_n)]
        # p_cs_double = [0 for i in range(self.logical_n)]
        # p_na_time = [0 for i in range(self.logical_n)]  
        
        l_trans = [0 for i in range(self.logical_n)]
        # p_trans = [0 for i in range(self.logical_n)]
        
        l_gates = [0 for i in range(self.logical_n)]
        # p_gates = [0 for i in range(self.logical_n)]
        
        l_fed_list = []
        # p_fed_list = []
        l_worst_list = []
        l_best_list = []
        l_nontrans_list = []
        
        l_last_cs_time = [0 for i in range(self.logical_n)]
        
        for gate in self.gate_list:
            if gate.trans:
                # 对于mem门
                lq = gate.logi_qubits[0]
                l_trans[lq] += 1
                l_na_time[lq] += gate.gate_time//2
                l_cs_time[lq] += gate.gate_time//2
                if gate.goto_mem !=0:
                    l_last_cs_time[lq] = gate.start_time + gate.gate_time
                else:
                    l_na_time[lq] += gate.start_time - l_last_cs_time[lq]
                    l_cs_time[lq] += gate.idle_aft[0]
            elif gate.num_qubits == 1:
                # 对于单比特门
                lq = gate.logi_qubits[0]
                l_cs_time[lq] += gate.gate_time + gate.idle_aft[0]
                l_cs_single[lq] += 1
            else:
                # 对于双比特门
                for lq ,idle in zip(gate.logi_qubits,gate.idle_aft):
                    if lq !=-1:
                        l_cs_time[lq] += gate.gate_time + idle
                        if gate.gate_type == SWAP:
                            l_cs_double[lq] += 3
                        elif gate.gate_type == "SWAP_T":
                            l_cs_double[lq] += 2
                        else:
                            l_cs_double[lq] += 1
        
        cs = CN_Qubits(0)
        na = CN_Qubits(1)
        for i in range(self.logical_n):
            l_fed = (np.exp(-(l_cs_time[i])/(cs.t1))*
                     np.exp(-l_na_time[i]/(na.t1))*
                     (cs.single_gate_fd**l_cs_single[i])*
                     (cs.double_gate_fd**l_cs_double[i])*
                     (self.trans_gate_fd**l_trans[i]))
            l_fed_list.append(l_fed)
        
        self.fed_logi_list = np.array(l_fed_list)
        self.fed_logi_ave = np.average(l_fed_list)
        return
    
    # 使用qiskit开源库mapping作为示例
    def mpping_by_qiskit(self,output_file = "qiskit.qasm"):
        coupling = CouplingMap(self.cs_coupling_graph.edges)
        qc = qasm3.load(self.source_file)
        mapped_qc = transpile(
                qc,
                basis_gates=["u","cz","swap"],
                coupling_map=coupling,
                routing_method='sabre' ,
                layout_method='sabre',
                optimization_level=1,
                seed_transpiler=42
            )
        qasm_string = qasm3.dumps(mapped_qc)
        with open(output_file, "w") as f:
            f.write(qasm_string)
        self.load_qasm_qiskit(qasm_file = output_file)
        self.schedule_gates()
        
    def load_test(self, testcase = 1):
        match testcase:
            case 1:
                l,w = 2,2
                use_file = "./test_circuits/small_test_1.qasm"
            case 2:
                l,w = 1,5
                use_file = "./test_circuits/small_test_2.qasm"
            case 3:
                l,w = 2,3
                use_file = "./test_circuits/small_test_3.qasm"
            case 4:
                l,w = 3,6
                use_file = "./test_circuits/medium_test_1.qasm"
            case 5:
                l,w = 3,5
                use_file = "./test_circuits/medium_test_2.qasm"
            case 6:
                l,w = 4,4
                use_file = "./test_circuits/medium_test_3.qasm"
            case _:
                print(f"test case {testcase} 不存在")
        
        self.blocks(l,w)
        self.load_qasm(qasm_file = use_file)
        self.schedule_gates()
        return
    
    '''
    任务目标-----------------------------------------
    '''
    def routing(self):
        exe_gate_list = []
        
        # 你要完成的部分---------------------
        # 这里只是复制了原list，啥都没干
        for gate in self.gate_list:
            exe_gate_list.append(gate)
        # ----------------------------------
        
        self.gate_list = exe_gate_list
        self.schedule_gates()
        return