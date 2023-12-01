from transverse_ising import AngListForCos, AngListForSine
import numpy as np

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.circuit.gate import Gate

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def QSPGateForCos(time: float, epsilon: float, theta: float) -> Gate:
    """AngListForCosから出力した角度を用いたQSPのx回転とz回転の交互配置のGateを作る
    
    Keyword arguments:
    time: cos(tx)のt
    epsilon: cos(tx)と近似した多項式の誤差
    theta: QSPのx回転の回す角度
    
    """
    
    qc = QuantumCircuit(1)
    ang_list = AngListForCos(time, epsilon)
    #print(ang_list)
    for ang in ang_list[:-1]:
        qc.rz(-2 * ang, 0)
        qc.rx(theta, 0)
    qc.rz(-2*ang_list[len(ang_list)-1], 0)
    
    QSPGateForCos = qc.to_gate()
    return QSPGateForCos
        
def QSPGateForSine(time: float, epsilon: float, theta: float)-> Gate:
    """AngListForISineから出力した角度を用いたQSPのx回転とz回転の交互配置のGateを作る
    
    Keyword arguments:
    time: isin(tx)のt
    epsilon: isin(tx)と近似した多項式の誤差
    theta: QSPのx回転の回す角度
    
    """
    
    qc = QuantumCircuit(1)
    ang_list = AngListForSine(time , epsilon)
    for ang in ang_list[:-1]:
        qc.rz(-2 * ang, 0)
        qc.rx(theta, 0)
    qc.rz(-2*ang_list[len(ang_list)-1],0)
    
    QSPGateForSine = qc.to_gate()
    return QSPGateForSine


def main():
    time = 10.
    epsilon = 0.1
    a_list = np.linspace(0,0.5,10)
    theta_list = [-2 * np.arccos(element) for element in a_list]
    
    finalresult =([])
    for theta in theta_list:    
        main_qc = QuantumCircuit(1)
    
        main_qc.h(0)
        #main_qc.append(QSPGateForCos(time, epsilon, theta), [0])
        main_qc.append(QSPGateForSine(time, epsilon, theta), [0])
        main_qc.h(0)
    
        #計測
        statevec_sim = Aer.get_backend('statevector_simulator')
        job = execute(main_qc, statevec_sim)
        result = job.result()
        finalresult.append(result.get_statevector(main_qc)[0])
    
    #プロット
    plt.plot(a_list, finalresult, "o")
    plt.show()


if __name__ == '__main__':
    main()