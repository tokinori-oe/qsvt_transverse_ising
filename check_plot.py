import numpy as np
import scipy
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.poly import PolyCosineTX, PolySineTX, TargetPolynomial

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy
from qiskit.circuit.gate import Gate

import matplotlib.pyplot as plt

def AngListForCosInQSP(time: float, epsilon: float) -> list[float]:
    """QSPでcosの近似多項式を作るための角度のリストを作る"""
    coef_cos = PolyCosineTX().generate(tau = time, epsilon = epsilon)
    poly = TargetPolynomial(coef_cos)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")
    ang_seq = [ang for sublist in ang_seq for ang in sublist]
    
    return ang_seq

def AngListForSineInQSP(time: float, epsilon: float) -> list[float]:
    """QSPでsinの近似多項式を作るための角度のリストを作る"""
    coef_sin = PolySineTX().generate(tau = time, epsilon = epsilon)
    poly = TargetPolynomial(coef_sin)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")
    ang_seq = [ang for sublist in ang_seq for ang in sublist]
    
    return ang_seq

def TransformIntoCosOfChebyshev(time: float, epsilon: float, x_list: list[float]) -> list[float]:
    """Cosをチェビシェフ級数展開した多項式の値のリストを出力する
    
    Keyword arguments:
    time: cos(tx)のt
    epsilon:cos(tx)を多項式近似するときの精度
    x_list:xのlist
    
    Returns:
    cos(tx)の近似多項式の値のリスト
    """
    r = scipy.optimize.fsolve(lambda r: (
        np.e * np.abs(time) / (2 * r))**r - (5 / 4) * epsilon, time)[0]
    R = np.floor(r / 2).astype(int)
    R = max(R, 1)
    answer_list = []
    for x in x_list:
        TransformedValue = scipy.special.jv(0, time) * np.polynomial.chebyshev.chebval(x, [1])
        for k in range(1, R+1):
            TransformedValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k, time) * \
                                    np.polynomial.chebyshev.chebval(x, [0] * (2 * k) + [1]))
        answer_list.append(TransformedValue)
    return answer_list

def TransformIntoSineOfChebyshev(time: float, epsilon: float, x_list: list[float]) -> list[float]:
    """Sineをチェビシェフ級数展開した多項式の値のリストを出力する
    
    Keyword arguments:
    time: sin(tx)のt
    epsilon: sin(tx)を多項式近似する時の精度
    x_list: xのlist
    
    Returns:
    sin(tx)の近似多項式の値のリスト
    """
    r = scipy.optimize.fsolve(lambda r: (
        np.e * np.abs(time) / (2 * r))**r - (5 / 4) * epsilon, time)[0]
    R = np.floor(r / 2).astype(int)
    R = max(R, 1)
    TransformedValue = 0
    answer_list = []
    for x in x_list:
        for k in range(1, R):
            TransformedValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k + 1, time) * \
                                    np.polynomial.chebyshev.chebval(x, [0] * (2 * k + 1) + [1]))
        answer_list.append(TransformedValue)
    return answer_list

def QSPGateForCos(time: float, epsilon: float, theta: float) -> Gate:
    """AngListForCosから出力した角度を用いたQSPのx回転とz回転の交互配置のGateを作る
    
    Keyword arguments:
    time: cos(tx)のt
    epsilon: cos(tx)と近似した多項式の誤差
    theta: QSPのx回転の回す角度
    
    """
    
    qc = QuantumCircuit(1)
    ang_list = AngListForCosInQSP(time, epsilon)
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
    ang_list = AngListForSineInQSP(time , epsilon)
    for ang in ang_list[:-1]:
        qc.rz(-2 * ang, 0)
        qc.rx(theta, 0)
    qc.rz(-2*ang_list[len(ang_list)-1],0)
    
    QSPGateForSine = qc.to_gate()
    return QSPGateForSine


def main():
    time = 10.
    epsilon = 0.01
    #a_list = np.linspace(0,1.0,10)
    a_list = [0.5, -0.5]
    theta_list = [-2 * np.arccos(element) for element in a_list]
        
    qsp_result =([])
    for theta in theta_list:    
        main_qc = QuantumCircuit(1)
        
        main_qc.h(0)
        main_qc.append(QSPGateForCos(time, epsilon, theta), [0])
        #main_qc.append(QSPGateForSine(time, epsilon, theta), [0])
        main_qc.h(0)
    
        #計測
        statevec_sim = Aer.get_backend('statevector_simulator')
        job = execute(main_qc, statevec_sim)
        result = job.result()
        qsp_result.append(result.get_statevector(main_qc)[0])
    
    #プロット
    chebyshev_value_list = TransformIntoCosOfChebyshev(time, epsilon, a_list)
    plt.plot(a_list, qsp_result, "o")
    plt.plot(a_list, chebyshev_value_list, "o")
    print(qsp_result)
    print(chebyshev_value_list)
    print([abs(value1 - value2) for value1, value2 in zip(chebyshev_value_list, qsp_result)])
    plt.show()


if __name__ == '__main__':
    main()