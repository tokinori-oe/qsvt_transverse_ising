#initialization
import matplotlib.pyplot as plt
import numpy as np
#np.set_printoptions(threshold=np.inf)
import math
import statistics
from typing import NamedTuple
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.poly import PolyCosineTX, PolySineTX, TargetPolynomial

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.gate import Gate

#import basic plot tools
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

'''
construct unitary gates that are components of Hamiltonian
unitary gates are;
U0 = -S1z * S2z
U1 = -S2z * S3z
U2 = -S3z * S4z
U3 = -S4z * S1z
U4 = S1x
U5 = S2x
U6 = S3x
U7 = S4x 
...

'''

class VarOfSystem(NamedTuple):
    NumOfSite : int
    NumOfSS :int
    NumOfSx :int
    NumOfUnitary :int
    NumOfAncillaForEncoding :int
    NumOfGateForEncoding :int
    ValueOfH: float
    NumOfAncillaForPolynomial: int
    NumOfGate: int

def CheckLessThan2ToTheN(N: int) -> int:
    """2^answerがN以上になる最小のanswerを出力する"""
    answer = 0
    while(N > pow(2, answer)):
        answer += 1
    return answer

def construct_G(var_of_system: VarOfSystem) -> Gate:
    """hamiltonianをencodingするブロックを指定する
    Site数がpow(2,N)の値かそうでないかを場合わけして、それぞれの場合でHの値に対応したoracleを作る
    """
    qc = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    theta_0 = np.arccos(np.sqrt(1 /(var_of_system.ValueOfH + 1))) 
    qc.ry(2 * theta_0, var_of_system.NumOfSite + var_of_system.NumOfAncillaForEncoding - 1)
    for gate in range(var_of_system.NumOfSite, var_of_system.NumOfSite + var_of_system.NumOfAncillaForEncoding - 1):
        qc.h(gate)
    Gstate = qc.to_gate()
    return Gstate

def PlaceOfXgate(decimal: int, var_of_system: VarOfSystem) -> list[int]:
    """Xgateをどこに挿入するかをdecimalに対応して指定する;10進数の値2進数のtupleに変換する
    
    Keyword arguments:
    decimal: 入力する十進数
    
    """
    
    binary_list = []
    result = []
    if decimal == 0:
        binary_list.append(0)
    while decimal > 0:
        binary_list.insert(0, decimal % 2)
        decimal //=2
    
    while len(binary_list) < CheckLessThan2ToTheN(var_of_system.NumOfUnitary):
        binary_list.insert(0,0)
    
    for i, list_i in enumerate(binary_list[::-1]):
        if list_i == 0:
            result.append(i)
    return result

def add_xgate(n: int, gate_list: list, var_of_system: VarOfSystem) -> Gate:
    """指定したゲートにx gateを作用させる関数
    
    Keyword arguments:
    n: 必要な全qubit数
    gate_list: x gateをどのゲートに挿入するかをまとめたリスト
    """
    qc = QuantumCircuit(n)
    #print(gate_list)
    for gate in gate_list:
        qc.x(gate + var_of_system.NumOfSite)
    addingXgate = qc.to_gate()
    return addingXgate

def qc_controlledSS(U_i: int, var_of_system: VarOfSystem) -> Gate:
    """controlled-SS gateを作る
    
    Keyword arguments:
    U_i: 何個目のユニタリーゲート化を指定する数字
    
    Returns:
    controlled-SSのgate
    """
    qc_forU = QuantumCircuit(var_of_system.NumOfSite)
    qc_forU.rz(-3*np.pi, U_i % var_of_system.NumOfSS)
    qc_forU.rz(-3*np.pi, (U_i+1) % var_of_system.NumOfSS)
    SSzgate = qc_forU.to_gate().control(var_of_system.NumOfAncillaForEncoding)
    
    #controlled-SSz gateを作成
    qc_controlled_U = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc_controlled_U.append(add_xgate(var_of_system.NumOfGateForEncoding, PlaceOfXgate(U_i, var_of_system), var_of_system),
                           list(range(var_of_system.NumOfGateForEncoding))) 
    qc_controlled_U.append(SSzgate, list(range(var_of_system.NumOfGateForEncoding))[::-1])
    qc_controlled_U.append(add_xgate(var_of_system.NumOfGateForEncoding, PlaceOfXgate(U_i, var_of_system), var_of_system),
                           list(range(var_of_system.NumOfGateForEncoding)))
    controlled_SS = qc_controlled_U.to_gate()
    
    return controlled_SS

def qc_controlledSx(U_i: int, var_of_system: VarOfSystem) -> Gate:
    """controlled-Sxgateを作る関数
    
    Keyword arguments:
    U_i: U_i: 何個目のユニタリーゲートかを指定する数字
    
    Returns:
    controlled-SSのgate
    """
    #Sxgateを作成
    qc_forSx = QuantumCircuit(var_of_system.NumOfSite)
    qc_forSx.x(U_i % var_of_system.NumOfSx)
    #qc_forSx = QuantumCircuit(var_of_system.NumOfSite)
    Sxgate = qc_forSx.to_gate().control(var_of_system.NumOfAncillaForEncoding)
    
    #controlled-Sx gateを作成
    qc_controlled_U = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc_controlled_U.append(add_xgate(var_of_system.NumOfGateForEncoding, PlaceOfXgate(U_i, var_of_system), var_of_system),
                           list(range(var_of_system.NumOfGateForEncoding)))
    qc_controlled_U.append(Sxgate, list(range(var_of_system.NumOfGateForEncoding))[::-1])
    qc_controlled_U.append(add_xgate(var_of_system.NumOfGateForEncoding, PlaceOfXgate(U_i, var_of_system), var_of_system),
                           list(range(var_of_system.NumOfGateForEncoding)))
    controlled_Sx = qc_controlled_U.to_gate()
    
    return controlled_Sx

def EncodingHamiltonian(var_of_system: VarOfSystem) -> Gate:
    """HamiltonianをUnitary matrixにencodeする
    
    ancillaは下のNumOfAncillaForEncoding個だけある
    """

    HamiltonianEncodedGate = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    #encoding hamiltonian
    #implement an oracle
    HamiltonianEncodedGate.append(construct_G(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
    
    #encoding unitary matrix
    for U_i in range(var_of_system.NumOfSS):
        HamiltonianEncodedGate.append(qc_controlledSS(U_i, var_of_system), 
                        list(range(var_of_system.NumOfGateForEncoding)))
        #print(U_i)
    for U_i in range(pow(2,var_of_system.NumOfAncillaForEncoding - 1) , pow(2, var_of_system.NumOfAncillaForEncoding - 1)  +var_of_system.NumOfSx ):
        HamiltonianEncodedGate.append(qc_controlledSx(U_i, var_of_system), 
                        list(range(var_of_system.NumOfGateForEncoding)))
        
    #transform H to exp(iH)
    
    #implement an oracle
    HamiltonianEncodedGate.append(construct_G(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
    
    H_EncodedGate = HamiltonianEncodedGate.to_gate()
    
    return H_EncodedGate

"---------------------------------------------------------------------------------------------------------------------------------------------------"

def AngListForCos(var_of_system: VarOfSystem, time: float, epsilon: float) -> list[float]:
    """cos(tx)を近似した多項式に対応する角度のリストを求める
    
    Keyword arguments:
    time: cos(tx)のt
    epsilon: cos(tx)と近似した多項式の誤差

    Returns:
    cos(tx)を近似した多項式に対応する角度のリスト
    """
    time *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + 
                    (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary) * ((1 + var_of_system.ValueOfH)/2))
    coef_cos = PolyCosineTX().generate(tau = time, epsilon = epsilon)
    poly = TargetPolynomial(coef_cos)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")
    ang_seq = [ang for sublist in ang_seq for ang in sublist]
    ang_seq = [ang + ang_seq[len(ang_seq) - 1] + (np.pi * (len(ang_seq) - 2)/ 2) if ang_i == 0 else ang - (np.pi/2) for ang, ang_i in enumerate(ang_seq[:-1])]
    
    return ang_seq

def AngListForSine(var_of_system: VarOfSystem, time: float, epsilon: float) -> list[float]:
    """isin(tx)を近似した多項式に対応する角度のリストを求める
    
    Keyword arguments:
    time: isin(tx)のt
    epsilon: isin(tx)と近似した多項式の誤差

    Returns:
    isin(tx)を近似した多項式に対応する角度のリスト
    """
    time *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + 
                    (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary) * ((1 + var_of_system.ValueOfH)/2))
    coef_sin = PolySineTX().generate(tau = time, epsilon = epsilon)
    poly = TargetPolynomial(coef_sin)
    ang_seq = QuantumSignalProcessingPhases(poly, method="tf")
    ang_seq = [ang for sublist in ang_seq for ang in sublist]
    ang_seq = [ang + ang_seq[len(ang_seq) - 1] + (np.pi * (len(ang_seq) - 2)/ 2) if ang_i == 0 else ang - (np.pi/2) for ang, ang_i in enumerate(ang_seq[:-1])]    
    return ang_seq


def projector(var_of_system: VarOfSystem) -> Gate:
    """1番下のancilla以外が左上のブロックを表す状態ならばそのancillaを反転させるようなGateを作る
    
    1番下のqubitがcontrollされるqubit
    """
    NumOfGateForProjector =  var_of_system.NumOfAncillaForEncoding + 1
    qc = QuantumCircuit(NumOfGateForProjector)
    
    for i in range(NumOfGateForProjector - 1):
        qc.x(i)
    qc.mcx(list(range(NumOfGateForProjector - 1)), NumOfGateForProjector - 1)
    for i in range(NumOfGateForProjector - 1):
        qc.x(i)
    
    Projector = qc.to_gate()
    return Projector

def PhaseShiftOperation(var_of_system: VarOfSystem, controlled_state: int, ang: float) -> Gate:
    """projector-controlled phase-shift operationを行う
    
    1番下のqubitがangだけ回すために用意したancilla
    
    Keyword arguments:
    controlled_state: 0-controlledか、1-controlledかを指定する
    
    Returns:
    projector-controlled phase-shift operationのGate
    """
    ang = -ang if controlled_state == 1 else ang
    NumOfGateForPhaseShiftOperation = var_of_system.NumOfGateForEncoding + 1
    qc = QuantumCircuit(NumOfGateForPhaseShiftOperation)
    
    qc.append(projector(var_of_system), list(range(var_of_system.NumOfSite, NumOfGateForPhaseShiftOperation)))
    qc.rz(-2 * ang, NumOfGateForPhaseShiftOperation - 1)
    qc.append(projector(var_of_system), list(range(var_of_system.NumOfSite, NumOfGateForPhaseShiftOperation)))
    
    phaseshiftgate = qc.to_gate()
    
    return phaseshiftgate

def Construct_U_phi(var_of_system: VarOfSystem, controlled_state: int, ang_seq: list[float]) -> Gate:
    """一つのancillaにコントロールされたU_phiのGateを作る
        
    Keyword arguments:
    controlled_state: 0...U_phi, 1...U_-phiを作る
    ans_seq_for_cos: cos(tau*a)の近似多項式をx基底で作るための角度のリスト
        
    Returns:
    一つのancillaにコントロールされたU_(-)phiのGate
    """
    NumOfGateForUphi = var_of_system.NumOfGateForEncoding + 1
    qc = QuantumCircuit(NumOfGateForUphi)
        
    if len(ang_seq) % 2 == 0:
        for ang_i, ang in enumerate(ang_seq[::-1]):
            if ang_i % 2 == 0:
                qc.append(EncodingHamiltonian(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
                qc.append(PhaseShiftOperation(var_of_system, controlled_state, ang), list(range(NumOfGateForUphi)))
            else:
                qc.append(EncodingHamiltonian(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
                qc.append(PhaseShiftOperation(var_of_system, controlled_state, ang), list(range(NumOfGateForUphi)))
    else:
        for ang_i, ang in enumerate(ang_seq[::-1]):
            if ang_i % 2 == 0:
                qc.append(EncodingHamiltonian(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
                qc.append(PhaseShiftOperation(var_of_system, controlled_state, ang), list(range(NumOfGateForUphi)))
            else:
                qc.append(EncodingHamiltonian(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
                qc.append(PhaseShiftOperation(var_of_system, controlled_state, ang), list(range(NumOfGateForUphi)))
    Ugate = qc.to_gate().control(1)
        
    return Ugate  
        

def CosGate(var_of_system: VarOfSystem, ang_seq_for_cos: list[float]) -> Gate:
    """cos(tau*a)の近似多項式に多項式変形する
    
    一番下のqubitが|0><0|, |1><1|を表現するためのancillaで,下から2番目のqubitがangだけ回すために
    用意したancilla.
    
    Keyword arguments:
    ans_seq_for_cos: cos(tau*a)の近似多項式をx基底で作るための角度のリスト
    
    Returns:
    cos(tau*a)の近似多項式を作るGate
    """
    
    NumOfGateForCosGate = var_of_system.NumOfGateForEncoding + 2    
    qc = QuantumCircuit(NumOfGateForCosGate)
    qc.h(NumOfGateForCosGate - 1)
    #|0><0|×U_Φ
    qc.append(Construct_U_phi(var_of_system, 0, ang_seq_for_cos), list(range(NumOfGateForCosGate)))
    #|1><1|×U_-Φ
    qc.x(NumOfGateForCosGate - 1)
    qc.append(Construct_U_phi(var_of_system, 1, ang_seq_for_cos), list(range(NumOfGateForCosGate)))
    qc.x(NumOfGateForCosGate - 1)
    
    qc.h(NumOfGateForCosGate - 1)
    cos_gate = qc.to_gate()
    return cos_gate
    
def ControlledCosGate(var_of_system: VarOfSystem, ang_seq_for_cos: list[float]) -> Gate:
    NumOfGateForCosGate = var_of_system.NumOfGateForEncoding + 2 
    qc = QuantumCircuit(NumOfGateForCosGate)
    qc.append(CosGate(var_of_system, ang_seq_for_cos), list(range(NumOfGateForCosGate)))
    controlled_cos_gate = qc.to_gate().control(1)
    return controlled_cos_gate
    
def SinGate(var_of_system: VarOfSystem, ang_seq_for_sin: list[int]) -> Gate:
    """sin(tau*a)の近似多項式に多項式変形する
    
    一番下のqubitが|0><0|, |1><1|を表現するためのancillaで,下から2番目のqubitがangだけ回すために
    用意したancilla.
    
    Keyword arguments:
    ans_seq_for_cos: sin(tau*a)の近似多項式をx基底で作るための角度のリスト
    
    Returns:
    sin(tau*a)の近似多項式を作るGate
    """ 
    NumOfGateForSinGate = var_of_system.NumOfGateForEncoding + 2    
    qc = QuantumCircuit(NumOfGateForSinGate)
    qc.h(NumOfGateForSinGate - 1)
    #|0><0|×U_Φ
    qc.append(Construct_U_phi(var_of_system, 0, ang_seq_for_sin), list(range(NumOfGateForSinGate)))
    #|1><1|×U_-Φ
    qc.x(NumOfGateForSinGate - 1)
    qc.append(Construct_U_phi(var_of_system, 1, ang_seq_for_sin), list(range(NumOfGateForSinGate)))
    qc.x(NumOfGateForSinGate - 1)
    
    qc.h(NumOfGateForSinGate - 1)
    
    qc.rz(-2 * (-3*np.pi/2), var_of_system.NumOfSite)
    sin_gate = qc.to_gate()
    return sin_gate

def ControlledSinGate(var_of_system: VarOfSystem, ang_seq_for_sin: list[int]) -> Gate:
    NumOfGateForSinGate = var_of_system.NumOfGateForEncoding + 2 
    qc = QuantumCircuit(NumOfGateForSinGate)
    qc.append(SinGate(var_of_system, ang_seq_for_sin), list(range(NumOfGateForSinGate)))
    controlled_sin_gate = qc.to_gate().control(1)
    return controlled_sin_gate

def ExpOverTwoGate(var_of_system: VarOfSystem, ang_seq_for_cos:list[float], ang_seq_for_sin: list[float]) -> Gate:
    """exp(-itH)/2をつくる
    
    NumOfAncillaForPolynomialは5qubitあって、小さいindexから順に、cosGateを作るための2qubit,
    sinGateを作るための2qubit, 線型結合させるための1qubitで並ぶ
    """
    NumOfGateForExpOverTwoGate = var_of_system.NumOfGateForEncoding + 5
    qc = QuantumCircuit(NumOfGateForExpOverTwoGate)
    
    qc.h(NumOfGateForExpOverTwoGate - 1)
    #CosGate
    #CosGateのancillaは3つだから違う,SinGateも同様に違う
    qc.append(ControlledCosGate(var_of_system, ang_seq_for_cos), list(range(var_of_system.NumOfGateForEncoding + 2)) + [NumOfGateForExpOverTwoGate - 1])
    #SinGate(-isinをEncodingする)
    qc.x(NumOfGateForExpOverTwoGate - 1)
    qc.append(ControlledSinGate(var_of_system, ang_seq_for_sin), list(range(var_of_system.NumOfGateForEncoding)) +
              list(range(var_of_system.NumOfGateForEncoding + 2, var_of_system.NumOfGateForEncoding + 4)) + [NumOfGateForExpOverTwoGate - 1])
    qc.x(NumOfGateForExpOverTwoGate - 1)

    qc.h(NumOfGateForExpOverTwoGate - 1)

    exp_over_two_gate = qc.to_gate()
    return exp_over_two_gate

def main():
    #系の設定
    var_of_system = VarOfSystem
    var_of_system.NumOfSite = 3
    var_of_system.ValueOfH = 1.0
    var_of_system.NumOfSS = var_of_system.NumOfSite
    var_of_system.NumOfSx = var_of_system.NumOfSite
    var_of_system.NumOfUnitary = var_of_system.NumOfSS + var_of_system.NumOfSx
    var_of_system.NumOfAncillaForEncoding = CheckLessThan2ToTheN(var_of_system.NumOfUnitary)
    var_of_system.NumOfGateForEncoding = var_of_system.NumOfSite + var_of_system.NumOfAncillaForEncoding
    var_of_system.NumOfAncillaForPolynomial = 5
    var_of_system.NumOfGate = var_of_system.NumOfGateForEncoding + var_of_system.NumOfAncillaForPolynomial
    
    MainGate = QuantumCircuit(var_of_system.NumOfGate)
    
    #Hamiltonianをexp(iHt)に多項式変形する
    start_time = 0.0
    end_time = 5.0
    time_step = 0.1 / 6.0
    #time_list = np.linspace(start_time, end_time, num = int((end_time - start_time) / time_step) + 1) 
    time_list = [1.0]
    epsilon = 0.01
    
    for time in time_list:
        #cosとsinの関数に対応する角度を求める
        ang_seq_for_cos = AngListForCos(var_of_system, time, epsilon)
        ang_seq_for_sin = AngListForSine(var_of_system, time, epsilon)
    
        #exp(-iHt)/2を作る
        MainGate.append(ExpOverTwoGate(var_of_system, ang_seq_for_cos, ang_seq_for_sin), list(range(var_of_system.NumOfGate)))
        
        #exp(-iHt)に増幅させる

        #測定
        statevec_sim = Aer.get_backend('statevector_simulator')
        job = execute(MainGate, statevec_sim)
        result = job.result()
        resultarr = np.array(result.get_statevector(MainGate))
        print(resultarr.shape)
    
if __name__ == '__main__':
    main()