import pytest
from transverse_ising import VarOfSystem, CheckLessThan2ToTheN, EncodingHamiltonian, qc_controlledSS, qc_controlledSx, construct_G
from transverse_ising import CosGate, SinGate, AngListForCos, AngListForSine
import numpy as np
import numpy.linalg as LA
import scipy.special

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


def setting_var_of_system(NumOfSite: int, ValueOfH: float) -> VarOfSystem:
    """系の変数を与える"""
    var_of_system = VarOfSystem
    var_of_system.NumOfSite = NumOfSite
    var_of_system.ValueOfH = ValueOfH
    var_of_system.NumOfSS = var_of_system.NumOfSite
    var_of_system.NumOfSx = var_of_system.NumOfSite
    var_of_system.NumOfUnitary = var_of_system.NumOfSS + var_of_system.NumOfSx
    var_of_system.NumOfAncillaForEncoding = CheckLessThan2ToTheN(var_of_system.NumOfUnitary)
    var_of_system.NumOfGateForEncoding = var_of_system.NumOfSite + var_of_system.NumOfAncillaForEncoding
    var_of_system.NumOfAncillaForPolynomial = 5
    var_of_system.NumOfGate = var_of_system.NumOfGateForEncoding + var_of_system.NumOfAncillaForPolynomial
        
    return var_of_system

def Zgate() -> np.ndarray:
    """Zgateの行列を返す"""
    return np.array([[1,0], [0, -1]])

def Igate() -> np.ndarray:
    """Identity Matrixを返す"""
    return np.array([[1,0], [0,1]])

def Xgate() -> np.ndarray:
    """Xgateの行列を返す"""
    return np.array([[0,1],[1,0]])

def SSz_matrix(x: int, var_of_system: VarOfSystem) -> np.ndarray:
    """S_z * S_zのmatrixを返す"""
    if x != var_of_system.NumOfSite -1:
        matrix_list = [Zgate() if i == x or i == x+1 else Igate() for i in range(var_of_system.NumOfSite)]
    else:
        matrix_list = [Zgate() if i == 0 or i == x else Igate() for i in range(var_of_system.NumOfSite)] 
        
    SS = np.array([-1])
    for matrix in matrix_list:
        SS = np.kron(SS, matrix)
    return SS

def Sx_matrix(x: int, var_of_system: VarOfSystem) -> np.ndarray:
    """S_xのmatrixを返す"""
    Sx = np.array([1])
    for i in range(var_of_system.NumOfSite):
        if i ==x:
            Sx = np.kron(Sx, Xgate())
        else:
            Sx = np.kron(Sx, Igate())
    return Sx

def Identity_matrixFortest(var_of_system: VarOfSystem) -> np.ndarray:
    """系のハミルトニアンと同じサイズのIdentity matrixを返す"""
    S = np.array([1])
    for i in range(var_of_system.NumOfSite):
        S = np.kron(S, Igate())
    return S

def AnswerMatrix(var_of_system: VarOfSystem) -> np.ndarray:
    """encodingされたhamiltonianのanswerを出力する"""
    answer = sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = answer + var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    #answer += sum(Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSite))
    return answer

@pytest.mark.parametrize(
    "NumOfSite, ValueOfH",
    [
        (pow(2,1), 1.0),
        (pow(2,2), 1.0),
        (pow(2,3), 1.0),
        (pow(2,1), 2.0),
        (pow(2,2), 2.0),
        (pow(2,3), 2.0),
        (pow(2,1), 3.0),
        (pow(2,2), 3.0),
        (pow(2,3), 3.0),
        (5, 1.0),
        (6, 1.0),
        (7, 1.0),
        (5, 2.0),
        (6, 2.0),
        (7, 2.0),
        (5, 3.0),
        (6, 3.0),
        (7, 3.0),
    ],
)
def test_Encoding(NumOfSite, ValueOfH):
    """ある磁場でsite数がNのときにハミルトニアンが正しくエンコーディングされているかテストする"""
    #このテスト関数をパラメタライズする
    #Given: Hamiltonian の変数を与える
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    #When   hamiltonianをencodingする
    qc = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc.append(EncodingHamiltonian(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
    #測定d
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))

    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    encoded_matrix *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + 
                    (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary) * ((1 + var_of_system.ValueOfH)/2))
        
    #Then   比較する,  answerとencoded_matrixの差がIdentity_matrixの倍だったらTrueになるようにする
    answer = sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = answer + var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    matrix_diff = encoded_matrix - answer
    if np.allclose(encoded_matrix, answer):
        assert True
    else:
        matrix_diff_normalized = matrix_diff / matrix_diff[0][0]
        identitiy_matrix = np.eye(matrix_diff.shape[0])
        #assert np.allclose(encoded_matrix, encoded_matrix.T)
        assert np.allclose(matrix_diff_normalized, identitiy_matrix)

@pytest.mark.parametrize(
    "NumOfSite, ValueOfH",
    [
        (pow(2,1), 1.0),
        (pow(2,2), 1.0),
        (pow(2,3), 1.0),
        (pow(2,1), 2.0),
        (pow(2,2), 2.0),
        (pow(2,3), 2.0),
        (pow(2,1), 3.0),
        (pow(2,2), 3.0),
        (3, 1.0),
        (5, 1.0),
        (6, 1.0),
        (7, 1.0),
        (3, 2.0),
        (5, 2.0),
        (6, 2.0),
        (7, 2.0),
    ],
)
def test_controlledSS(NumOfSite, ValueOfH):
    """controlledSSをテストする"""
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    qc = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc.append(construct_G(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
    for U_i in range(var_of_system.NumOfSS):
        qc.append(qc_controlledSS(U_i, var_of_system),
                list(range(var_of_system.NumOfGateForEncoding)))
    qc.append(construct_G(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
    #測定
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))

    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    encoded_matrix *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + 
                    (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary) * ((1 + var_of_system.ValueOfH)/2))
    answer =  sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSS))
    answer = answer +  var_of_system.ValueOfH * sum(Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSx))
    #この比較方法が磁場が1.0より大きいときに正しくない
    matrix_diff = encoded_matrix - answer
    print(encoded_matrix)
    print(answer)
    if np.allclose(encoded_matrix, answer):
        assert True
    else:
        matrix_diff_normalized = matrix_diff / matrix_diff[0][0]
        identitiy_matrix = np.eye(matrix_diff.shape[0])
        #assert np.allclose(encoded_matrix, encoded_matrix.T)
        assert np.allclose(matrix_diff_normalized, identitiy_matrix)


@pytest.mark.parametrize(
    "NumOfSite, ValueOfH",
    [
        (pow(2,1), 1.0),
        (pow(2,2), 1.0),
        (pow(2,3), 1.0),
        (pow(2,1), 2.0),
        (pow(2,2), 2.0),
        (pow(2,3), 2.0),
        (pow(2,1), 3.0),
        (pow(2,2), 3.0),
        (3, 1.0),
        (5, 1.0),
        (6, 1.0),
        (7, 1.0),
        (3, 2.0),
        (5, 2.0),
        (6, 2.0),
        (7, 2.0),
    ],
)
def test_controlledSx(NumOfSite, ValueOfH):
    """controlledSxをテストする"""
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    qc = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc.append(construct_G(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
    #print(var_of_system.NumOfGateForEncoding)
    for U_i in range(pow(2,var_of_system.NumOfAncillaForEncoding - 1) , pow(2, var_of_system.NumOfAncillaForEncoding - 1)  +var_of_system.NumOfSx ):
        qc.append(qc_controlledSx(U_i, var_of_system),
                list(range(var_of_system.NumOfGateForEncoding)))
    qc.append(construct_G(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
    #測定
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))

    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    encoded_matrix *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + 
                    (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary) * ((1 + var_of_system.ValueOfH)/2))
    answer = var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSx))
    answer += sum(Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSS))
    print(encoded_matrix)
    print(answer)
    #この比較方法が磁場が1.0より大きいときに正しくない
    matrix_diff = encoded_matrix - answer
    print(encoded_matrix)
    print(answer)
    if np.allclose(encoded_matrix, answer):
        assert True
    else:
        matrix_diff_normalized = matrix_diff / matrix_diff[0][0]
        identitiy_matrix = np.eye(matrix_diff.shape[0])
        #assert np.allclose(encoded_matrix, encoded_matrix.T)
        assert np.allclose(matrix_diff_normalized, identitiy_matrix)
        
def test_Identity():
    NumOfSite = 2
    ValueOfH = 1.0
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    qc = QuantumCircuit(var_of_system.NumOfGateForEncoding)
    qc.append(construct_G(var_of_system), list(range(var_of_system.NumOfGateForEncoding)))
    qc.append(construct_G(var_of_system).inverse(), list(range(var_of_system.NumOfGateForEncoding)))
    #測定
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))

    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    encoded_matrix *= (var_of_system.NumOfSite * (1 + var_of_system.ValueOfH) + (pow(2, var_of_system.NumOfAncillaForEncoding) - var_of_system.NumOfUnitary))
    answer = sum(Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSite * 2))
    assert np.allclose(encoded_matrix, answer)
    
def TransformEigenValueToCosOfChebyShev(epsilon: float, eig_value: float, time: float) -> float:
    """cosをchebyshev級数展開した多項式を出力する
    
    Keyword arguments:
    LenOfAngSeq: pyqspで出力した角度のリストの長さ
    eig_value: 固有値
    time: 経過時間
    
    Returns:
    cosをchebyshev級数展開した多項式にeig_valueとtimeを代入した値
    """
    TransformedEigenValue = scipy.special.jv(0, time)
    r = scipy.optimize.fsolve(lambda r: (
        np.e * np.abs(time) / (2 * r))**r - (5 / 4) * epsilon, time)[0]
    R = np.floor(r / 2).astype(int)
    R = max(R, 1)
    for k in range(1, R+1):
        TransformedEigenValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k, time) *
                                    np.polynomial.chebyshev.chebval(eig_value, [0] * (2 * k) + [1]))
    return TransformedEigenValue
    
def TransformMatrixToCosOfChebyShev(epsilon: float, encoded_matrix: np.ndarray, time: float) -> np.ndarray:
    """matrixの固有値をcosをchebyshev級数展開した多項式に変換する
    
    Keyword arguments:
    LenOfAngSeq: pyqspで出力した角度のリストの長さ
    eig_value: 固有値
    time: 経過時間
    
    Returns:
    cosの多項式変形したmatrix
    """
    TransformedMatrix = np.zeros(encoded_matrix.shape)
    eig_values, eig_vecs = LA.eig(encoded_matrix)
    
    transformed_eig_values = np.array([TransformEigenValueToCosOfChebyShev(epsilon, eig_value, time) for eig_value in eig_values])
    diagonal_matrix = np.diag(transformed_eig_values)
    print(eig_values)
    print(transformed_eig_values)
    TransformedMatrix = eig_vecs @ diagonal_matrix @ eig_vecs.T
    
    return TransformedMatrix
            
def TransformEigenValueToSinOfChebyShev(epsilon: float, eig_value: float, time: float) ->float:
    """sinをchebyshev級数展開した多項式を出力する
    
    Keyword arguments:
    LenOfAngSeq: pyqspで出力した角度のリストの長さ
    eig_value: 固有値
    time: 経過時間
    
    Returns:
    sinをchebyshev級数展開した多項式にeig_valueとtimeを代入した値
    """
    TransformedEigenValue = 0
    r = scipy.optimize.fsolve(lambda r: (
        np.e * np.abs(time) / (2 * r))**r - (5 / 4) * epsilon, time)[0]
    R = np.floor(r / 2).astype(int)
    R = max(R, 1)
    for k in range(1, R):
        TransformedEigenValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k + 1, time) *
                                    np.polynomial.chebyshev.chebval(eig_value, [0] * (2 * k + 1) + [1]))
    return TransformedEigenValue
        
def TransformMatrixToSinOfChebyShev(epsilon: float, encoded_matrix: np.ndarray, time: float) -> np.ndarray:
    """sinをchebyshev級数展開した多項式にmatrixを多項式変換する
    
    Keyword arguments:
    LenOfAngSeq: pyqspで出力した角度のリストの長さ
    eig_value: 固有値
    time: 経過時間
    
    Returns:
    sinの多項式変形したmatrix
    """
    TransformedMatrix = np.zeros(encoded_matrix.shape)
    eig_values, eig_vecs = LA.eig(encoded_matrix)
        
    for eig_i, eig_value in enumerate(eig_values):
        TransformedMatrix += TransformEigenValueToSinOfChebyShev(epsilon, eig_value, time) * np.dot(eig_vecs[eig_i].T, eig_vecs[eig_i])
        
    return TransformedMatrix

@pytest.mark.parametrize(
    "NumOfSite, ValueOfH, time, epsilon",
    [
        (pow(2,1), 1.0, 1.0, 0.1),
        (pow(2,2), 1.0, 1.0, 0.2),
        (pow(2,3), 1.0, 1.0, 0.1),
        (pow(2,1), 2.0, 1.0, 0.2),
        (pow(2,2), 2.0, 1.0, 0.1),
        (pow(2,3), 2.0, 1.0, 0.1),
        (pow(2,1), 3.0, 1.0, 0.1),
        (pow(2,2), 3.0, 1.0, 0.1),
        (3, 1.0, 1.0, 0.1),
        (5, 1.0, 1.0, 0.1),
        (6, 1.0, 1.0, 0.1),
        (7, 1.0, 1.0, 0.1),
        (3, 2.0, 1.0, 0.1),
        (5, 2.0, 1.0, 0.1),
        (6, 2.0, 1.0, 0.1),
        (7, 2.0, 1.0, 0.1),
    ],
)
def test_QSVTAndCosOfChebyshev(NumOfSite: int, ValueOfH: float, time: float, epsilon: float):
    """EncodingしたHamiltonianをcosを展開した多項式に変形してテストする
    EncodingしたHamiltonian: 横磁場イジングのHamiltonian
    Testするために使うbit数: NumOfGateForEncoding + 2
    """
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    NumOfGateForTestCos = var_of_system.NumOfGateForEncoding + 2
    qc = QuantumCircuit(NumOfGateForTestCos)
    
    qc.append(CosGate(var_of_system, AngListForCos(var_of_system, time, epsilon)), list(range(NumOfGateForTestCos)))
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))
    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    #answerのmatrixを作成する
    answer = sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = answer + var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = TransformMatrixToCosOfChebyShev(epsilon, answer, time)
    #testする
    assert(np.allclose(answer, encoded_matrix))
    
@pytest.mark.parametrize(
    "NumOfSite, ValueOfH, time, epsilon",
    [
        (pow(2,1), 1.0, 1.0, 0.1),
        (pow(2,2), 1.0, 1.0, 0.2),
        (pow(2,3), 1.0, 1.0, 0.1),
        (pow(2,1), 2.0, 1.0, 0.2),
        (pow(2,2), 2.0, 1.0, 0.1),
        (pow(2,3), 2.0, 1.0, 0.1),
        (pow(2,1), 3.0, 1.0, 0.1),
        (pow(2,2), 3.0, 1.0, 0.1),
        (3, 1.0, 1.0, 0.1),
        (5, 1.0, 1.0, 0.1),
        (6, 1.0, 1.0, 0.1),
        (7, 1.0, 1.0, 0.1),
        (3, 2.0, 1.0, 0.1),
        (5, 2.0, 1.0, 0.1),
        (6, 2.0, 1.0, 0.1),
        (7, 2.0, 1.0, 0.1),
    ],
)

def test_QSVTAndMinusISinOfChebyshev(NumOfSite: int, ValueOfH: float, time: float, epsilon: float):
    """EncodingしたHamiltonianをsinを展開した多項式に変形してテストする
    EncodingしたHamiltonianは横磁場イジングのHamiltonian
    Testするために使うbit数: NumOfGateForEncoding + 2
    """
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    NumOfGateForTestSin = var_of_system.NumOfGateForEncoding + 2
    qc = QuantumCircuit(NumOfGateForTestSin)
    
    qc.append(SinGate(var_of_system, AngListForSine(var_of_system, time, epsilon)))
    backend = Aer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    result = job.result()
    whole_matrix =np.array(result.get_unitary(qc))
    encoded_matrix = whole_matrix[:pow(2,var_of_system.NumOfSite), :pow(2, var_of_system.NumOfSite)]
    encoded_matrix = encoded_matrix.T
    #answerのmatrixを作成する
    answer = sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = answer + var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = TransformMatrixToSinOfChebyShev(epsilon, answer, time)
    answer *= answer * -1j
    #testする
    assert(np.allclose(answer, encoded_matrix))
    
def test_ExpOverTwo():
    pass

"""
def main():
    NumOfSite = 2
    ValueOfH = 1.0
    epsilon = 0.1
    time = 1.0
    var_of_system = setting_var_of_system(NumOfSite, ValueOfH)
    answer = sum(SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    answer = answer + var_of_system.ValueOfH * sum(Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
    print(answer)
    answer = TransformMatrixToCosOfChebyShev(epsilon, answer, time)
    print(answer)
    
if __name__ == '__main__':
    main()
"""
