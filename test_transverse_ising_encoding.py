import pytest
from transverse_ising import VarOfSystem, CheckLessThan2ToTheN, EncodingHamiltonian, qc_controlledSS, qc_controlledSx, construct_G
from transverse_ising import CosGate, SinGate
import numpy as np
import numpy.linalg as LA
import scipy.special

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy

class TestEncoding:
    def setting_var_of_system(self, NumOfSite: int, ValueOfH: float) -> VarOfSystem:
        """系の変数を与える"""
        var_of_system = VarOfSystem
        var_of_system.NumOfSite = 3
        var_of_system.ValueOfH = 1.0
        var_of_system.NumOfSS = var_of_system.NumOfSite
        var_of_system.NumOfSx = var_of_system.NumOfSite
        var_of_system.NumOfUnitary = var_of_system.NumOfSS + var_of_system.NumOfSx
        var_of_system.NumOfAncillaForEncoding = CheckLessThan2ToTheN(var_of_system.NumOfUnitary)
        var_of_system.NumOfGateForEncoding = var_of_system.NumOfSite + var_of_system.NumOfAncillaForEncoding
        
        return var_of_system

    def Zgate(self) -> np.ndarray:
        """Zgateの行列を返す"""
        return np.array([[1,0], [0, -1]])

    def Igate(self) -> np.ndarray:
        """Identity Matrixを返す"""
        return np.array([[1,0], [0,1]])

    def Xgate(self) -> np.ndarray:
        """Xgateの行列を返す"""
        return np.array([[0,1],[1,0]])

    def SSz_matrix(self, x: int, var_of_system: VarOfSystem) -> np.ndarray:
        """S_z * S_zのmatrixを返す"""
        if x != var_of_system.NumOfSite -1:
            matrix_list = [self.Zgate() if i == x or i == x+1 else self.Igate() for i in range(var_of_system.NumOfSite)]
        else:
            matrix_list = [self.Zgate() if i == 0 or i == x else self.Igate() for i in range(var_of_system.NumOfSite)] 
        
        SS = np.array([-1])
        for matrix in matrix_list:
            SS = np.kron(SS, matrix)
        return SS

    def Sx_matrix(self, x: int, var_of_system: VarOfSystem) -> np.ndarray:
        """S_xのmatrixを返す"""
        Sx = np.array([1])
        for i in range(var_of_system.NumOfSite):
            if i ==x:
                Sx = np.kron(Sx, self.Xgate())
            else:
                Sx = np.kron(Sx, self.Igate())
        return Sx

    def Identity_matrixFortest(self, var_of_system: VarOfSystem) -> np.ndarray:
        """系のハミルトニアンと同じサイズのIdentity matrixを返す"""
        S = np.array([1])
        for i in range(var_of_system.NumOfSite):
            S = np.kron(S, self.Igate())
        return S

    def AnswerMatrix(self, var_of_system: VarOfSystem) -> np.ndarray:
        """encodingされたhamiltonianのanswerを出力する"""
        answer = sum(self.SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
        answer = answer + var_of_system.ValueOfH * sum(self.Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
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
    def test_Encoding(self, NumOfSite, ValueOfH):
        """ある磁場でsite数がNのときにハミルトニアンが正しくエンコーディングされているかテストする"""
        #このテスト関数をパラメタライズする
        #Given: Hamiltonian の変数を与える
        var_of_system = self.setting_var_of_system(NumOfSite, ValueOfH)
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
        answer = sum(self.SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
        answer = answer + var_of_system.ValueOfH * sum(self.Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSite))
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
    def test_controlledSS(self, NumOfSite, ValueOfH):
        """controlledSSをテストする"""
        var_of_system = self.setting_var_of_system(NumOfSite, ValueOfH)
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
        answer =  sum(self.SSz_matrix(x, var_of_system) for x in range(var_of_system.NumOfSS))
        answer = answer +  var_of_system.ValueOfH * sum(self.Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSx))
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
    def test_controlledSx(self, NumOfSite, ValueOfH):
        """controlledSxをテストする"""
        var_of_system = self.setting_var_of_system(NumOfSite, ValueOfH)
        qc = QuantumCircuit(var_of_system.NumOfGate)
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
        answer = var_of_system.ValueOfH * sum(self.Sx_matrix(x, var_of_system) for x in range(var_of_system.NumOfSx))
        answer += sum(self.Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSS))
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
        
    def test_Identity(self):
        NumOfSite = 2
        ValueOfH = 1.0
        var_of_system = self.setting_var_of_system(NumOfSite, ValueOfH)
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
        answer = sum(self.Identity_matrixFortest(var_of_system) for _ in range(var_of_system.NumOfSite * 2))
        assert np.allclose(encoded_matrix, answer)

