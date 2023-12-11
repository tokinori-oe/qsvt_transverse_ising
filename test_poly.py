import pytest
from transverse_ising import VarOfSystem, CheckLessThan2ToTheN, EncodingHamiltonian, qc_controlledSS, qc_controlledSx, construct_G
from transverse_ising import CosGate, SinGate
from test_encoding import TestEncoding
import numpy as np
import numpy.linalg as LA
import scipy.special

#import qiskit
from qiskit import IBMQ, Aer, transpile, execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.providers.ibmq import least_busy


class TestTransformingToExpOverTwo:
    def TransformEigenValueToCosOfChebyShev(self, LenOfAngSeq: int, eig_value: float, time: float) -> float:
        TransformedEigenValue = scipy.special.jv(0, time)
        for k in range(1, LenOfAngSeq):
            TransformedEigenValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k, time) *
                                        np.polynomial.chebyshev.chebval(eig_value, [0] * (2 * k) + [1]))
        return TransformedEigenValue
    
    def TransformMatrixToCosOfChebyShev(self, LenOfAngSeq: int, encoded_matrix: np.ndarray, time: float) -> np.ndarray:
        TransformedMatrix = np.zeros(encoded_matrix.shape)
        eig_values, eig_vecs = LA.eig(encoded_matrix)
        
        for eig_i, eig_value in enumerate(eig_values):
            TransformedMatrix += self.TransformEigenValueToCosOfChebyShev(LenOfAngSeq, eig_value, time) * np.dot(eig_vecs[eig_i].T, eig_vecs[eig_i])
            
        return TransformedMatrix
            
    def TransformEigenValueToSinOfChebyShev(self, LenOfAngSeq: int, eig_value: float, time: float) ->float:
        TransformedEigenValue = 0
        for k in range(1, LenOfAngSeq):
            TransformedEigenValue += 2 * ( pow(-1, k) * scipy.special.jv(2 * k + 1, time) *
                                        np.polynomial.chebyshev.chebval(eig_value, [0] * (2 * k + 1) + [1]))
        return TransformedEigenValue
        
    def TransformMatrixToSinOfChebyShev(self, LenOfAngSeq: int, encoded_matrix: np.ndarray, time: float) -> np.ndarray:
        TransformedMatrix = np.zeros(encoded_matrix.shape)
        eig_values, eig_vecs = LA.eig(encoded_matrix)
        
        for eig_i, eig_value in enumerate(eig_values):
            TransformedMatrix += self.TransformEigenValueToSinOfChebyShev(LenOfAngSeq, eig_value, time) * np.dot(eig_vecs[eig_i].T, eig_vecs[eig_i])
            
        return TransformedMatrix

    def test_QSVTAndCosOfChebyshev(self):
        pass

    def test_QSVTAndSinOfChebyshev(self):
        pass

    def test_QSVTAndMinusISinOfChebyshev(self):
        pass
    
    def test_ExpOverTwo(self):
        pass
