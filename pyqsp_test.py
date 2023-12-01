from pyqsp.angle_sequence import QuantumSignalProcessingPhases, StringPolynomial
from pyqsp.response import PlotQSPResponse

poly = StringPolynomial("np.cos(3*x)", 6)
ang_seq = QuantumSignalProcessingPhases(poly, method="tf")

PlotQSPResponse(ang_seq, target=poly, signal_operator="Wx")