from matplotlib import axes
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Tuple, List

class Gate:
    'Base class for all quantum gates. Each specific gate will inherit from this class and implement the matrix method to define its behavior. The num_qutrits attribute specifies how many qutrits the gate operates on, which is important for correctly applying the gate in a quantum circuit.'
    num_qutrits: int

    def matrix(self) -> np.ndarray:
        raise NotImplementedError

class CustomGate(Gate):
    """
    User-defined quantum gate for ternary (qutrit) systems.

    Behavior:

        This class allows users to define arbitrary gates by directly supplying
        their unitary matrix representation. The gate can operate on:

            - 1 qutrit (no control)
            - 2 qutrits (1 control + 1 target)
            - 3 qutrits (2 controls + 1 target)

    Attributes
    ----------
    num_qutrits : int
        Number of qutrits the gate acts on (1, 2, or 3).
    _matrix : np.ndarray
        The unitary matrix representing the gate.
    
    Example 1 - Single-qutrit gate:

        from qutritlab import *

        M = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=complex)

        gate = CustomGate(M, num_qutrits=1)

        c = Circuit(num_qutrits=1)
        c.add_operation(gate)

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    
    Example 2 - Two-qutrit controlled gate:

        from qutritlab import *

        lam = np.pi

        M = np.array([[1,0,0, 0,0,0, 0,0,0],
               [0,1,0, 0,0,0, 0,0,0],
               [0,0,1, 0,0,0, 0,0,0],
               [0,0,0, np.exp(-1j*lam/2), 0, 0, 0,0,0],
               [0,0,0, 0, np.exp(1j*lam/2), 0, 0,0,0],
               [0,0,0, 0, 0, 1, 0,0,0],
               [0,0,0, 0,0,0, 1,0,0],
               [0,0,0, 0,0,0, 0,1,0],
               [0,0,0, 0,0,0, 0,0,1]], dtype=complex)

        gate = CustomGate(M, num_qutrits=2)

        c = Circuit(num_qutrits=2)

        c.add_operation(X(subspace=(0,1)), targets=[0])
        c.add_operation(H(subspace=(0,1)), targets=[1])
        c.add_operation(gate, controls=[0], targets=[1])
        c.add_operation(H(subspace=(0,1)), targets=[1])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """

    def __init__(self, matrix: np.ndarray, num_qutrits: int):

        if num_qutrits not in (1, 2, 3):
            raise ValueError("num_qutrits must be 1, 2, or 3")

        expected_dim = 3 ** num_qutrits

        if matrix.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Matrix must be {expected_dim}x{expected_dim} "
                f"for {num_qutrits} qutrit(s)"
            )

        self._matrix = matrix.astype(complex)
        self.num_qutrits = num_qutrits

    def matrix(self) -> np.ndarray:
        return self._matrix

class H(Gate):
    """
    Generalized Ternary Hadamard gate.

    - If no subspace is provided:
        Applies a full qutrit Hadamard gate, creating superposition over |0>, |1>, |2>.
        
    - If subspace=(i,j):
        applies a qubit Hadamard on that 2D subspace and leaves the third level unchanged.

    Example:

        from qutritlab import *

        c = Circuit(num_qutrits=1)
        c.add_operation(H())

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """
    num_qutrits = 1

    def __init__(self, subspace=None):
        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        # --- Case 1: Full ternary Hadamard ---
        if self.subspace is None:
            w = np.exp(2j * np.pi / 3)
            return np.array([
                [1, 1, 1],
                [1, w, w**2],
                [1, w**2, w]
            ], dtype=complex) / np.sqrt(3)

        # --- Case 2: Subspace Hadamard (qubit-style) ---
        i, j = self.subspace

        M = np.eye(3, dtype=complex)

        # zero out the 2x2 block
        M[i, i] = 0
        M[j, j] = 0
        M[i, j] = 0
        M[j, i] = 0

        # insert Hadamard block
        h = 1 / np.sqrt(2)
        M[i, i] = h
        M[i, j] = h
        M[j, i] = h
        M[j, j] = -h

        return M
    
class H_dagger(Gate):
    'Conjugate transpose of the generalized Ternary Hadamard gate. This gate is the inverse of the H gate and can be used to reverse the transformation applied by H. It works on the entire 3-dimensional space.'
    ''
    num_qutrits = 1

    def matrix(self):
        w = np.exp(2j * np.pi / 3)
        H = np.array([
            [1, 1, 1],
            [1, w, w**2],
            [1, w**2, w]
        ], dtype=complex) / np.sqrt(3)

        return np.conjugate(H.T)

class SHIFT(Gate):
    """
    Ternary Shift gate. 

    Behavior:
        This gate performs a cyclic permutation of the basis states: |0⟩ → |1⟩, |1⟩ → |2⟩, |2⟩ → |0⟩
    
    Example:

        from qutritlab import *

        c = Circuit(num_qutrits=1)
        c.add_operation(gate=SHIFT())

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities )
        print("Measurements:", results.measurements)

    """
    num_qutrits = 1

    def matrix(self):
        return np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=complex)

class X(Gate):
    """
    Ternary X gate acting on a chosen 2-level subspace.

    Example:

        from qutritlab import *

        c = Circuit(num_qutrits=1)
        c.add_operation(gate=X(subspace=(0,2)))

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """

    num_qutrits = 1

    def __init__(self, subspace=(0, 1)):
        if len(subspace) != 2:
            raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
        self.subspace = tuple(subspace)

    def matrix(self):
        i, j = self.subspace

        M = np.eye(3, dtype=complex)

        M[i, i] = 0
        M[j, j] = 0
        M[i, j] = 1
        M[j, i] = 1

        return M
    
class Y(Gate):
    """
    Ternary Y gate acting on a chosen 2-level subspace.

    Behavior:
        - Y(subspace=(0,1)) → |0⟩ → i|1⟩, |1⟩ → -i|0⟩
        - Y(subspace=(0,2)) → |0⟩ → i|2⟩, |2⟩ → -i|0⟩
        - Y(subspace=(1,2)) → |1⟩ → i|2⟩, |2⟩ → -i|1⟩
    
    Example:

        from qutritlab import *

        c = Circuit(num_qutrits=1)
        c.add_operation(gate=Y(subspace=(0,2)))

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """

    num_qutrits = 1

    def __init__(self, subspace=(0, 1)):
        if len(subspace) != 2:
            raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
        self.subspace = tuple(subspace)

    def matrix(self):
        i, j = self.subspace

        M = np.eye(3, dtype=complex)

        # zero out the diagonal entries of the subspace
        M[i, i] = 0
        M[j, j] = 0

        # imaginary off-diagonal elements
        M[i, j] = -1j
        M[j, i] = 1j

        return M 

class Z(Gate):
    """
    Ternary Z gate acting on a chosen 2-level subspace.

    Behavior:
        - Z() → |0⟩ → |0⟩, |1⟩ → ω|1⟩, |2⟩ → ω²|2⟩
        - Z(subspace=(0,1)) → |0⟩ → |0⟩, |1⟩ → -|1⟩
        - Z(subspace=(0,2)) → |0⟩ → |0⟩, |2⟩ → -|2⟩
        - Z(subspace=(1,2)) → |1⟩ → |1⟩, |2⟩ → -|2⟩
    
    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=1)

        c.add_operation(gate=H())
        c.add_operation(gate=Z())
        c.add_operation(gate=H())
        
        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """

    num_qutrits = 1

    def __init__(self, subspace=None):
        if subspace != None:
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        if self.subspace != None:
            i, j = self.subspace
            M = np.eye(3, dtype=complex)
            M[j, j] = -1  # phase flip on second basis state

            return M
        else:
            w = np.exp(2j * np.pi / 3)
            return [[1,0,0],[0,w,0],[0,0,w**2]]

class RX(Gate):
    """
    Rotational X gate for qutrits.

    Behavior:
        - If no subspace is provided:
        applies the generalized ternary RX rotation (acts on all 3 levels)

        - If subspace=(i,j):
        applies a qubit RX rotation on that 2D subspace and leaves the third level unchanged

        - RX(lam) → full ternary rotation
        - RX(lam, subspace=(0,1)) → RX01
        - RX(lam, subspace=(0,2)) → RX02
        - RX(lam, subspace=(1,2)) → RX12
    
    Example:

        from qutritlab import * 
  
        c = Circuit(num_qutrits=1)
        c.add_operation(gate=RX(np.pi/2, subspace=(0,1)))

        results = Simulator().run(c, shots = 1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """

    num_qutrits = 1

    def __init__(self, lam: float, subspace=None):
        self.lam = lam

        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        # --- Case 1: Full ternary RX ---
        if self.subspace is None:
            lam = self.lam
            return np.array([
                [(1+2*np.cos(lam))/3,
                 (-1j*np.sin(lam) + (1-np.cos(lam)))/3,
                 (-1j*np.sin(lam) + (1-np.cos(lam)))/3],

                [(-1j*np.sin(lam) + (1-np.cos(lam)))/3,
                 (1+2*np.cos(lam))/3,
                 (-1j*np.sin(lam) + (1-np.cos(lam)))/3],

                [(-1j*np.sin(lam) + (1-np.cos(lam)))/3,
                 (-1j*np.sin(lam) + (1-np.cos(lam)))/3,
                 (1+2*np.cos(lam))/3]
            ], dtype=complex)

        # --- Case 2: Subspace RX (embedded qubit rotation) ---
        i, j = self.subspace
        lam = self.lam

        M = np.eye(3, dtype=complex)

        # zero out the 2x2 block
        M[i, i] = 0
        M[j, j] = 0
        M[i, j] = 0
        M[j, i] = 0

        c = np.cos(lam / 2)
        s = -1j * np.sin(lam / 2)

        # insert RX block
        M[i, i] = c
        M[j, j] = c
        M[i, j] = s
        M[j, i] = s

        return M

class RY(Gate):
    """
    Rotational Y gate for qutrits.

    - If no subspace is provided:
        applies a generalized ternary RY rotation (acts on all 3 levels)

    - If subspace=(i,j):
        applies a qubit RY rotation on that 2D subspace and leaves the third level unchanged

    Behavior:
        - RY(lam) → full ternary rotation
        - RY(lam, subspace=(0,1)) → RY01
        - RY(lam, subspace=(0,2)) → RY02
        - RY(lam, subspace=(1,2)) → RY12
    
    Example:

        from qutritlab import * 
  
        c = Circuit(num_qutrits=1)
        c.add_operation(gate=RY(np.pi, subspace=(0,2)))

        results = Simulator().run(c, shots = 1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """

    num_qutrits = 1

    def __init__(self, lam: float, subspace=None):
        self.lam = lam

        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        lam = self.lam

        # --- Case 1: Full ternary RY ---
        if self.subspace is None:
            c = np.cos(lam)
            s = np.sin(lam)

            return np.array([
                [(1 + 2*c)/3, (-s + (1-c))/3, ( s + (1-c))/3],
                [( s + (1-c))/3, (1 + 2*c)/3, (-s + (1-c))/3],
                [(-s + (1-c))/3, ( s + (1-c))/3, (1 + 2*c)/3]
            ], dtype=complex)

        # --- Case 2: Subspace RY (embedded qubit rotation) ---
        i, j = self.subspace

        M = np.eye(3, dtype=complex)

        c = np.cos(lam / 2)
        s = np.sin(lam / 2)

        # zero out 2x2 block
        M[i, i] = 0
        M[j, j] = 0
        M[i, j] = 0
        M[j, i] = 0

        # insert RY block
        M[i, i] = c
        M[i, j] = -s
        M[j, i] = s
        M[j, j] = c

        return M
    
class RZ(Gate):
    """
    Rotational Z gate for qutrits.

    - If no subspace is provided:
        applies the generalized ternary RZ rotation (acts on all 3 levels)

    - If subspace=(i,j):
        applies a qubit RZ rotation on that 2D subspace and leaves the third level unchanged

    Behavior:
        - RZ(lam) → full ternary rotation
        - RZ(lam, subspace=(0,1)) → RZ01
        - RZ(lam, subspace=(0,2)) → RZ02
        - RZ(lam, subspace=(1,2)) → RZ12
    
    Example: 

        from qutritlab import * 
  
        c = Circuit(num_qutrits=1)

        c.add_operation(gate=H())
        c.add_operation(gate=RZ(2*np.pi/3))
        c.add_operation(gate=H())

        results = Simulator().run(c, shots = 1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """

    num_qutrits = 1

    def __init__(self, lam: float, subspace=None):
        self.lam = lam

        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two basis indices, e.g. (0,1)")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        lam = self.lam

        # --- Case 1: Full ternary RZ ---
        if self.subspace is None:
            return np.array([
                [1, 0, 0],
                [0, np.exp(1j * lam), 0],
                [0, 0, np.exp(2j * lam)]
            ], dtype=complex)

        # --- Case 2: Subspace RZ (embedded qubit rotation) ---
        i, j = self.subspace

        M = np.eye(3, dtype=complex)

        phase_minus = np.exp(-1j * lam / 2)
        phase_plus  = np.exp(1j * lam / 2)

        M[i, i] = phase_minus
        M[j, j] = phase_plus

        return M
    
class CP(Gate):
    """
    Generalized controlled-Phase (CP) gate for qutrits.

    Behavior:
        The gate applies a phase that depends multiplicatively on both the control and target qutrit levels. 
    
    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=2)

        c.add_operation(gate=SHIFT(), targets=[0])

        c.add_operation(gate=H(), targets=[1])
        c.add_operation(gate=CP(2*np.pi/3), controls=[0], targets=[1])
        c.add_operation(gate=H(), targets=[1])

        esults = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array(np.diag([1, 1, 1,1, np.exp(1j*self.lam), np.exp(2j*self.lam),1, np.exp(2j*self.lam), np.exp(4j*self.lam)]))

class CSUM(Gate):
    """
    Generalized controlled-Sum (CSUM) gate for qutrits.

    Behavior:
        This gate performs a modular addition of the control qutrit to the target qutrit. 
        Specifically, it adds the value of the control qutrit (0, 1, or 2) to the target qutrit and takes the result modulo 3'
    
    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=2)

        c.add_operation(gate=SHIFT(), targets=[0])
        c.add_operation(gate=CSUM(), controls=[0], targets=[1])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """
    num_qutrits = 2

    def matrix(self):
        G = np.zeros((9,9), dtype=complex)

        for c in range(3):
            for t in range(3):
                in_idx  = 3*c + t
                out_idx = 3*c + ((t + c) % 3)
                G[out_idx, in_idx] = 1

        return G

class CCSUM(Gate):
    """
    Generalized Controlled-CSUM (CCSUM) gate with two control qutrits.

    Behavior:
            Controlled CSUM gate. This gate performs a modular addition of the two control qutrits to the target qutrit. 
             Specifically, it adds the values of the two control qutrits (each can be 0, 1, or 2) to the target qutrit and takes the result modulo 3
    
    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=3)

        c.add_operation(gate=SHIFT(), targets=[0])
        c.add_operation(gate=SHIFT(), targets=[1])

        c.add_operation(gate=CCSUM(), controls=[0,1], targets=[2])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)

    """
    num_qutrits = 3

    def matrix(self):
        m = np.zeros((27,27))
        for c1 in range(3):
            for c2 in range(3):
                for t in range(3):
                    in_idx = c1*9 + c2*3 + t
                    if c1 == c2:
                        t_out = (t + c1) % 3
                    else:
                        t_out = t
                    out_idx = c1*9 + c2*3 + t_out
                    m[out_idx, in_idx] = 1
        return m

class CNOT(Gate):
    """
    Generalized controlled-X (CNOT) gate for qutrits.

    Parameters:
        control_state: which control state activates the gate (0, 1, or 2)
        subspace: (i, j) specifying which two levels of the target to swap

    Behavior:
        Applies an X (swap) on the target subspace (i,j)
        if the control qutrit is in state |control_state⟩.

    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=2)

        c.add_operation(gate=SHIFT(), targets=[1])
        c.add_operation(gate=CNOT(control_state=1, subspace=[0,2]), controls=[1], targets=[0])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """

    num_qutrits = 2

    def __init__(self, control_state=1, subspace=(0, 1)):
        if control_state not in (0, 1, 2):
            raise ValueError("control_state must be 0, 1, or 2")
        self.control_state = control_state

        if len(subspace) != 2:
            raise ValueError("subspace must contain exactly two indices")
        self.subspace = tuple(subspace)

    def matrix(self):
        control = self.control_state
        i, j = self.subspace

        M = np.eye(9, dtype=complex)

        # basis index: |c, t⟩ → 3*c + t
        def idx(c, t):
            return 3 * c + t

        # swap only inside the controlled block
        a = idx(control, i)
        b = idx(control, j)

        # perform swap
        M[a, a] = 0
        M[b, b] = 0
        M[a, b] = 1
        M[b, a] = 1

        return M
    
class CCX(Gate):
    """
    Generalized doubly-controlled X (CCX) gate for qutrits.

    Parameters:
        control_state: the state both control qutrits must be in (0, 1, or 2)
        subspace: (i, j) specifying which two levels of the target to swap

    Behavior:
        Applies an X (swap) on the target subspace (i,j)
        iff BOTH control qutrits are in state |control_state⟩.

    Example:
        - CCX(control_state=1, subspace=(0,1))  → CCX1101
        - CCX(control_state=1, subspace=(0,2))  → CCX1102
        - CCX(control_state=2, subspace=(1,2))  → CCX2212
    """

    num_qutrits = 3

    def __init__(self, control_state=1, subspace=(0, 1)):
        if control_state not in (0, 1, 2):
            raise ValueError("control_state must be 0, 1, or 2")
        self.control_state = control_state

        if len(subspace) != 2:
            raise ValueError("subspace must contain exactly two indices")
        self.subspace = tuple(subspace)

    def matrix(self):
        c = self.control_state
        i, j = self.subspace

        M = np.eye(27, dtype=complex)

        # basis index: |c1, c2, t⟩ → 9*c1 + 3*c2 + t
        def idx(c1, c2, t):
            return 9 * c1 + 3 * c2 + t

        # only activate when BOTH controls match
        a = idx(c, c, i)
        b = idx(c, c, j)

        # perform swap on target subspace
        M[a, a] = 0
        M[b, b] = 0
        M[a, b] = 1
        M[b, a] = 1

        return M

class CH(Gate):
    """
    Controlled Hadamard gate for qutrits.

    Parameters:
        control_state: which control state activates the gate (0, 1, or 2)
        subspace: (i, j) specifying the 2D subspace on the target

    Behavior:
        - If subspace is None:
            applies the full ternary Hadamard (DFT-3) when control == control_state

        - If subspace=(i,j):
            applies a qubit Hadamard on that subspace when control == control_state
    """

    num_qutrits = 2

    def __init__(self, control_state=1, subspace=None):
        if control_state not in (0, 1, 2):
            raise ValueError("control_state must be 0, 1, or 2")
        self.control_state = control_state

        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two indices")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        control = self.control_state

        # basis index: |c, t⟩ → 3*c + t
        def idx(c, t):
            return 3 * c + t

        M = np.eye(9, dtype=complex)

        # --- Case 1: Full ternary Hadamard ---
        if self.subspace is None:
            w = np.exp(2j * np.pi / 3)
            H3 = np.array([
                [1, 1, 1],
                [1, w, w**2],
                [1, w**2, w]
            ], dtype=complex) / np.sqrt(3)

            # replace the 3x3 block corresponding to control_state
            for t_in in range(3):
                for t_out in range(3):
                    M[idx(control, t_out), idx(control, t_in)] = H3[t_out, t_in]

            return M

        # --- Case 2: Subspace Hadamard ---
        i, j = self.subspace

        h = 1 / np.sqrt(2)

        # zero out the 2x2 block first
        M[idx(control, i), idx(control, i)] = 0
        M[idx(control, j), idx(control, j)] = 0
        M[idx(control, i), idx(control, j)] = 0
        M[idx(control, j), idx(control, i)] = 0

        # insert Hadamard block
        M[idx(control, i), idx(control, i)] = h
        M[idx(control, i), idx(control, j)] = h
        M[idx(control, j), idx(control, i)] = h
        M[idx(control, j), idx(control, j)] = -h

        return M
        
class CRZ(Gate):
    """
    Controlled RZ gate for qutrits.

    Parameters:
        lam: rotation angle
        control_state: which control state activates the gate (0, 1, or 2)
        subspace: (i, j) specifying the 2D subspace on the target

    Behavior:
        - If subspace is None:
            applies the generalized ternary RZ when control == 1 (lam) or 2 (2*lam)

        - If subspace=(i,j):
            applies a qubit RZ on that subspace when control == control_state
    
    Example:

        from qutritlab import * 

        c = Circuit(num_qutrits=2)

        c.add_operation(gate=X(subspace=[0,2]), targets=[0])

        c.add_operation(gate=H(subspace=[0,1]), targets=[1])
        c.add_operation(gate=CRZ(np.pi, control_state=2, subspace=[0,1]), controls=[0], targets=[1])
        c.add_operation(gate=H(subspace=[0,1]), targets=[1])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)


    """

    num_qutrits = 2

    def __init__(self, lam: float, control_state=None, subspace=None):
        self.lam = lam

        if control_state is not None and control_state not in (0, 1, 2):
            raise ValueError("control_state must be 0, 1, or 2")
        self.control_state = control_state

        if subspace is not None:
            if len(subspace) != 2:
                raise ValueError("subspace must contain exactly two indices")
            self.subspace = tuple(subspace)
        else:
            self.subspace = None

    def matrix(self):
        lam = self.lam

        # --- Case 1: Original generalized CRZ ---
        if self.subspace is None:
            return np.diag([
                1, 1, 1,
                1, np.exp(1j * lam), np.exp(2j * lam),
                1, np.exp(2j * lam), np.exp(4j * lam)
            ]).astype(complex)

        # --- Case 2: Subspace-controlled RZ ---
        i, j = self.subspace
        control = self.control_state

        if control is None:
            raise ValueError("control_state must be specified when using subspace")

        M = np.eye(9, dtype=complex)

        # indices in computational basis |c, t⟩ → 3*c + t
        def idx(c, t):
            return 3 * c + t

        phase_minus = np.exp(-1j * lam / 2)
        phase_plus  = np.exp(1j * lam / 2)

        # apply only when control matches
        M[idx(control, i), idx(control, i)] = phase_minus
        M[idx(control, j), idx(control, j)] = phase_plus

        return M
    
class CRZ1(Gate):
    'Controlled RZ gate. This gate applies a |1⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |1⟩, and does nothing if the control is in state |0⟩ or |2⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([[1,0,0, 0,0,0, 0,0,0],
               [0,1,0, 0,0,0, 0,0,0],
               [0,0,1, 0,0,0, 0,0,0],
               [0,0,0, np.exp(-1j*self.lam/2), 0, 0, 0,0,0],
               [0,0,0, 0, np.exp(1j*self.lam/2), 0, 0,0,0],
               [0,0,0, 0, 0, 1, 0,0,0],
               [0,0,0, 0,0,0, 1,0,0],
               [0,0,0, 0,0,0, 0,1,0],
               [0,0,0, 0,0,0, 0,0,1]], dtype=complex)


class CRZ2(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |0⟩ and |2⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |1⟩, and does nothing if the control is in state |0⟩ or |2⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([
    [1,0,0, 0,0,0, 0,0,0],
    [0,1,0, 0,0,0, 0,0,0],
    [0,0,1, 0,0,0, 0,0,0],

    [0,0,0, 1,0,0, 0,0,0],
    [0,0,0, 0,1,0, 0,0,0],
    [0,0,0, 0,0,1, 0,0,0],

    [0,0,0, 0,0,0, np.exp(-1j*(self.lam/2)), 0, 0],
    [0,0,0, 0,0,0, 0, np.exp(1j*(self.lam)/2), 0],
    [0,0,0, 0,0,0, 0, 0, 1]], dtype=complex)

class CRZ101(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |0⟩ and |1⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |0⟩, and does nothing if the control is in state |1⟩ or |2⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, np.exp(1j*self.lam/2), 0, 0, 0, 0, 0],
    [0, 0, 0, 0, np.exp(-1j*self.lam/2), 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    
class CRZ102(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |0⟩ and |2⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |0⟩, and does nothing if the control is in state |1⟩ or |2⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 1, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 1, 0, 0, 0, 0, 0, 0],    
                [0, 0, 0, np.exp(1j*self.lam/2), 0, 0, 0, 0, 0], 
                [0, 0, 0, 0, 1, 0, 0, 0, 0],                 
                [0, 0, 0, 0, 0, np.exp(-1j*self.lam/2), 0, 0, 0], 
                [0, 0, 0, 0, 0, 0, 1, 0, 0],  
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  
                 [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
class CRZ112(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |1⟩ and |2⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |0⟩, and does nothing if the control is in state |1⟩ or |2⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 1, 0, 0, 0, 0, 0, 0, 0],  
                [0, 0, 1, 0, 0, 0, 0, 0, 0],     
                [0, 0, 0, 1, 0, 0, 0, 0, 0],                 
                [0, 0, 0, 0, np.exp(1j*self.lam/2), 0, 0, 0, 0],  
                [0, 0, 0, 0, 0, np.exp(-1j*self.lam/2), 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0], 
                [0, 0, 0, 0, 0, 0, 0, 1, 0],  
                [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    
class CRZ201(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |0⟩ and |1⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |2⟩, and does nothing if the control is in state |0⟩ or |1⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, np.exp(1j*self.lam/2), 0, 0],
        [0, 0, 0, 0, 0, 0, 0, np.exp(-1j*self.lam/2), 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    
class CRZ202(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |0⟩ and |0⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |2⟩, and does nothing if the control is in state |0⟩ or |1⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, np.exp(-1j*self.lam/2), 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(1j*self.lam/2)]], dtype=complex)

class CRZ212(Gate):
    'Controlled RZ gate. This gate applies a Z rotation to the |1⟩ and |2⟩ subspace of the target qutrit based on the state of the control qutrit. The parameter lam determines the angle of the rotation applied when the control is in state |2⟩, and does nothing if the control is in state |0⟩ or |1⟩'
    num_qutrits = 2

    def __init__(self, lam: float):
        self.lam = lam

    def matrix(self):
        return np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, np.exp(1j*self.lam/2), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, np.exp(-1j*self.lam/2)]], dtype=complex)

class SWAP(Gate):
    """
    Ternary SWAP gate

    Behavior:
        This gate swaps the states of two qutrits. For example, if the input state is |a,b⟩, where a and b can be 0, 1, or 2, the output state will be |b,a⟩

    Example:

        from qutritlab import *

        c = Circuit(num_qutrits=2)
        c.add_operation(gate=X(subspace=(0,1)), controls=[0], targets=[1])
        c.add_operation(gate=SWAP(), controls=[0], targets=[1])

        results = Simulator().run(c, shots=1000, ternary_string=True)

        print("State:", results.state)
        print("Probabilities:", results.probabilities)
        print("Measurements:", results.measurements)
    """
    num_qutrits = 2

    def matrix(self):
        
        return np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # |0,0> → |0,0>
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # |0,1> → |1,0>
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # |0,2> → |2,0>
    [0, 1, 0, 0, 0, 0, 0, 0, 0],  # |1,0> → |0,1>
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # |1,1> → |1,1>
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # |1,2> → |2,1>
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # |2,0> → |0,2>
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # |2,1> → |1,2>
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # |2,2> → |2,2>
    ], dtype=complex)
    
@dataclass
class Operation:
    'Represents a single operation in a quantum circuit, consisting of a gate and the target qutrit(s) it is applied to. The Operation class encapsulates the information about which gate is being applied and to which qutrit(s) it should be applied, allowing the Simulator to correctly apply the gate during the execution of the circuit.'
    gate: Gate
    controls: Tuple[int, ...]
    targets: Tuple[int, ...]

class Circuit:
    """
    Represents a quantum circuit composed of a sequence of operations.

    A circuit consists of quantum gates applied to specific qutrits in a
    defined order. This class provides methods to construct a circuit by
    adding operations and specifying their target and control qutrits.
    """

    def __init__(self, num_qutrits: int):
        """
        Initialize a new quantum circuit.

        Args:
            num_qutrits (int): The number of qutrits in the circuit.
        """
        self.num_qutrits = num_qutrits
        self.operations: List[Operation] = []

    def add_operation(self, gate: Gate, controls=None, targets=None):
        """
        Add a quantum operation (gate application) to the circuit.

        The behavior depends on the ``targets`` argument:
        - If ``targets`` is None and the gate acts on a single qutrit, the
          operation is applied to all qutrits in the circuit.
        - If ``targets`` is a single integer, the operation is applied to
          that qutrit.

        Args:
            gate (Gate): The quantum gate to apply.
            controls (Iterable[int], optional): Indices of control qutrits.
                Defaults to None (no controls).
            targets (int or Iterable[int], optional): Target qutrit index or
                indices. Defaults to None.

        Raises:
            ValueError: If ``targets`` is None and the gate requires more than
                one qutrit.

        Returns:
            None
        """
        if targets == None:
            if gate.num_qutrits == 1:
                for i in range(self.num_qutrits):
                    self.operations.append(
                        Operation(
                            gate,
                            tuple(controls) if controls is not None else (),
                            (i,),
                        )
                    )
                return
            else:
                raise ValueError(
                    f"{gate.__class__.__name__} requires {gate.num_qutrits} targets"
                )

        if len(targets) == 1 and isinstance(targets[0], (list, tuple)):
            targets = tuple(targets[0])

        self.operations.append(
            Operation(
                gate,
                tuple(controls) if controls is not None else (),
                tuple(targets),
            )
        )

@dataclass
class Results:
    state: Tuple[int, ...]
    probabilities: any
    measurements: list

class Simulator:
    """
    Simulates the execution of a quantum circuit.

    This class applies a sequence of quantum gates from a given ``Circuit``
    object to an initial quantum state. The result is the final state vector
    after all operations have been applied.
    """

    def run(self, circuit: Circuit, shots, ternary_string=None, binary_string=None):
        """
        Execute a quantum circuit simulation.

        The simulation starts from the default initial state (|0⟩ for all
        qutrits). Each operation in the circuit is applied sequentially
        to evolve the state.

        After applying all operations, measurement probabilities and sampled
        measurement results are computed.

        Args:
            circuit (Circuit): The quantum circuit to simulate.
            shots (int): Number of measurement samples to generate.
            ternary_string (str, optional): Optional ternary basis string for
                measurement interpretation.
            binary_string (str, optional): Optional binary basis string for
                measurement interpretation.

        Returns:
            Results: An object containing:
                - state (np.ndarray): Final state vector.
                - probabilities (np.ndarray): Measurement probabilities.
                - measurements (list): Sampled measurement outcomes.
        """
        N = circuit.num_qutrits
        state = np.zeros(3**N, dtype=complex)
        state[0] = 1

        for op in circuit.operations:
            state = self._apply(state, op, N)

        probs = get_probabilities(state)
        measurements = get_measurements(
            state,
            shots,
            ternary_string=ternary_string,
            binary_string=binary_string,
        )

        results = Results(
            state=state,
            probabilities=probs,
            measurements=measurements,
        )

        return results
    
# class Simulator:
#     'Simulates the execution of a quantum circuit by applying the gates in sequence to an initial state. The Simulator class takes a Circuit object and computes the resulting quantum state after all operations have been applied.'
#     def run(self, circuit: Circuit, shots, ternary_string=None, binary_string=None):
#         'Performs an execution of a quantum circuit by applying the gates in sequence to an initial state. The initial state is typically the |0⟩ state for all qutrits, represented as a vector with a 1 in the first position and 0s elsewhere. The method iterates through each operation in the circuit, applying the corresponding gate to the current state vector, and returns the final state after all operations have been applied.'
#         N = circuit.num_qutrits
#         state = np.zeros(3**N, dtype=complex)
#         state[0] = 1

#         for op in circuit.operations:
#             state = self._apply(state, op, N)

#         probs = get_probabilities(state)
#         measurements = get_measurements(state,shots,ternary_string=ternary_string, binary_string=binary_string)

#         results = Results(state=state, probabilities=probs, measurements=measurements)

#         return results

    def _apply(self, state, op: Operation, N: int):
        'Applies a single operation (gate) to a quantum state'
        G = op.gate.matrix()
        k = op.gate.num_qutrits

        if k == 1:
            return self._apply_single(state, G, op.targets[0], N)
        elif k == 2:
                return self._apply_two(state, G, op.controls[0], op.targets, N)
        elif k == 3:
                return self._apply_three(state, G,op.controls, op.targets, N)
        raise NotImplementedError

    def _apply_single(self, state, G, target, N):
        'Applies a single-qutrit gate to a quantum state'
        psi = state.reshape((3,) * N, order="F")
        psi = np.moveaxis(psi, target, 0)
        psi = np.tensordot(G, psi, axes=[[1],[0]])
        psi = np.moveaxis(psi, 0, target)

        return psi.reshape(-1, order="F")

    def _apply_two(self, state, G, control ,targets, N):
        'Applies a two-qutrit gate to the quantum state. '
        target = targets[0]

        psi = state.reshape((3,) * N)
        psi = np.transpose(psi, list(reversed(range(N))))
        axes = [control, target] + [i for i in range(N) if i not in (control, target)]
        psi = np.transpose(psi, axes)

        psi = psi.reshape(9, -1)
        psi = G @ psi
        psi = psi.reshape((3, 3) + (3,) * (N - 2))

        inv = np.argsort(axes)
        psi = np.transpose(psi, inv)
        psi = np.transpose(psi, list(reversed(range(N))))

        return psi.reshape(-1)

    def _apply_three(self, state, G, controls,targets, N):
        c2, c1 = controls
        t1 = targets[0]

        psi = state.reshape((3,) * N)
        psi = np.transpose(psi, list(reversed(range(N))))
        axes = [c2, c1, t1] + [i for i in range(N) if i not in (c2,c1,t1)]
        psi = np.transpose(psi, axes)

        psi = psi.reshape(27, -1)
        psi = G @ psi
        psi = psi.reshape((3,3,3) + (3,) * (N-3))

        inv = np.argsort(axes)
        psi = np.transpose(psi, inv)
        psi = np.transpose(psi, list(reversed(range(N))))

        return psi.reshape(-1)

def get_probabilities(state):
    'Given a quantum state vector, this function calculates the probability of measuring each possible outcome. The probabilities are obtained by taking the absolute square of the amplitudes in the state vector and normalizing them so that they sum to 1.'
    probs = np.abs(state)**2
    return probs / probs.sum()

# def get_measurements(state, shots=1000, ternary_string = False, binary_string = False):
#     'Performs measurements of a quantum state by generating random samples based on the probabilities of each outcome. The function takes a quantum state vector, the number of measurement shots to simulate, and an optional parameter to determine the format of the output. It returns a dictionary (or sorted list) containing the counts of each measured outcome.'
#     probs = get_probabilities(state)
#     samples = np.random.choice(len(probs), p=probs, size=shots)
#     counts = {}
#     binary_counts = {}
#     if ternary_string == True:
#         counts = Counter(samples)
#         counts = sorted([(_ternary(i), c) for i, c in counts.items()],key=lambda x: int(x[0]))
    
#     if binary_string == True:
#         counts = Counter(samples)
#         for s in samples: 
#             binary_counts += bin(counts[s])[2:]
#         counts = sorted(binary_counts)
    
#     if binary_string == False and ternary_string == False:
#         for s in samples:
#             counts[s] = counts.get(s, 0) + 1
#     return counts

# from collections import Counter
# import numpy as np
import math

def to_ternary_padded(x, n):
    trits = []
    for _ in range(n):
        trits.append(str(x % 3))
        x //= 3
    return ''.join(reversed(trits)).zfill(n)


def get_measurements(state, shots=1000, ternary_string=False, binary_string=False):
    probs = get_probabilities(state)
    samples = np.random.choice(len(probs), p=probs, size=shots)

    n_qutrits = round(math.log(len(probs), 3))  # assumes full 3^n space

    counts = Counter(samples)

    if ternary_string:
        counts = {
            to_ternary_padded(i, n_qutrits): c
            for i, c in counts.items()
        }
        return sorted(counts.items(), key=lambda x: int(x[0], 3))

    elif binary_string:
        # keeping your structure but fixing it
        binary_counts = {
            format(i, f"0{n_qutrits}b"): c
            for i, c in counts.items()
        }
        return sorted(binary_counts.items(), key=lambda x: int(x[0], 2))

    else:
        return dict(counts)
    
def _ternary (n):
    if n == 0:
        return '0'
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(str(r))
    return ''.join(reversed(nums))

if __name__ == "__main__":
    c = Circuit(2)

    sim = Simulator()
    state = sim.run(c)

    print("State:", state)
    print("Probabilities:", get_probabilities(state))

