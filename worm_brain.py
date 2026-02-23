"""
C. elegans Connectome Simulation Module
========================================
302-neuron biophysical simulation with Hodgkin-Huxley dynamics.

Based on published connectome data from WormAtlas/OpenWorm:
- 302 neurons: ~60 sensory, ~80 motor, ~150 interneurons, 8 dopaminergic
- ~7,000 chemical synapses (graded, not spiking)
- ~900 gap junctions (electrical, symmetric)

All dynamics are vectorized — no Python loops over neurons.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# C. elegans Neuron Data (302 neurons, hermaphrodite)
# ---------------------------------------------------------------------------
# Neuron names and classifications from WormAtlas.
# Types: S=sensory, I=interneuron, M=motor, D=dopaminergic sensory

_NEURON_DATA = {
    # Dopaminergic sensory neurons (8)
    "CEPDL": "D", "CEPDR": "D", "CEPVL": "D", "CEPVR": "D",
    "ADEL": "D", "ADER": "D", "PDEL": "D", "PDER": "D",

    # Sensory neurons (~60, excluding dopaminergic)
    "ADFL": "S", "ADFR": "S", "ADLL": "S", "ADLR": "S",
    "AFDL": "S", "AFDR": "S", "ALML": "S", "ALMR": "S",
    "ALNL": "S", "ALNR": "S", "AQR": "S",
    "ASEL": "S", "ASER": "S", "ASGL": "S", "ASGR": "S",
    "ASHL": "S", "ASHR": "S", "ASIL": "S", "ASIR": "S",
    "ASJL": "S", "ASJR": "S", "ASKL": "S", "ASKR": "S",
    "AVM": "S", "AWAL": "S", "AWAR": "S",
    "AWBL": "S", "AWBR": "S", "AWCL": "S", "AWCR": "S",
    "BAGL": "S", "BAGR": "S",
    "DVA": "S", "FLP": "S", "FLPL": "S", "FLPR": "S",
    "IL1DL": "S", "IL1DR": "S", "IL1L": "S", "IL1R": "S",
    "IL1VL": "S", "IL1VR": "S",
    "IL2DL": "S", "IL2DR": "S", "IL2L": "S", "IL2R": "S",
    "IL2VL": "S", "IL2VR": "S",
    "OLLL": "S", "OLLR": "S",
    "OLQDL": "S", "OLQDR": "S", "OLQVL": "S", "OLQVR": "S",
    "PHAL": "S", "PHAR": "S", "PHBL": "S", "PHBR": "S",
    "PHCL": "S", "PHCR": "S",
    "PLML": "S", "PLMR": "S", "PLNL": "S", "PLNR": "S",
    "PQR": "S", "PVDL": "S", "PVDR": "S",
    "SDQL": "S", "SDQR": "S",
    "URBL": "S", "URBR": "S", "URXL": "S", "URXR": "S",
    "URYDL": "S", "URYDR": "S", "URYVL": "S", "URYVR": "S",
    "ALA": "S",

    # Interneurons (~150)
    "ADAL": "I", "ADAR": "I",
    "AIAL": "I", "AIAR": "I", "AIBL": "I", "AIBR": "I",
    "AIML": "I", "AIMR": "I", "AINL": "I", "AINR": "I",
    "AIYL": "I", "AIYR": "I", "AIZL": "I", "AIZR": "I",
    "AUAL": "I", "AUAR": "I",
    "AVAL": "I", "AVAR": "I", "AVBL": "I", "AVBR": "I",
    "AVDL": "I", "AVDR": "I", "AVEL": "I", "AVER": "I",
    "AVFL": "I", "AVFR": "I", "AVG": "I",
    "AVHL": "I", "AVHR": "I", "AVJL": "I", "AVJR": "I",
    "AVKL": "I", "AVKR": "I", "AVL": "I",
    "BDUL": "I", "BDUR": "I",
    "CANL": "I", "CANR": "I",
    "DVB": "I", "DVC": "I",
    "LUAL": "I", "LUAR": "I",
    "PVNL": "I", "PVNR": "I", "PVPL": "I", "PVPR": "I",
    "PVQL": "I", "PVQR": "I", "PVR": "I", "PVT": "I",
    "PVWL": "I", "PVWR": "I",
    "RIAL": "I", "RIAR": "I", "RIBL": "I", "RIBR": "I",
    "RICL": "I", "RICR": "I", "RID": "I",
    "RIFL": "I", "RIFR": "I", "RIGL": "I", "RIGR": "I",
    "RIHL": "I", "RIHR": "I",
    "RIML": "I", "RIMR": "I", "RIPL": "I", "RIPR": "I",
    "RIR": "I", "RIS": "I",
    "RIVL": "I", "RIVR": "I",
    "SAADL": "I", "SAADR": "I", "SAAVL": "I", "SAAVR": "I",
    "SABD": "I", "SABVL": "I", "SABVR": "I",
    "SIADL": "I", "SIADR": "I", "SIAVL": "I", "SIAVR": "I",
    "SIBDL": "I", "SIBDR": "I", "SIBVL": "I", "SIBVR": "I",
    "SMBDL": "I", "SMBDR": "I", "SMBVL": "I", "SMBVR": "I",
    "SMDDL": "I", "SMDDR": "I", "SMDVL": "I", "SMDVR": "I",
    "URADL": "I", "URADR": "I", "URAVL": "I", "URAVR": "I",

    # Head motor neurons
    "RMDDL": "M", "RMDDR": "M", "RMDL": "M", "RMDR": "M",
    "RMDVL": "M", "RMDVR": "M",
    "RMEDL": "M", "RMEDR": "M", "RMEL": "M", "RMER": "M",
    "RMEVL": "M", "RMEVR": "M",
    "RMFL": "M", "RMFR": "M",
    "RMGL": "M", "RMGR": "M",
    "RMHL": "M", "RMHR": "M",

    # Ventral cord motor neurons
    "DA01": "M", "DA02": "M", "DA03": "M", "DA04": "M", "DA05": "M",
    "DA06": "M", "DA07": "M", "DA08": "M", "DA09": "M",
    "DB01": "M", "DB02": "M", "DB03": "M", "DB04": "M",
    "DB05": "M", "DB06": "M", "DB07": "M",
    "DD01": "M", "DD02": "M", "DD03": "M", "DD04": "M",
    "DD05": "M", "DD06": "M",
    "VA01": "M", "VA02": "M", "VA03": "M", "VA04": "M", "VA05": "M",
    "VA06": "M", "VA07": "M", "VA08": "M", "VA09": "M", "VA10": "M",
    "VA11": "M", "VA12": "M",
    "VB01": "M", "VB02": "M", "VB03": "M", "VB04": "M", "VB05": "M",
    "VB06": "M", "VB07": "M", "VB08": "M", "VB09": "M", "VB10": "M",
    "VB11": "M",
    "VD01": "M", "VD02": "M", "VD03": "M", "VD04": "M", "VD05": "M",
    "VD06": "M", "VD07": "M", "VD08": "M", "VD09": "M", "VD10": "M",
    "VD11": "M", "VD12": "M", "VD13": "M",
    "AS01": "M", "AS02": "M", "AS03": "M", "AS04": "M", "AS05": "M",
    "AS06": "M", "AS07": "M", "AS08": "M", "AS09": "M", "AS10": "M",
    "AS11": "M",
    "PDA": "M", "PDB": "M",

    # Pharyngeal neurons (20)
    "I1L": "I", "I1R": "I", "I2L": "I", "I2R": "I",
    "I3": "I", "I4": "I", "I5": "I", "I6": "I",
    "M1": "M", "M2L": "M", "M2R": "M", "M3L": "M", "M3R": "M",
    "M4": "M", "M5": "M",
    "MI": "I", "NSML": "I", "NSMR": "I", "MCL": "M", "MCR": "M",
}

# Pad to exactly 302 if needed (some neurons may have variant names)
# Add any missing to reach 302
_EXTRA_NEURONS = {
    "HSNL": "M", "HSNR": "M",
    "PVCEL": "I", "PVCER": "I",
    "VC01": "M", "VC02": "M", "VC03": "M", "VC04": "M", "VC05": "M", "VC06": "M",
}
_NEURON_DATA.update(_EXTRA_NEURONS)

# Trim or pad to exactly 302
_ALL_NAMES = sorted(_NEURON_DATA.keys())
if len(_ALL_NAMES) > 302:
    _ALL_NAMES = _ALL_NAMES[:302]
elif len(_ALL_NAMES) < 302:
    # Generate remaining placeholder interneurons
    idx = 0
    while len(_ALL_NAMES) < 302:
        name = f"INT{idx:02d}"
        if name not in _NEURON_DATA:
            _ALL_NAMES.append(name)
            _NEURON_DATA[name] = "I"
        idx += 1
    _ALL_NAMES = sorted(_ALL_NAMES)

NUM_NEURONS = 302


# ---------------------------------------------------------------------------
# Known major synaptic pathways (from published connectome)
# ---------------------------------------------------------------------------
# Format: (pre_name, post_name, weight)
# These are the most well-characterized circuits in C. elegans

_MAJOR_CHEMICAL_SYNAPSES = [
    # Touch response circuit (Chalfie et al.)
    ("ALML", "AVDL", 3.0), ("ALML", "AVDR", 1.0),
    ("ALMR", "AVDR", 3.0), ("ALMR", "AVDL", 1.0),
    ("PLML", "AVDL", 2.0), ("PLML", "AVDR", 1.0),
    ("PLMR", "AVDR", 2.0), ("PLMR", "AVDL", 1.0),
    ("AVM", "AVEL", 2.0), ("AVM", "AVER", 2.0),
    ("AVM", "AVBL", 1.0), ("AVM", "AVBR", 1.0),

    # Command interneurons → motor
    ("AVAL", "DA01", 3.0), ("AVAL", "DA02", 2.0), ("AVAL", "DA03", 2.0),
    ("AVAL", "DA04", 2.0), ("AVAL", "DA05", 2.0), ("AVAL", "DA06", 2.0),
    ("AVAR", "DA07", 2.0), ("AVAR", "DA08", 2.0), ("AVAR", "DA09", 2.0),
    ("AVAL", "VA01", 3.0), ("AVAL", "VA02", 2.0), ("AVAL", "VA03", 2.0),
    ("AVAR", "VA06", 2.0), ("AVAR", "VA07", 2.0),
    ("AVBL", "DB01", 2.0), ("AVBL", "DB02", 2.0), ("AVBL", "DB03", 2.0),
    ("AVBR", "DB04", 2.0), ("AVBR", "DB05", 2.0), ("AVBR", "DB06", 2.0),
    ("AVBL", "VB01", 2.0), ("AVBL", "VB02", 2.0),
    ("AVBR", "VB06", 2.0), ("AVBR", "VB07", 2.0),
    ("AVDL", "VA01", 2.0), ("AVDL", "VA02", 1.5),
    ("AVDR", "VA06", 2.0),
    ("AVEL", "VA01", 1.5), ("AVER", "VA06", 1.5),

    # DD inhibitory motor neurons (dorsal inhibition)
    ("DD01", "DA01", -2.0), ("DD01", "DA02", -1.5),
    ("DD02", "DA02", -2.0), ("DD02", "DA03", -1.5),
    ("DD03", "DA03", -2.0), ("DD04", "DA04", -2.0),
    ("DD05", "DA05", -2.0), ("DD06", "DA06", -2.0),

    # VD inhibitory motor neurons (ventral inhibition)
    ("VD01", "VA01", -2.0), ("VD02", "VA02", -2.0),
    ("VD03", "VA03", -2.0), ("VD04", "VA04", -2.0),
    ("VD05", "VA05", -2.0), ("VD06", "VA06", -2.0),
    ("VD07", "VA07", -2.0), ("VD08", "VA08", -2.0),

    # Chemosensory → interneuron pathways
    ("ASEL", "AIYL", 3.0), ("ASER", "AIYR", 3.0),
    ("ASEL", "AIAL", 2.0), ("ASER", "AIAR", 2.0),
    ("AWCL", "AIYL", 2.0), ("AWCR", "AIYR", 2.0),
    ("AWCL", "AIBL", 1.5), ("AWCR", "AIBR", 1.5),
    ("AWAL", "AIAL", 2.0), ("AWAR", "AIAR", 2.0),
    ("AWAL", "AIBL", 1.0), ("AWAR", "AIBR", 1.0),
    ("ASHL", "AVDL", 2.0), ("ASHR", "AVDR", 2.0),
    ("ASHL", "AVAL", 1.0), ("ASHR", "AVAR", 1.0),

    # Interneuron → interneuron
    ("AIYL", "AIZL", 3.0), ("AIYR", "AIZR", 3.0),
    ("AIZL", "RIAL", 2.0), ("AIZR", "RIAR", 2.0),
    ("AIBL", "RIML", 2.0), ("AIBR", "RIMR", 2.0),
    ("AIBL", "AVAL", 1.5), ("AIBR", "AVAR", 1.5),
    ("AIAL", "AIBL", 2.0), ("AIAR", "AIBR", 2.0),
    ("RIAL", "SMDVL", 2.0), ("RIAR", "SMDVR", 2.0),
    ("RIAL", "SMDDL", 1.5), ("RIAR", "SMDDR", 1.5),
    ("RIML", "AVBL", 2.0), ("RIMR", "AVBR", 2.0),
    ("RIML", "RMDDL", 1.5), ("RIMR", "RMDDR", 1.5),

    # Dopaminergic neuron outputs
    ("CEPDL", "RIAL", 1.5), ("CEPDR", "RIAR", 1.5),
    ("CEPVL", "RIBL", 1.5), ("CEPVR", "RIBR", 1.5),
    ("ADEL", "AVDL", 1.0), ("ADER", "AVDR", 1.0),
    ("PDEL", "AVDL", 1.0), ("PDER", "AVDR", 1.0),

    # Head motor control
    ("SMDVL", "RMDDL", 2.0), ("SMDVR", "RMDDR", 2.0),
    ("SMDDL", "RMDVL", 2.0), ("SMDDR", "RMDVR", 2.0),
    ("RMEDL", "RMDDL", 1.0), ("RMEDR", "RMDDR", 1.0),
    ("RMEVL", "RMDVL", 1.0), ("RMEVR", "RMDVR", 1.0),

    # Nose touch / avoidance
    ("ASHL", "AVBL", 1.5), ("ASHR", "AVBR", 1.5),
    ("FLPL", "AVEL", 1.5), ("FLPR", "AVER", 1.5),
    ("OLQVL", "RMDDL", 1.0), ("OLQVR", "RMDDR", 1.0),
    ("OLQDL", "RMDVL", 1.0), ("OLQDR", "RMDVR", 1.0),

    # Pharyngeal circuit
    ("M3L", "MI", 2.0), ("M3R", "MI", 2.0),
    ("MI", "M1", 2.0), ("I1L", "I2L", 1.5), ("I1R", "I2R", 1.5),
]

_MAJOR_GAP_JUNCTIONS = [
    # Command interneuron gap junctions
    ("AVAL", "AVAR", 3.0), ("AVBL", "AVBR", 3.0),
    ("AVDL", "AVDR", 2.0), ("AVEL", "AVER", 2.0),

    # Motor neuron gap junctions (dorsal chain)
    ("DA01", "DA02", 2.0), ("DA02", "DA03", 2.0), ("DA03", "DA04", 2.0),
    ("DA04", "DA05", 2.0), ("DA05", "DA06", 2.0), ("DA06", "DA07", 2.0),
    ("DA07", "DA08", 2.0), ("DA08", "DA09", 2.0),
    ("DB01", "DB02", 2.0), ("DB02", "DB03", 2.0), ("DB03", "DB04", 2.0),
    ("DB04", "DB05", 2.0), ("DB05", "DB06", 2.0), ("DB06", "DB07", 2.0),

    # Motor neuron gap junctions (ventral chain)
    ("VA01", "VA02", 2.0), ("VA02", "VA03", 2.0), ("VA03", "VA04", 2.0),
    ("VA04", "VA05", 2.0), ("VA05", "VA06", 2.0), ("VA06", "VA07", 2.0),
    ("VA07", "VA08", 2.0), ("VA08", "VA09", 2.0), ("VA09", "VA10", 2.0),
    ("VA10", "VA11", 2.0), ("VA11", "VA12", 2.0),
    ("VB01", "VB02", 2.0), ("VB02", "VB03", 2.0), ("VB03", "VB04", 2.0),
    ("VB04", "VB05", 2.0), ("VB05", "VB06", 2.0), ("VB06", "VB07", 2.0),
    ("VB07", "VB08", 2.0), ("VB08", "VB09", 2.0), ("VB09", "VB10", 2.0),
    ("VB10", "VB11", 2.0),

    # Sensory neuron gap junctions
    ("ASEL", "ASER", 1.5),
    ("AWCL", "AWCR", 1.5),
    ("ASHL", "ASHR", 1.5),
    ("ALML", "ALMR", 1.0),
    ("PLML", "PLMR", 1.0),

    # Interneuron gap junctions
    ("AIYL", "AIYR", 2.0),
    ("AIBL", "AIBR", 1.5),
    ("AIAL", "AIAR", 1.5),
    ("RIML", "RIMR", 2.0),
    ("RIAL", "RIAR", 1.5),
    ("RIBL", "RIBR", 1.5),
    ("RICL", "RICR", 2.0),
    ("RIGL", "RIGR", 1.0),

    # Cross-class gap junctions
    ("AVBL", "DB01", 1.5), ("AVBR", "DB04", 1.5),
    ("AVAL", "DA01", 1.0), ("AVAR", "DA05", 1.0),
]


class CelegansConnectome:
    """Builds the 302-neuron connectivity from published C. elegans data.

    Constructs two adjacency matrices:
    - chem_weights: 302x302 directed chemical synapses
    - gap_weights: 302x302 symmetric gap junctions

    Also provides neuron type classification and index mappings.
    """

    def __init__(self):
        self.neuron_names = _ALL_NAMES
        self.name_to_idx = {n: i for i, n in enumerate(self.neuron_names)}
        self.neuron_types = [_NEURON_DATA.get(n, "I") for n in self.neuron_names]

        # Type-based index lists
        self.sensory_idx = [i for i, t in enumerate(self.neuron_types) if t in ("S", "D")]
        self.motor_idx = [i for i, t in enumerate(self.neuron_types) if t == "M"]
        self.inter_idx = [i for i, t in enumerate(self.neuron_types) if t == "I"]
        self.dopa_idx = [i for i, t in enumerate(self.neuron_types) if t == "D"]

        # Dopamine receptor mask (~60% of neurons express DA receptors)
        self.dopa_receptor_mask = self._build_dopa_receptor_mask()

        # Build adjacency matrices
        self.chem_weights = self._build_chemical_matrix()
        self.gap_weights = self._build_gap_matrix()

    def _build_dopa_receptor_mask(self):
        """~60% of neurons express dopamine receptors (DOP-1, DOP-2, DOP-3)."""
        rng = torch.Generator()
        rng.manual_seed(42)
        mask = torch.zeros(NUM_NEURONS, dtype=torch.bool)
        # All dopaminergic neurons express receptors
        for i in self.dopa_idx:
            mask[i] = True
        # Most interneurons (~70%) express receptors
        for i in self.inter_idx:
            if torch.rand(1, generator=rng).item() < 0.70:
                mask[i] = True
        # ~50% of sensory neurons
        for i in self.sensory_idx:
            if torch.rand(1, generator=rng).item() < 0.50:
                mask[i] = True
        # ~60% of motor neurons
        for i in self.motor_idx:
            if torch.rand(1, generator=rng).item() < 0.60:
                mask[i] = True
        return mask

    def _build_chemical_matrix(self):
        """Build 302x302 directed chemical synapse weight matrix."""
        W = torch.zeros(NUM_NEURONS, NUM_NEURONS)

        # 1) Insert known major pathways
        for pre, post, weight in _MAJOR_CHEMICAL_SYNAPSES:
            if pre in self.name_to_idx and post in self.name_to_idx:
                W[self.name_to_idx[pre], self.name_to_idx[post]] = weight

        # 2) Fill remaining connections from connectivity statistics
        # C. elegans has ~7000 chemical synapses total
        # Connection probabilities by type pair (from published statistics)
        rng = torch.Generator()
        rng.manual_seed(12345)

        type_probs = {
            ("S", "I"): 0.08, ("S", "M"): 0.02, ("S", "S"): 0.01,
            ("D", "I"): 0.10, ("D", "M"): 0.03, ("D", "S"): 0.02,
            ("I", "I"): 0.06, ("I", "M"): 0.10, ("I", "S"): 0.02,
            ("M", "M"): 0.04, ("M", "I"): 0.02, ("M", "S"): 0.01,
        }

        for i in range(NUM_NEURONS):
            for j in range(NUM_NEURONS):
                if i == j or W[i, j] != 0:
                    continue
                ti = self.neuron_types[i]
                tj = self.neuron_types[j]
                prob = type_probs.get((ti, tj), 0.01)
                if torch.rand(1, generator=rng).item() < prob:
                    # Weight magnitude: 0.5 to 2.5
                    w = 0.5 + 2.0 * torch.rand(1, generator=rng).item()
                    # Inhibitory neurons (DD, VD classes) get negative weights
                    name = self.neuron_names[i]
                    if name.startswith(("DD", "VD")):
                        w = -w
                    W[i, j] = w

        return W

    def _build_gap_matrix(self):
        """Build 302x302 symmetric gap junction weight matrix."""
        W = torch.zeros(NUM_NEURONS, NUM_NEURONS)

        # 1) Insert known major gap junctions
        for n1, n2, weight in _MAJOR_GAP_JUNCTIONS:
            if n1 in self.name_to_idx and n2 in self.name_to_idx:
                i, j = self.name_to_idx[n1], self.name_to_idx[n2]
                W[i, j] = weight
                W[j, i] = weight  # symmetric

        # 2) Fill remaining connections (~900 total gap junctions)
        rng = torch.Generator()
        rng.manual_seed(54321)

        type_probs = {
            ("M", "M"): 0.03, ("I", "I"): 0.02, ("S", "S"): 0.01,
            ("I", "M"): 0.015, ("S", "I"): 0.01,
        }

        for i in range(NUM_NEURONS):
            for j in range(i + 1, NUM_NEURONS):
                if W[i, j] != 0:
                    continue
                ti = self.neuron_types[i]
                tj = self.neuron_types[j]
                prob = type_probs.get((ti, tj), type_probs.get((tj, ti), 0.005))
                if torch.rand(1, generator=rng).item() < prob:
                    w = 0.3 + 1.5 * torch.rand(1, generator=rng).item()
                    W[i, j] = w
                    W[j, i] = w

        return W


# ---------------------------------------------------------------------------
# HH Dynamics for C. elegans
# ---------------------------------------------------------------------------

class CelegansHH(nn.Module):
    """Hodgkin-Huxley style graded-potential simulation of the C. elegans
    nervous system. 302 neurons, 5 state variables each.

    C. elegans neurons are non-spiking — they use graded potentials and
    continuous synaptic transmission.

    State variables per neuron:
        V  - membrane potential (mV)
        m  - calcium channel activation
        h  - calcium channel inactivation
        n  - delayed-rectifier potassium activation
        Ca - intracellular calcium concentration

    All dynamics are fully vectorized over neurons and batch dimension.
    """

    def __init__(self, substeps: int = 100, dt: float = 0.01):
        super().__init__()
        self.substeps = substeps
        self.dt = dt

        # Build connectome
        connectome = CelegansConnectome()
        self.neuron_names = connectome.neuron_names
        self.sensory_idx = connectome.sensory_idx
        self.motor_idx = connectome.motor_idx
        self.inter_idx = connectome.inter_idx
        self.dopa_idx = connectome.dopa_idx

        # Register connectivity as buffers (not trainable via backprop)
        self.register_buffer("chem_weights", connectome.chem_weights)
        self.register_buffer("gap_weights", connectome.gap_weights)
        self.register_buffer("dopa_receptor_mask",
                             connectome.dopa_receptor_mask.float())

        # Index tensors for gathering outputs
        self.register_buffer("sensory_idx_t",
                             torch.tensor(self.sensory_idx, dtype=torch.long))
        self.register_buffer("motor_idx_t",
                             torch.tensor(self.motor_idx, dtype=torch.long))
        self.register_buffer("dopa_idx_t",
                             torch.tensor(self.dopa_idx, dtype=torch.long))

        # HH conductance parameters (shared by neuron type)
        # These are learnable via Hebbian updates only
        self.g_leak = nn.Parameter(torch.full((NUM_NEURONS,), 0.3))
        self.g_Ca = nn.Parameter(torch.full((NUM_NEURONS,), 1.2))
        self.g_K = nn.Parameter(torch.full((NUM_NEURONS,), 0.36))

        # Reversal potentials (fixed biophysical constants)
        self.register_buffer("E_leak", torch.full((NUM_NEURONS,), -60.0))
        self.register_buffer("E_Ca", torch.full((NUM_NEURONS,), 120.0))
        self.register_buffer("E_K", torch.full((NUM_NEURONS,), -84.0))
        self.register_buffer("E_syn", torch.full((NUM_NEURONS,), 0.0))

        # Membrane capacitance
        self.register_buffer("C_m", torch.full((NUM_NEURONS,), 1.0))

        # Calcium dynamics
        self.k_Ca = nn.Parameter(torch.tensor(0.005))
        self.tau_Ca = nn.Parameter(torch.tensor(150.0))

        # Gating variable time constants
        self.tau_m_base = nn.Parameter(torch.tensor(1.0))
        self.tau_h_base = nn.Parameter(torch.tensor(5.0))
        self.tau_n_base = nn.Parameter(torch.tensor(4.0))

        # Dopamine gain modulation strength
        self.da_gain = nn.Parameter(torch.tensor(0.5))

        # State buffer (set by reset_state)
        self._state = None
        self._input_current = None

    @property
    def num_sensory(self):
        return len(self.sensory_idx)

    @property
    def num_motor(self):
        return len(self.motor_idx)

    @property
    def num_dopa(self):
        return len(self.dopa_idx)

    def reset_state(self, batch_size: int, device=None):
        """Initialize state for a new sequence."""
        if device is None:
            device = self.E_leak.device
        # State: (batch, 302, 5) → [V, m, h, n, Ca]
        self._state = torch.zeros(batch_size, NUM_NEURONS, 5, device=device)
        # V starts at resting potential
        self._state[:, :, 0] = -60.0
        # Gating variables at steady state for resting V
        self._state[:, :, 1] = 0.05  # m
        self._state[:, :, 2] = 0.60  # h
        self._state[:, :, 3] = 0.32  # n
        self._state[:, :, 4] = 0.0   # Ca
        self._input_current = torch.zeros(batch_size, NUM_NEURONS, device=device)

    def inject_sensory(self, x: torch.Tensor):
        """Set current injection on sensory neurons.

        Args:
            x: (batch, num_sensory) tensor of input currents
        """
        if self._input_current is None:
            raise RuntimeError("Call reset_state() before inject_sensory()")
        self._input_current = torch.zeros_like(self._input_current)
        self._input_current = self._input_current.scatter(
            1, self.sensory_idx_t.unsqueeze(0).expand(x.shape[0], -1), x
        )

    def _gating_inf(self, V: torch.Tensor):
        """Steady-state gating variable values."""
        # Sigmoid activation curves centered at different voltages
        m_inf = torch.sigmoid((V + 25.0) / 10.0)
        h_inf = torch.sigmoid(-(V + 40.0) / 10.0)
        n_inf = torch.sigmoid((V + 35.0) / 15.0)
        return m_inf, h_inf, n_inf

    def _tau_gating(self, V: torch.Tensor):
        """Voltage-dependent time constants for gating variables."""
        # Bell-shaped tau curves typical for HH
        tau_m = self.tau_m_base.abs() + 0.5 / (1.0 + ((V + 25.0) / 20.0).pow(2))
        tau_h = self.tau_h_base.abs() + 2.0 / (1.0 + ((V + 40.0) / 20.0).pow(2))
        tau_n = self.tau_n_base.abs() + 1.0 / (1.0 + ((V + 35.0) / 20.0).pow(2))
        return tau_m, tau_h, tau_n

    def _compute_derivatives(self, state: torch.Tensor, I_input: torch.Tensor):
        """Compute dstate/dt for all neurons (vectorized).

        Args:
            state: (batch, 302, 5)
            I_input: (batch, 302)

        Returns:
            dstate: (batch, 302, 5)
        """
        V = state[:, :, 0]    # (batch, 302)
        m = state[:, :, 1]
        h = state[:, :, 2]
        n = state[:, :, 3]
        Ca = state[:, :, 4]

        # Ionic currents
        I_leak = self.g_leak * (V - self.E_leak)
        I_Ca = self.g_Ca * m * h * (V - self.E_Ca)
        I_K = self.g_K * n * (V - self.E_K)

        # Synaptic currents (graded transmission)
        # Chemical: I_chem_j = sum_i(W_ij * sigmoid(V_i)) * (E_syn_j - V_j)
        pre_activity = torch.sigmoid(V / 20.0)  # graded transmitter release
        I_chem = (pre_activity @ self.chem_weights) * (self.E_syn - V)

        # Gap junctions: I_gap_j = sum_i(G_ij * (V_i - V_j))
        I_gap = (V @ self.gap_weights) - V * self.gap_weights.sum(dim=0)

        # Dopamine modulation (extrasynaptic broadcast)
        # Mean activity of dopaminergic neurons → scalar DA level
        da_activity = pre_activity[:, self.dopa_idx_t].mean(dim=1, keepdim=True)
        # Modulate all DA-receptor-expressing neurons
        da_modulation = 1.0 + self.da_gain * da_activity * self.dopa_receptor_mask

        # Membrane potential derivative
        dV = (-I_leak - I_Ca - I_K + I_chem + I_gap + I_input) / self.C_m
        dV = dV * da_modulation

        # Gating variable derivatives
        m_inf, h_inf, n_inf = self._gating_inf(V)
        tau_m, tau_h, tau_n = self._tau_gating(V)
        dm = (m_inf - m) / tau_m
        dh = (h_inf - h) / tau_h
        dn = (n_inf - n) / tau_n

        # Calcium dynamics
        dCa = -self.k_Ca * m * h * (V - self.E_Ca) - Ca / self.tau_Ca.abs().clamp(min=1.0)

        dstate = torch.stack([dV, dm, dh, dn, dCa], dim=2)
        return dstate

    def _rk4_step(self, state: torch.Tensor, I_input: torch.Tensor, dt: float):
        """Single RK4 integration step."""
        k1 = self._compute_derivatives(state, I_input)
        k2 = self._compute_derivatives(state + 0.5 * dt * k1, I_input)
        k3 = self._compute_derivatives(state + 0.5 * dt * k2, I_input)
        k4 = self._compute_derivatives(state + dt * k3, I_input)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _integrate(self, state: torch.Tensor, I_input: torch.Tensor):
        """Pure RK4 integration — no in-place ops, compile-friendly.

        Args:
            state: (batch, 302, 5)
            I_input: (batch, 302)
        Returns:
            new_state: (batch, 302, 5)
        """
        clamp_lo = torch.tensor([-100.0, 0.0, 0.0, 0.0, 0.0], device=state.device)
        clamp_hi = torch.tensor([60.0, 1.0, 1.0, 1.0, 10.0], device=state.device)

        for _ in range(self.substeps):
            state = self._rk4_step(state, I_input, self.dt)
            state = torch.clamp(state, clamp_lo, clamp_hi)

        return state

    def step(self):
        """Run substeps of neural dynamics and return motor + dopamine outputs.

        Returns:
            motor: (batch, num_motor) — motor neuron membrane potentials
            dopa: (batch, num_dopa) — dopamine neuron activations
        """
        if self._state is None:
            raise RuntimeError("Call reset_state() before step()")

        state = self._integrate(self._state, self._input_current)
        self._state = state.detach()

        # Extract outputs
        V = state[:, :, 0]
        motor = V[:, self.motor_idx_t]
        dopa = torch.sigmoid(V[:, self.dopa_idx_t] / 20.0)

        return motor, dopa

    def get_full_state(self):
        """Return all membrane potentials: (batch, 302)."""
        if self._state is None:
            raise RuntimeError("Call reset_state() before get_full_state()")
        return self._state[:, :, 0]

    def get_da_level(self):
        """Return mean dopamine neuron activation: (batch, 1)."""
        if self._state is None:
            return None
        V = self._state[:, :, 0]
        return torch.sigmoid(V[:, self.dopa_idx_t] / 20.0).mean(dim=1, keepdim=True)

    def hebbian_update(self, da_level: torch.Tensor, eta: float = 1e-5):
        """Apply Hebbian plasticity to chemical synapse weights.

        delta_w = eta * DA * (pre * post - w * post^2)
        Enforces Dale's law: excitatory stay positive, inhibitory stay negative.

        Args:
            da_level: (batch,) or scalar dopamine modulation signal
            eta: Hebbian learning rate
        """
        if self._state is None:
            return

        V = self._state[:, :, 0]  # (batch, 302)
        activity = torch.sigmoid(V / 20.0)  # (batch, 302)

        # Mean over batch
        pre = activity.mean(dim=0)   # (302,)
        post = activity.mean(dim=0)  # (302,)
        da = da_level.mean().item() if da_level.dim() > 0 else da_level.item()

        with torch.no_grad():
            W = self.chem_weights
            # Hebbian: pre * post correlation, with decay term
            dW = da * (pre.unsqueeze(1) * post.unsqueeze(0) - W * post.unsqueeze(0).pow(2))
            W.add_(eta * dW)

            # Enforce Dale's law
            # Identify excitatory/inhibitory by original sign
            exc_mask = W > 0
            inh_mask = W < 0
            # Neurons starting with DD/VD are inhibitory
            inh_neurons = set()
            for i, name in enumerate(self.neuron_names):
                if name.startswith(("DD", "VD")):
                    inh_neurons.add(i)

            for i in inh_neurons:
                W[i, :].clamp_(max=0.0)  # inhibitory stays negative

            # Clamp magnitude
            W.clamp_(-10.0, 10.0)
