from typing import Optional
import os
from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node
from maize_TS_searches import RunMLFSM, RunPRFO

class _FeedCalculator(Node):
    calculator = Parameter[str]()
    out = Output["ASECalculator"]()
    def run(self) -> None:
        # Load calculator
        if self.calculator.value == "qchem":
            from ase.calculators.qchem import QChem

            calc = QChem(
                label="fsm",
                method="wb97x-v",
                basis="def2-tzvp",
                charge=chg,
                multiplicity=mult,
                sym_ignore="true",
                symmetry="false",
                scf_algorithm="diis_gdm",
                scf_max_cycles="500",
                nt=nt,
            )
        elif self.calculator.value == "xtb":
            from xtb.ase.calculator import XTB  # type: ignore [import-not-found]

            calc = XTB(method="GFN2-xTB")
        elif self.calculator.value == "uma_s":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=dev)
            calc = FAIRChemCalculator(predictor, task_name="omol")
        elif self.calculator.value == "uma_m":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
            calc = FAIRChemCalculator(predictor, task_name="omol")
        elif self.calculator.value == "eSEN":
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device=dev)
            calc = FAIRChemCalculator(predictor)
        elif self.calculator.value == "aimnet2":
            from aimnet2calc import AIMNet2ASE  # type: ignore [import-not-found]

            calc = AIMNet2ASE("aimnet2", charge=chg, mult=mult)
        elif self.calculator.value == "emt":
            from ase.calculators.emt import EMT

            calc = EMT()
        elif self.calculator.value == "maceomol":
            import torch
            from mace.calculators import mace_omol

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            calc = mace_omol(model="extra_large",device=dev)
        else:
            raise ValueError(f"Unknown calculator {calculator}")
        
        self.out.send(calc)

class _FeedAtoms(Node):
    path: Parameter[str] = Parameter()
    out: Output["ASEAtoms"] = Output()
    def run(self) -> None:
        from ase.io import read
        atoms = read(self.path.value)
        self.out.send(atoms)

class OptimizeGeometryAtoms(Node):
    """
    Receive ASE Atoms and return optimized ASE Atoms.
    """
    atoms_in: Input["ASEAtoms"] = Input()
    method: Parameter[str] = Parameter(default="uma")
    fmax: Parameter[float] = Parameter(default=0.05)
    workdir: Parameter[str] = Parameter(default="work_opt")
    atoms_out: Output["ASEAtoms"] = Output()
    def run(self) -> None:
        from ase.io import read, write  # lazy import
        atoms = self.atoms_in.receive()
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]
        from ase.optimize import FIRE

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")

        # Use node parameter (default 0.05 eV/Å)
        fmax = float(self.fmax.value)

        def _optimize(f_max: float, structure, calc):
            structure = structure.copy()
            structure.calc = calc
            dyn = FIRE(structure)
            dyn.run(fmax=f_max)
            return structure

        optimized = _optimize(fmax, atoms, calc)
        self.atoms_out.send(optimized)


# build and run
if __name__ == "__main__":
    import argparse, os
    from maize.core.workflow import Workflow

    parser = argparse.ArgumentParser(description="MAIZE TS workflow with UMA-m and ML-FSM")
    parser.add_argument("--reactant", required=True, help="Path to reactant structure (xyz/traj)")
    parser.add_argument("--product", required=True, help="Path to product structure (xyz/traj)")
    parser.add_argument("--workdir", default="work", help="Root working directory")
    parser.add_argument("--ts-out", default="ts_guess.xyz", help="Filename for TS guess (written in ts workdir)")

    # OptimizeGeometryAtoms
    parser.add_argument("--fmax", type=float, default=0.05, help="FIRE force threshold (eV/Å)")

    # RunMLFSM controls
    parser.add_argument("--interp", choices=["ric","lst","cart"], default="ric")
    parser.add_argument("--calculator", choices=["uma_s","uma_m","eSEN"], default="uma_m")
    parser.add_argument("--nnodes-min", type=int, default=18)
    parser.add_argument("--ninterp", type=int, default=50)
    parser.add_argument("--method", choices=["L-BFGS-B","CG"], default="L-BFGS-B")
    parser.add_argument("--maxls", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--dmax", type=float, default=0.05)
    parser.add_argument("--interpolate-only", action="store_true")
    parser.add_argument("--outdir",default=".")
    args = parser.parse_args()

    # Build the graph
    flow = Workflow(name="ts_search_with_mlfsm")
    feedCalc = flow.add(_FeedCalculator, name = "feed_calculator", parameters=dict(calculator=args.calculator))
    feedR = flow.add(_FeedAtoms, name="feed_reactant", parameters=dict(path=args.reactant))
    feedP = flow.add(_FeedAtoms, name="feed_product", parameters=dict(path=args.product))

    optR = flow.add(OptimizeGeometryAtoms, name="opt_reactant", parameters=dict(
        fmax=args.fmax,
        workdir=os.path.join(args.workdir, "opt_reactant"),
    ))
    optP = flow.add(OptimizeGeometryAtoms, name="opt_product", parameters=dict(
        fmax=args.fmax,
        workdir=os.path.join(args.workdir, "opt_product"),
    ))

    mlfsm = flow.add(RunMLFSM, name="mlfsm_run", parameters=dict(
        interp=args.interp,
        nnodes_min=args.nnodes_min,
        ninterp=args.ninterp,
        method=args.method,
        maxls=args.maxls,
        maxiter=args.maxiter,
        dmax=args.dmax,
        outdir=args.outdir
    ))

    prfo = flow.add(RunPRFO, name="run_prfo")

    # Wire it up
    flow.connect(feedR.out, optR.atoms_in)
    flow.connect(feedP.out, optP.atoms_in)
    flow.connect(feedCalc.out, mlfsm.calculator)
    flow.connect(optR.atoms_out, mlfsm.reactant)
    flow.connect(optP.atoms_out, mlfsm.product)
    flow.connect(mlfsm.ts_out, prfo.ts_guess)
    flow.connect(mlfsm.run_directory, prfo.run_directory)

    # Run
    flow.check()
    flow.execute()

