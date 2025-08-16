from typing import Optional
import os
from maize.core.interface import Parameter, Output, Input
from maize.core.node import Node

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
        if self.workdir.value:
            os.makedirs(self.workdir.value, exist_ok=True)
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

class RunMLFSM(Node):
    """
    Run ML-FSM (UMA-m) starting from optimized ASE Atoms and write stacked vfile*.xyz.

    Steps:
      1) Align product to reactant (rigid trans/rot)
      2) Load UMA-m calculator
      3) Build FreezingString(reactant, product, nnodes_min, interp, ninterp)
      4) Grow + optimize using RIC optimizer
      5) Write string
    """

    reactant: Input["ASEAtoms"] = Input()
    product: Input["ASEAtoms"] = Input()

    # Core FSM controls
    nnodes_min: Parameter[int] = Parameter(default=18)
    interp: Parameter[str] = Parameter(default="ric")
    ninterp: Parameter[int] = Parameter(default=50)
    method: Parameter[str] = Parameter(default="L-BFGS-B")
    maxls: Parameter[int] = Parameter(default=3)
    maxiter: Parameter[int] = Parameter(default=2)
    dmax: Parameter[float] = Parameter(default=0.05)
    interpolate_only: Parameter[bool] = Parameter(default=False)

    # I/O
    workdir: Parameter[str] = Parameter(default="work_fsm")
    vfile_dir_name: Parameter[str] = Parameter(default="fsm_outputs")

    vfile_dir: Output[str] = Output()

    def _align_to(self, A, B):
        """Align structure A to B via rigid transform using project_trans_rot."""
        from mlfsm.geom import project_trans_rot
        Apos = A.get_positions()
        Bpos = B.get_positions()
        _, A_aligned = project_trans_rot(Bpos, Apos)  # project A onto B frame
        A2 = A.copy()
        A2.set_positions(A_aligned.reshape(-1, 3))
        return A2

    def run(self) -> None:
        import os
        import torch
        from fairchem.core import FAIRChemCalculator, pretrained_mlip  # type: ignore [import-not-found]
        from mlfsm.cos import FreezingString
        from mlfsm.opt import InternalsOptimizer
        from ase.io import write

        os.makedirs(self.workdir.value, exist_ok=True)
        vdir = os.path.join(self.workdir.value, self.vfile_dir_name.value)
        os.makedirs(vdir, exist_ok=True)
        self.logger.info(f"MLFSM writing outputs under: {vdir}")
        
        # Receive optimized endpoints
        r = self.reactant.receive()
        p = self.product.receive()

        # Calculator (UMA-m)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device=dev)
        calc = FAIRChemCalculator(predictor, task_name="omol")

        # Align product to reactant
        p_aligned = self._align_to(p, r)

        # Build the string
        string = FreezingString(
            r, p,
            int(self.nnodes_min.value),
            str(self.interp.value),
            int(self.ninterp.value),
        )

        if bool(self.interpolate_only.value):
            string.interpolate(vdir)
        else:
            optimizer = InternalsOptimizer(
                calc,
                str(self.method.value),
                int(self.maxiter.value),
                int(self.maxls.value),
                float(self.dmax.value),
            )
            #run fsm
            while string.growing:
                string.grow()
                string.optimize(optimizer)
                string.write(vdir)

        self.vfile_dir.send(vdir)


class ExtractTSFromVfiles(Node):
    """Pick a TS guess from a directory of vfile*.xyz files.
    Uses your read_vfile() logic on the LAST vfile (sorted order).
    """
    vfiles_dir_in: Input[str] = Input()
    filename_prefix: Parameter[str] = Parameter(default="vfile")
    filename_suffix: Parameter[str] = Parameter(default=".xyz")
    ts_out_path: Parameter[str] = Parameter(default="ts_guess.xyz")
    workdir: Parameter[str] = Parameter(default="work_ts")
    ts_guess_path: Output[str] = Output()

    def _read_vfile(self, path: str):
        from ase.io import read, write  # type: ignore
        frames = read(path, format="xyz", index=":")
        with open(path, "r") as f:
            data = f.readlines()
        energies = []
        for line in data:
            parts = line.split()
            if len(parts) == 2:
                try:
                    energies.append(float(parts[-1]))
                except Exception:
                    pass
        return energies, frames

    def run(self) -> None:
        from glob import glob
        from ase.io import write  # type: ignore
        import os

        vdir = self.vfiles_dir_in.receive()
        os.makedirs(self.workdir.value, exist_ok=True)
        out_path = os.path.join(self.workdir.value, os.path.basename(self.ts_out_path.value))

        # Find the last vfile in sorted order
        files = sorted(
            f for f in glob(os.path.join(vdir, "*"))
            if os.path.basename(f).startswith(self.filename_prefix.value)
            and os.path.basename(f).endswith(self.filename_suffix.value)
        )
        if not files:
            raise RuntimeError(f"No vfiles found in {vdir}")
        vfile = files[-1]

        energies, frames = self._read_vfile(vfile)
        if not energies or not frames:
            raise RuntimeError(f"Could not parse energies/frames from {vfile}")

        # TS index = argmax energy
        ts_idx = max(range(len(energies)), key=lambda i: energies[i])
        write(out_path, frames[ts_idx])
        self.ts_guess_path.send(out_path)

class _PrintPath(Node):
    inp: Input[str] = Input()
    def run(self) -> None:
        p = self.inp.receive()
        self.logger.info("TS guess written to: %s", p)


# build and run
if __name__ == "__main__":
    import argparse, os
    from maize.core.workflow import Workflow

    # Ensure feeder exists (reads a single structure)
    class _FeedAtoms(Node):
        path: Parameter[str] = Parameter()
        out: Output["ASEAtoms"] = Output()
        def run(self) -> None:
            from ase.io import read
            atoms = read(self.path.value)
            self.out.send(atoms)

    parser = argparse.ArgumentParser(description="MAIZE TS workflow with UMA-m and ML-FSM")
    parser.add_argument("--reactant", required=True, help="Path to reactant structure (xyz/traj)")
    parser.add_argument("--product", required=True, help="Path to product structure (xyz/traj)")
    parser.add_argument("--workdir", default="work", help="Root working directory")
    parser.add_argument("--ts-out", default="ts_guess.xyz", help="Filename for TS guess (written in ts workdir)")

    # OptimizeGeometryAtoms
    parser.add_argument("--fmax", type=float, default=0.05, help="FIRE force threshold (eV/Å)")

    # RunMLFSM controls
    parser.add_argument("--interp", choices=["ric","lst","cart"], default="ric")
    parser.add_argument("--nnodes-min", type=int, default=18)
    parser.add_argument("--ninterp", type=int, default=50)
    parser.add_argument("--method", choices=["L-BFGS-B","CG"], default="L-BFGS-B")
    parser.add_argument("--maxls", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=2)
    parser.add_argument("--dmax", type=float, default=0.05)
    parser.add_argument("--interpolate-only", action="store_true")
    parser.add_argument("--vdir-name", default="fsm_outputs", help="Subdirectory name for vfile outputs")

    args = parser.parse_args()

    # Build the graph
    flow = Workflow(name="ts_search_with_mlfsm")

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
        workdir=os.path.join(args.workdir, "fsm"),
        vfile_dir_name=args.vdir_name,
        interp=args.interp,
        nnodes_min=args.nnodes_min,
        ninterp=args.ninterp,
        method=args.method,
        maxls=args.maxls,
        maxiter=args.maxiter,
        dmax=args.dmax,
        interpolate_only=args.interpolate_only,
    ))

    extract = flow.add(ExtractTSFromVfiles, name="extract_ts", parameters=dict(
        workdir=os.path.join(args.workdir, "ts"),
        ts_out_path=args.ts_out,
    ))

    printer = flow.add(_PrintPath, name="print_ts_path")

    # Wire it up
    flow.connect(feedR.out, optR.atoms_in)
    flow.connect(feedP.out, optP.atoms_in)
    flow.connect(optR.atoms_out, mlfsm.reactant)
    flow.connect(optP.atoms_out, mlfsm.product)
    flow.connect(mlfsm.vfile_dir, extract.vfiles_dir_in)
    flow.connect(extract.ts_guess_path, printer.inp)

    # Run
    flow.check()
    flow.execute()

    print("=== Finished ===")
    print(f"Vfiles directory: {os.path.join(args.workdir, 'fsm', args.vdir_name)}")
    print(f"TS guess path:    {os.path.join(args.workdir, 'ts', os.path.basename(args.ts_out))}")
