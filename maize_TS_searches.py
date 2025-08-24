from maize.core.interface import Parameter, Output, Input, MultiInput
from maize.core.node import Node
import os
from mlfsm.cos import FreezingString
from mlfsm.opt import CartesianOptimizer
from mlfsm.geom import project_trans_rot
import numpy as np
import ase

class RunMLFSM(Node):
    """
    Run the ML-FSM.
    
    Inputs:
        -Reactant: ase.Atoms
        -Product: ase.Atoms
        -Calculator: ase.calculator
        -run_directory: str
    Outputs:
        -TS Guess: ase.Atoms
    """
    reactant = Input["ase.Atoms"]()
    product = Input["ase.Atoms"]()
    calculator = Input["ASECalculator"]()
    run_directory = Output[str]()
    ts_out = Output[str]()
        
    nnodes_min: Parameter[int] = Parameter(default=18)
    interp: Parameter[str] = Parameter(default="ric")
    ninterp: Parameter[int] = Parameter(default=50)
    method: Parameter[str] = Parameter(default="L-BFGS-B")
    maxls: Parameter[int] = Parameter(default=3)
    maxiter: Parameter[int] = Parameter(default=2)
    dmax: Parameter[float] = Parameter(default=0.05)
    outdir: Parameter[str] = Parameter(default=".")
    
    def get_ts(self, fsm_string) -> ase.Atoms:
        """
        Get the ase.Atoms object of the TS guess from final FSM string
        """
        path = fsm_string.r_string + fsm_string.p_string[::-1]
        energy = np.array(fsm_string.r_energy + fsm_string.p_energy[::-1])
        energy = list(energy - energy.min())
        ts_index = energy.index(max(energy))
        return path[ts_index]
        
    
    def run(self) -> None:
        """
        Runs FSM with specified parameters and inputs
        """

        #Get inputs
        reactant = self.reactant.receive()
        product = self.product.receive()
        calculator = self.calculator.receive()
        
        #Align reactant and product
        _,prod_aligned = project_trans_rot(reactant.get_positions(),product.get_positions())
        product.set_positions(prod_aligned.reshape(-1,3))
        
        #Set up optimizer
        optimizer = CartesianOptimizer(
            calculator,
            self.method.value,
            self.maxiter.value,
            self.maxls.value,
            self.dmax.value
        )

        #Build the string
        string = FreezingString(
            reactant,
            product,
            self.nnodes_min.value,
            self.interp.value,
            self.ninterp.value,
        )

        #Run FSM
        while string.growing:
            string.grow()
            string.optimize(optimizer)
            string.write(self.outdir.value)

        #get TS guess geometry
        ts_guess = self.get_ts(string)

        #send TS guess
        self.ts_out.send(ts_guess)
        self.run_directory.send(self.outdir.value)


class RunPRFO(Node):
    """
    Run QChem PRFO TS search.
    
    Inputs:
        -TS_Guess: ase.Atoms
    Outputs:
        -None
    """

    ts_guess = Input["ASEAtoms"]()
    run_directory = Input[str]()
    
    method: Parameter[str] = Parameter(default = "wb97x-v")
    basis: Parameter[str] = Parameter(default = "def2-tzvp")

    def _writetsqcin(self,structure,filename,chg,mult) -> None:
        """
        Writes and submits a TS.qcin file to run the PRFO

        TODO: Make this significantly more flexible through parameters and inputs where appropriate
        """
        chem_symb = structure.get_chemical_symbols()
        coordinates = structure.get_positions()
        with open(filename,'w') as f:
            f.write(f'$molecule\n{chg} {mult}\n')
            for i in range(len(chem_symb)):
                f.write(chem_symb[i])
                f.write(' ')
                for coord in coordinates[i]:
                    f.write(str(coord))
                    f.write(' ')
                f.write('\n')
            f.write('$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n'
                    'basis {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n@@@\n\n'.format(self.method.value,self.basis.value))
            f.write('$rem\nJOBTYPE       TS\nMETHOD       {}\n'
                    'BASIS       {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\ngeom_opt_hessian read\nscf_guess read\n'
                    'mem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS       FALSE\n$end\n\n$molecule\nread\n$end\n\n@@@\n\n'.format(self.method.value,self.basis.value))
            f.write('$end\n\n$rem\nJOBTYPE       freq\nmethod {}\n'
                    'basis {}\n'
                    'scf_max_cycles 250\ngeom_opt_max_cycles 250\nmem_total 40000\nmem_static 6000\n'
                    'WAVEFUNCTION_ANALYSIS FALSE\n$end\n\n$molecule\nread\n$end\n'.format(self.method.value,self.basis.value))

    def run(self) -> None:
        ts_guess = self.ts_guess.receive()
        run_directory = self.run_directory.receive()
        
        num_threads = Parameter(default=8)

        #some calculators store charges this way
        charge = np.sum(ts_guess.get_initial_charges())
        multiplicity = np.sum(ts_guess.get_initial_magnetic_moments()) + 1
        filename = self.run_directory.value + "ts_guess.qcin"
        _writetsqcin(
            structure = ts_guess,
            filename = filename,
            chg = charge,
            mult = multiplicity
        )
        os.system(f"qchem -nt {self.num_threads.value} {filename} {filename}.out")
            
                     