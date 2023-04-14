from .LCAODensityCalculatorBase import LCAODensityCalculator
from itertools import islice
import platform
import shutil
import pathlib
import subprocess

calc_defaults = {
    'filebase': 'orca',
    'output_format': 'mkl'
}

qm_defaults = {
    'method': 'PBE',
    'basis_set': 'def2-SVP',
    'multiplicity': 1,
    'charge': 0,
    'keywords': [],
    'blocks': {}
}

def batched(iterable: iter, n:int) -> iter:
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

class ORCADensityCalculator(LCAODensityCalculator):
    xyz_format = 'cartesian'
    provides_output = ('mkl', 'wfn')
    
    def __init__(self, *args, abs_orca_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if abs_orca_path is not None:
            self.abs_orca_path = abs_orca_path
        elif platform.system() == 'Windows':
            self.abs_orca_path = shutil.which('orca.exe')
        elif platform.system() == 'Darwin':
            self.abs_orca_path = shutil.which('orca')
        else:
            #assume linux
            self.abs_orca_path = shutil.which('orca')

    def check_availability(self) -> bool:
        if self.abs_orca_path is not None:
            path = pathlib.Path(self.abs_orca_path)
            return path.exists()
        else:
            return False
    
    def calculate_density(self, elements, xyz, cluster_charge_dict={}):
        # Merge defaults and user-supplied options
        qm_options = qm_defaults.copy()
        
        #blocks and keyword merging needs to account for different cases
        new_lower = [key.lower() for key in self.qm_options['keywords']]
        keep_kw = list(kw for kw in qm_defaults['keywords'] if kw not in new_lower)

        qm_opt_blocks = self.qm_options.get('blocks', {})
        lower_keys = tuple(key.lower() for key in qm_opt_blocks.keys())
        keep_blocks = {key: val for key, val in qm_defaults['blocks'].items()
                        if key.lower() not in lower_keys}

        qm_options.update(self.qm_options)
        qm_options['keywords'] += keep_kw
        qm_options['blocks'].update(keep_blocks)

        calc_options = calc_defaults.copy()
        calc_options.update(self.calc_options)

        if len(cluster_charge_dict.get('charges', [])) > 0:
            cc_file = self._generate_cluster_charge_file(cluster_charge_dict)
            cc_filename = f"{calc_options['filebase']}.pc"
            with open(cc_filename, 'w') as fo:
                fo.write(cc_file)

            qm_options['blocks']['pointcharges'] = "'cc_filename'"
 
        # Create the input file content
        input_content = self._generate_orca_input(elements, xyz, qm_options)

        # Write the input file to disk
        input_filename = f"{calc_options['filebase']}.inp"
        with open(input_filename, 'w') as fo:
            fo.write(input_content)

        #Execute ORCA with the generated input file 
        out_filename = f"{calc_options['filebase']}.out"
        with open(out_filename, 'w') as fo:
            subprocess.call(
                [self.abs_orca_path, input_filename],
                stdout=fo,
                stderr=subprocess.STDOUT
            )

        format_standardise = calc_options['output_format'].lower().replace('.', '')
        if  format_standardise == 'mkl':
            subprocess.check_output(['orca_2mkl', calc_options['filebase']])
        elif format_standardise == 'wfn':
            subprocess.check_output(['orca_2aim', calc_options['filebase']])
        else:
            raise NotImplementedError('output_format from OrcaCalculator is not implemented. Choose either mkl or wfn')

    def _generate_cluster_charge_file(cluster_charge_dict):

        position_strings = iter(
            ' '.join(f'{val: 12.8f}' for val in single_position)
            for single_position in cluster_charge_dict['positions']
        )

        charge_block = '\n'.join(
            f'{charge: 9.6f} {pos_string}' for charge, pos_string 
            in zip(cluster_charge_dict['charges'], position_strings)
        )
        
        return f"{len(cluster_charge_dict['charges'])}\n{charge_block}\n"

    def _generate_orca_input(self, elements, xyz, qm_options):
        # Set up the ORCA input file header
        header = f"! {qm_options['method']} {qm_options['basis_set']}"

        for entries in batched(qm_options['keywords'], 5):
            header += '\n!' + ' '.join(entries)      

        blocks = ''.join(
            f'\n%{key}\n{entry}\nend\n' if ' ' in entry.strip() 
            else f'\n%{key} {entry}\n'
            for key, entry in qm_options['blocks'].items() 
        )

        charge_mult = f"*xyz {qm_options['charge']} {qm_options['multiplicity']}"

        # Generate the coordinates section
        coordinates = [f"{element} {x} {y} {z}" for element, (x, y, z) in zip(elements, xyz)]
        coordinates_section = '\n'.join(coordinates)

        # Combine sections into a complete input file
        orca_input = f"{header}\n{blocks}\n{charge_mult}\n{coordinates_section}\n*\n"
        return orca_input
    
    def cif_output(self) -> str:
        # TODO: Implement the logic to generate a CIF output from the calculation
        return 'Someone needs to implement this before production'


        