import argparse
from .scaff.scaff_f0j import ScaffF0jSource

def tsc_cli(cif_path, scif_path, block_name, tsc_path):
    f0jeval = ScaffF0jSource.from_settings_cif(scif_path, block_name)
    f0jeval.cif2tsc(cif_path, block_name, tsc_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='qctbx command line interface',
        description='Can be used to do calculations using the qctbx from the command line.'
    )
    parser.add_argument(
        '--cif_path',
        required=True,
        help='Path to a crystallographic information file. Needs to contain a data_<block-name> entry.'
    )

    parser.add_argument(
        '--scif_path',
        required=False,
        default=None,
        help='Path to a settings crystallographic information file. Needs to contain a settings_<block-name> entry. Default: Same as cif_path.'
    )

    parser.add_argument(
        '--block_name',
        required=True,
        help='Name of the block to select in the data and settings cif file(s).'
    )

    parser.add_argument(
        '--tsc_path',
        required=True,
        help='Path for the tsc(b) file to be created. If file ending is .tscb the file will be in binary tsc format'
    )

    args = parser.parse_args()
    if args.scif_path is None:
        scif_path = args.cif_path
    else:
        scif_path = args.scif_path

    tsc_cli(args.cif_path, scif_path, args.block_name, args.tsc_path)
