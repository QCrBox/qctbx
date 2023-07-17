import numpy as np
from collections.abc import Iterable
from typing import Tuple, List, Union, Dict

def parse_tsc_data_line(
        line: str
    ) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """
    Parses a line of TSC data.

    Parameters
    ----------
    line : str
        The line of TSC data to parse.

    Returns
    -------
    tuple
        A tuple containing the indices h, k, l and the array of f0j values.
    """
    h_str, k_str, l_str, *f0j_strs =  line.split()
    f0js = np.array([float(val.split(',')[0]) + 1j * float(val.split(',')[1]) for val in f0j_strs])
    return (int(h_str), int(k_str), int(l_str)), f0js

class TSCFile:
    """
    A class representing a TSC file as defined in doi:10.48550/arXiv.1911.08847

    A TSC file contains atomic form factors for a list of atoms and miller 
    indicees

    You can get data for atoms for example with tsc['C1'] or tsc[['C1', 'C2']]
    currently setting is not implemented this way. All data is represented 
    in the data attribute

    Attributes
    ----------
    header : dict
        A dictionary holding the header information from the TSC file.
    data : dict
        A dictionary mapping tuples (h, k, l) to numpy arrays of f0j values,
        where the ordering of the values is given by the content of the 
        scatterers property / the SCATTERERS entry in the header.
    """

    header = {
        'TITLE': 'generic_tsc',
        'SYMM': 'expanded',
        'SCATTERERS' : ''        
    }
    data = {}
    
    @property
    def scatterers(self) -> List[str]:
        """
        Retrieves scatterers from the TSC file as a list of strings generated
        from the SCATTERERS header entry.

        Returns
        -------
        list
            A list of scatterer names.
        """

        return self.header['SCATTERERS'].strip().split()

    @scatterers.setter
    def scatterers(self, scatterers: Iterable):
        """
        Sets the scatterers in the TSC file.

        The input scatterers are converted to a space-separated string and 
        stored in the header under the key 'SCATTERERS'.

        Parameters
        ----------
        scatterers : iterable
            An iterable of scatterer names.
        """
        self.header['SCATTERERS'] = ' '.join(str(val) for val in scatterers)

    @classmethod
    def from_file(cls, filename: str) -> "TSCFile":
        """
        Constructs a TSCFile object from a file.

        The function reads the TSC file, parses its header and data sections,
        and constructs a TSCFile instance with these data.

        Parameters
        ----------
        filename : str
            The name of the TSC file to read.

        Returns
        -------
        TSCFile
            A TSCFile instance with data loaded from the file.
        """
        with open(filename, 'r') as fo:
            tsc_content = fo.read()
        header_str, data_str = tsc_content.split('DATA:\n')

        header_split = iter(val.split(':') for val in header_str.strip().split('\n'))

        new_obj = cls()
        key = None
        for line_split in header_split:
            if len(line_split) == 2 and key is not None:
                new_obj.header[key] = entry
                key, entry = line_split
            else:
                entry = '\n' + line_split[0]
        new_obj.header[key] = entry

        new_obj.header.update({key: val.strip() for key, val in header_split})

        parsed_iter = iter(parse_tsc_data_line(line) for line in data_str.strip().split('\n'))

        new_obj.data = {hkl: f0js for hkl, f0js in parsed_iter}

        return new_obj

    def to_file(self, filename: str) -> None:
        """
        Writes the TSCFile object to a file.

        The function formats the header and data sections of the TSCFile object
        and writes them to a file. Currently no safety checks are implemented
        SCATTERERS and data need to match

        Parameters
        ----------
        filename : str
            The name of the file to write.
        """
        header_str = '\n'.join(f'{key}: {value}' for key, value in self.header.items())
        data_iter = iter(f"{int(hkl[0])} {int(hkl[1])} {int(hkl[2])} {' '.join(f'{np.real(val):.8e},{np.imag(val):.8e}' for val in values)}" for hkl, values in self.data.items())
        data_str = '\n'.join(data_iter)

        with open(filename, 'w') as fo:
            fo.write(f'{header_str}\nDATA:\n{data_str}\n')
    
    def __getitem__(
        self,
        atom_site_label: Union[str, Iterable]
    ) -> Dict[Tuple[int, int, int], np.ndarray]:
        """
        Retrieves f0j values for a given atom site label.

        The function allows indexing the TSCFile object by atom site label or a 
        list of labels. If the given label is not found among the scatterers,
        a ValueError is raised.

        Parameters
        ----------
        atom_site_label : str or iterable
            The atom site label or a list of labels to retrieve f0j values for.

        Returns
        -------
        dict
            A dictionary where each key is a tuple of indices (h, k, l) and the 
            corresponding value is a numpy array of f0j values for the given
            label(s).

        Raises
        ------
        ValueError
            If an unknown atom site label is used for indexing.
        """
        try:
            if isinstance(atom_site_label, str):
                index = self.scatterers.index(atom_site_label)
                return {hkl: f0js[index] for hkl, f0js in self.data.items()}
            elif isinstance(atom_site_label, Iterable):
                indexes = np.array([self.scatterers.index(label) for label in atom_site_label])
                return {hkl: f0js[indexes] for hkl, f0js in self.data.items()}
            else:
                index = self.scatterers.index(atom_site_label)
                return {hkl: f0js[index] for hkl, f0js in self.data.items()}
        except ValueError:
            if isinstance(atom_site_label, str):
                unknown = [atom_site_label]
            elif isinstance(atom_site_label, Iterable):
                unknown = [label for label in atom_site_label if label not in self.scatterers]
            else:
                unknown = [atom_site_label]
            raise ValueError(f'Unknown atom label(s) used for lookup from TSCFile: {" ".join(unknown)}')
        
