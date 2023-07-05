from textwrap import dedent
import warnings

functional_synonyms = {
    'scanfunc': 'scan'
}

functionals_bibtex = {
    'scan': (
        'SCAN',
        """
            @article{SCAN,
                title = {Strongly Constrained and Appropriately Normed Semilocal Density Functional},
                author = {Sun, Jianwei and Ruzsinszky, Adrienn and Perdew, John P.},
                journal = {Phys. Rev. Lett.},
                volume = {115},
                issue = {3},
                pages = {036402},
                numpages = {6},
                year = {2015},
                month = {Jul},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevLett.115.036402},
                url = {https://link.aps.org/doi/10.1103/PhysRevLett.115.036402}
            }
        """
    ),

    'pbe': (
        'PBE',
        """
            @article{PBE,
                title = {Generalized Gradient Approximation Made Simple},
                author = {Perdew, John P. and Burke, Kieron and Ernzerhof, Matthias},
                journal = {Phys. Rev. Lett.},
                volume = {77},
                issue = {18},
                pages = {3865--3868},
                numpages = {0},
                year = {1996},
                month = {Oct},
                publisher = {American Physical Society},
                doi = {10.1103/PhysRevLett.77.3865},
                url = {https://link.aps.org/doi/10.1103/PhysRevLett.77.3865}
            }
        """
    )
}

def get_functional_citation(name):
    if name.lower() in functional_synonyms:
        name = functional_synonyms[name.lower()]

    if name.lower() in functionals_bibtex:
        bibtex_key, bibtex_entry = functionals_bibtex[name.lower()]
        return bibtex_key, dedent(bibtex_entry).strip()
    
    warnings.warn(f'No library bibtex entry found for {name}. You need to add the source manually.')
    return name + '??', ''

basis_set_synonyms = {
    'def2-svp': 'karlsruhe',
    'def2-tzvp': 'karlsruhe',
    'def2-qzvp': 'karlsruhe',
}

basis_set_bibtex = {
    'karlsruhe': (
        'KarlsruheBasis',
        """
        @Article{B508541A,
            author ="Weigend, Florian and Ahlrichs, Reinhart",
            title  ="Balanced basis sets of split valence{,} triple zeta valence and quadruple zeta valence quality for H to Rn: Design and assessment of accuracy",
            journal  ="Phys. Chem. Chem. Phys.",
            year  ="2005",
            volume  ="7",
            issue  ="18",
            pages  ="3297-3305",
            publisher  ="The Royal Society of Chemistry",
            doi  ="10.1039/B508541A",
            url  ="http://dx.doi.org/10.1039/B508541A",
        }
        """
    )
}

def get_basis_citation(name):
    if name.lower() in basis_set_synonyms:
        name = basis_set_synonyms[name.lower()]

    if name.lower() in basis_set_bibtex:
        bibtex_key, bibtex_entry = basis_set_bibtex[name.lower()]
        return bibtex_key, dedent(bibtex_entry).strip()
    
    warnings.warn(f'No library bibtex entry found for {name}. You need to add the source manually.')
    return name + '??', ''
    
partitioning_bibtex = {
    'hirshfeld': (
        'Hirshfeld',
        """
        @article{Hirshfeld,
            title = {Bonded-atom fragments for describing molecular charge densities},
            journal = {Theoretica chimica acta},
            volume = {44},
            number = {2},
            pages = {129-138},
            year = {1997},
            doi = {10.1007/BF00549096},
            url = {https://doi.org/10.1007/BF00549096},
            author = {Hirshfeld, F. L.}
        }
        """
    )
}