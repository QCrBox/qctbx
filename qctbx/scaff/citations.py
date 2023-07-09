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
    ),
    'hirshfeld-i': (
        'Hirshfeld-I',
        """"
        @article{Hirshfeld-I,
            author = {Bultinck, Patrick and Van Alsenoy, Christian and Ayers, Paul W. and Carbó-Dorca, Ramon},
            title = "{Critical analysis and extension of the Hirshfeld atoms in molecules}",
            journal = {The Journal of Chemical Physics},
            volume = {126},
            number = {14},
            pages = {144111},
            year = {2007},
            month = {04},
            abstract = "{The computational approach to the Hirshfeld [Theor. Chim. Acta 44, 129 (1977)] atom in a molecule is critically investigated, and several difficulties are highlighted. It is shown that these difficulties are mitigated by an alternative, iterative version, of the Hirshfeld partitioning procedure. The iterative scheme ensures that the Hirshfeld definition represents a mathematically proper information entropy, allows the Hirshfeld approach to be used for charged molecules, eliminates arbitrariness in the choice of the promolecule, and increases the magnitudes of the charges. The resulting “Hirshfeld-I charges” correlate well with electrostatic potential derived atomic charges.}",
            issn = {0021-9606},
            doi = {10.1063/1.2715563},
            url = {https://doi.org/10.1063/1.2715563},
            eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.2715563/15397044/144111\_1\_online.pdf},
        }
        """
    ),
    'iterstockholder': (
        'IterStockholder',
        """
        @article{IterStockholder,
            author ="Lillestolen, Timothy C. and Wheatley, Richard J.",
            title  ="Redefining the atom: atomic charge densities produced by an iterative stockholder approach",
            journal  ="Chem. Commun.",
            year  ="2008",
            issue  ="45",
            pages  ="5909-5911",
            publisher  ="The Royal Society of Chemistry",
            doi  ="10.1039/B812691G",
            url  ="http://dx.doi.org/10.1039/B812691G",
        }
        """
    ),
    'mbis': (
        'MBIS',
        """
        @article{MBIS,
            title={Minimal basis iterative stockholder: atoms in molecules for force-field development},
            author={Verstraelen, Toon and Vandenbrande, Steven and Heidar-Zadeh, Farnaz and Vanduyfhuys, Louis and Van Speybroeck, Veronique and Waroquier, Michel and Ayers, Paul W},
            journal={Journal of Chemical Theory and Computation},
            volume={12},
            number={8},
            pages={3894--3912},
            year={2016},
            publisher={ACS Publications},
            url={https://doi.org/10.1021/acs.jctc.6b00456},
            doi={10.1021/acs.jctc.6b00456}
        }
        """
    )
}

partitioning_synonyms = {}

def get_partitioning_citation(name):
    if name.lower() in partitioning_synonyms:
        name = partitioning_synonyms[name.lower()]

    if name.lower() in partitioning_bibtex:
        bibtex_key, bibtex_entry = partitioning_bibtex[name.lower()]
        return bibtex_key, dedent(bibtex_entry).strip()
    
    warnings.warn(f'No library bibtex entry found for {name}. You need to add the source manually.')
    return name + '??', ''