from abc import abstractmethod, ABC

class DensityCalculator(ABC):

    @abstractmethod
    def check_availability(self) -> bool:
        pass

    @abstractmethod
    def calculate_density(self, elements, xyz):
        pass

    @abstractmethod
    def cif_output(self) -> str:
        pass