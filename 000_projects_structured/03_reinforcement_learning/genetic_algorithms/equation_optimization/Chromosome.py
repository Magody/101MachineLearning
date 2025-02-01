import random

class Chromosome:
    
    def __init__(self, gens_set: list, size: int):
        self.gens = Chromosome.choose_random_sample(gens_set, size)

    def __str__(self) -> str:
        out = "|"
        for gen in self.gens:
            out += str(gen) + " "

        return out + "|"

    @staticmethod
    def choose_random_sample(array: list, size: int) -> list:
        random_sample = [0 for _ in range(size)]

        for i in range(size):
            random_sample[i] = random.choice(array)

        return random_sample

