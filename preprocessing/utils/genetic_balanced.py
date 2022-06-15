from statistics import mean
from threading import Thread, Condition, Lock
from time import sleep, time
from IPython.display import clear_output
import random
import copy
import numpy as np

#Funcion generadora para dividir un array en bloques del mismo tamaño
def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i + n]

#Devuelve el valor y el índice del menor o mayor elemento en un array
def i_min_max(arr, flag):
    ma = float(flag)
    i_ma = 0
    for i, v in enumerate(arr):
        if v > ma:
            ma = v
            i_ma = i
    return i_ma, ma


def i_max(arr):
    ma = float('-inf')
    i_ma = 0
    for i, v in enumerate(arr):
        if v > ma:
            ma = v
            i_ma = i
    return i_ma, ma

def i_min(arr):
    mi = float('inf')
    i_mi = 0
    for i, v in enumerate(arr):
        if v < mi:
            mi = v
            i_mi = i
    return i_mi, mi


#Objeto que representa un genotipo
class GenotypeClassReport(object):
    """
    chain: [['img.jpg', {'fish 1': 10}]]
    """
    def __init__(self, image_set, num_images=20, chain=None):
        self._img_set = np.array(image_set)
        
        if chain != None:
            self._chain = chain
        else:
            self._chain = []
            for i in range(num_images):
                self._chain.append(random.choice(image_set))


    #Muta un gen de este genotipo
    def mutate(self):
        chrom = random.choice(range(len(self._chain)))
        self._chain[chrom] = random.choice(self._img_set)


    #Cruza este genotipo con el genotipo pasado por parámetro y devuelve uno
    #nuevo con el resultado del cruce de estos dos
    def cross(self, genotype):
        g1 = copy.deepcopy(self._chain)
        g2 = copy.deepcopy(genotype._chain)

        sl = random.choice(range(len(g1)))
        first = random.choice([0, 1])

        g3 = []
        if first == 0:
            g3 = g1[:sl] + g2[sl:]
        else:
            g3 = g2[:sl] + g1[sl:]

        return GenotypeClassReport(self._img_set, chain=g3)
    
    
    #Devuelve la puntuación total del genotipo que se ha establecido 
    def get_score(self):
        acum = 0
        
        class_dict_stats = {}
        for img, report in self._chain:
            for k, v in report.items():
                if k in class_dict_stats:
                    class_dict_stats[k] += v
                else:
                    class_dict_stats[k] = v
        
        values = np.array(list(class_dict_stats.values()))
        var = np.var(values)
        
        images = np.array([np.where(self._img_set == img)[0][0] for img, _ in self._chain])
        occurences = np.sum(np.bincount(images))

        acum += var * 0.6 #penalty for variance
        acum += occurences * 0.9 #penalty for occurences

        return acum
        
        
    #Cuenta el número de veces que aparece un gen en la cadena entera
    def count(self, chrom):
        return self._chain.count(chrom)
    
    def get_chain(self):
        return self._chain

    def __repr__(self):
        return str(self._chain)
    
    def __str__(self):
        return str(self._chain)

    def __getitem__(self, idx):
        return self._chain[idx]

    def __len__(self):
        return len(self._chain)


#Clase que representa al algoritmo genético
class Genetic(object):
    def __init__(self, images, image_set_size, living_things=8, iterations=50, reproductions=1, probabilistic_repoblation=True, mutations=1, n_threads=4, selection='tournament', tournament_percent=0.3, probabilistic_mutation=True):
        self.IMAGES = images
        
        self.__img_set_size = image_set_size
        self.__living_things = living_things
        self.__iterations = iterations
        self.__reproductions = reproductions
        self.__mutations = mutations
        self.__n_threads = n_threads
        self.__selection = selection
        self.__tournament_percent = tournament_percent
        self.__probabilistic_repoblation = probabilistic_repoblation
        self.__probabilistic_mutation = probabilistic_mutation

        self.fitness_list = []
        self.genotypes = []

        self.best_genotype = None
        self.best_fenotype = None
        self.best_score = float('inf')

        self.repoblate = False

        self.__shit = 0


    def init(self):
        #init population
        for i in range(self.__living_things):
            gen = GenotypeClassReport(self.IMAGES, self.__img_set_size) #initialize gen
            self.genotypes.append(gen)
            self.fitness_list.append(0)


    #Función para utilizar por los hilos que evaluen los genotipos
    @staticmethod
    def evaluation_thread(gen, fitness_list, index):
        fitness_list[index] = gen.get_score()


    #Evaluación de la población, se divide las operaciones en bloques de hilos, el nuemro de hilos
    #será el que se especifique en la variable n_threads
    def evaluation(self):
        i = -1

        #Divide in threads
        for genotype_chunk in chunks(self.genotypes, self.__n_threads):
            running_threads = []
            for gen in genotype_chunk:
                i += 1
                T = Thread(
                        target=Genetic.evaluation_thread,
                        args=(gen, self.fitness_list, i))

                running_threads.append(T)
                T.start()


            for thread in running_threads:
                thread.join()


    #Seleccionamos una parte de la población para cruzarla y crear nuevos genes
    def selection_cross(self):
        #Decidimos si hacemos una repoblación o no
        if not self.repoblate:
            #Seleccionamos los genes a cruzar por torneo
            if self.__selection == 'tournament':
                fitness_samples = int(len(self.fitness_list) * self.__tournament_percent)
                for i in range(self.__reproductions):
                    i_fitnesses = []
                    for n in range(2): #2 fathers, 1 per tournament
                        fitnesses = []
                        for fit in range(fitness_samples): #tournament
                            fitnesses.append(random.choice(self.fitness_list))

                        i_fitnesses.append(i_min(fitnesses)[0]) #select the best in the tournament

                    #reproduce the fathers selected in the tournaments
                    gen1 = self.genotypes[i_fitnesses[0]]
                    gen2 = self.genotypes[i_fitnesses[1]]

                    self.genotypes.append(gen1.cross(gen2))
                    self.fitness_list.append(0)

            elif self.__selection == 'weight_list':
                #Selccionamos los genes a crucar por un método de 'lista pesada' en la que los 
                #Genes con mejor "fit" tendrán más posibilidades de ser seleccionados
                # wl = []
                # for i, fit in enumerate(self.fitness_list):
                #     wl += [i] * int(fit)
                
                # for i in range(self.__reproductions):
                #     #reproduce gen
                #     gen = self.genotypes[random.choice(wl)].cross(self.genotypes[random.choice(wl)])

                #     self.genotypes.append(gen)
                #     self.fitness_list.append(0)
                raise Exception('Not implemented')

        else:
            #Si hay repoblación, solamente está implementado el método de selección por torneo
            #Cambiaremos toda la población por una nueva resultada del cruce de la anterior
            if self.__selection == 'tournament':
                fitness_samples = int(len(self.fitness_list) * self.__tournament_percent)
                repoblation = []
                for i in range(self.__living_things):
                    i_fitnesses = []
                    for n in range(2): #2 fathers, 1 per tournament
                        fitnesses = []
                        for fit in range(fitness_samples): #tournament
                            fitnesses.append(random.choice(self.fitness_list))

                        i_fitnesses.append(i_min(fitnesses)[0]) #select the best in the tournament

                    #reproduce the fathers selected in the tournaments
                    gen1 = self.genotypes[i_fitnesses[0]]
                    gen2 = self.genotypes[i_fitnesses[1]]
                    repoblation.append(gen1.cross(gen2))

                self.genotypes = repoblation

            elif self.__selection == 'weight_list': #With repoblation, weighted list is not implemented
                raise Exception('Not implemented repoblation with weighted list selection.')

    #Mutación de los genes
    def mutation(self):
        for i in range(self.__mutations): #mutamos el numero de veces provisto por la variable "mutations"
            for gen in self.genotypes:
                if self.__probabilistic_mutation and random.choice([0, 1]) == 1:
                    gen.mutate()


    #Aqui decidiremos los genes que continuarán para la siguiente porlbación
    def reselection(self):
        if self.__selection == 'tournament': #Seleccion por torneo
            fitness_samples = int(len(self.fitness_list) * self.__tournament_percent)
            for i in range(self.__reproductions):
                fitnesses = []
                for fit in range(fitness_samples): #tournament
                    fitnesses.append(random.choice(self.fitness_list))

                idx = (i_max(fitnesses)[0]) #select the worst in the tournament

                del self.genotypes[idx]
                del self.fitness_list[idx]

        #Selección por lista pesada donde los peores genes tendrán más posibilidad de ser elegidos
        elif self.__selection == 'weight_list':
            # for i in range(self.__reproductions):
            #     ma = max(self.fitness_list) + 1
            #     wl = []
            #     for i, fit in enumerate(self.fitness_list):
            #         wl += [i] * int(ma - fit)

            #     idx = random.choice(wl)

            #     del self.genotypes[idx]
            #     del self.fitness_list[idx]
            raise Exception('Not implemented')


    #Seleccionamos al mejor gen de la generación como una posible solución,
    #siempre y cuando sea mejor que una solución ya escogida
    def select_solution(self):
        for i, fit in enumerate(self.fitness_list):
            if fit < self.best_score:
                self.best_score = copy.copy(fit)
                self.best_genotype = copy.deepcopy(self.genotypes[i])


    def optimize(self):
        self.init()

        self.evaluation()

        #loop
        progress_size = 50
        print('[>', end='\r')
        _s = 0
        _p = 0
        step = self.__iterations / 50
        for i in range(self.__iterations):
            if self.__probabilistic_repoblation:
                self.repoblate = random.choice([True, False])

            self.selection_cross() #crucamos
            self.mutation() #mutamos
            self.evaluation() #evaluamos

            #si no hay repoblación, seleccionaremos una parte de la población para continuar a la siguiente genreación
            if not self.repoblate:
                self.reselection()

            self.select_solution() #Seleccionamos una posible solución

            #Dibujo de la línea de progreso
            if _s >= step:
                _p += 1
                _s = 0
                clear_output(wait=True)
                print('[{}>{}] {:.2f}%, current best score: {}'.format(''.join(['=']*_p), ''.join([' ']*(progress_size-_p)), _p/progress_size*100, self.best_score),  end='\r')

            _s += 1

        _p += 1
        clear_output(wait=True)
        print('[{}>{}] {:.2f}%'.format(''.join(['=']*_p), ''.join([' ']*(progress_size-_p)), _p/progress_size*100),  end='\r')
        print(' DONE!, best score: {}'.format(self.best_score))