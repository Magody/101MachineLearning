import pygame
import sys
import random
import math
import numpy as np

# Inicialización de Pygame
pygame.init()

# Dimensiones de la ventana
WIDTH, HEIGHT = 800, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sistema Multiagente de Gestión de Residuos Urbanos")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 100, 255)
RED = (200, 0, 0)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# FPS
FPS = 60
clock = pygame.time.Clock()

# Definición del Entorno Urbano
class CityEnvironment:
    def __init__(self, num_points):
        self.num_points = num_points
        self.waste_points = self.generate_waste_points(num_points)
        self.centers = self.generate_centers()
        self.grid_size = 20  # Tamaño de la cuadrícula para el algoritmo A*
        
        # Create classification agents associated with each center
        self.classification_agents = [
            ClassificationAgent(name="ClassificationAgent_Orgánico", x=90, y=100, associated_center=self.centers[0], color=BLACK),
            ClassificationAgent(name="ClassificationAgent_Inorgánico", x=390, y=100, associated_center=self.centers[1], color=BLACK),
            ClassificationAgent(name="ClassificationAgent_Otro", x=690, y=100, associated_center=self.centers[2], color=BLACK)
        ]
        
        # Pass classification agents to CentralStation
        self.central_station = CentralStation(classification_agents=self.classification_agents)

    def generate_waste_points(self, num_points):
        waste_points = []
        for _ in range(num_points):
            point = WastePoint(
                x=random.randint(50, WIDTH - 50),
                y=random.randint(50, HEIGHT - 50),
                waste_type=random.choice(['orgánico', 'inorgánico', 'otro']),
                weight=1  # random.randint(1, 1)  # Peso entre 1 y 5 unidades
            )
            waste_points.append(point)
        return waste_points

    def generate_centers(self):
        centers = [
            TreatmentCenter(x=100, y=100, waste_type='orgánico', color=GREEN),
            TreatmentCenter(x=400, y=100, waste_type='inorgánico', color=BLUE),
            TreatmentCenter(x=700, y=100, waste_type='otro', color=YELLOW)
        ]
        return centers

    def draw(self, window):
        for point in self.waste_points:
            point.draw(window)
        for center in self.centers:
            center.draw(window)
        self.central_station.draw(window)

# Puntos de Residuos
class WastePoint:
    def __init__(self, x, y, waste_type, weight):
        self.x = x
        self.y = y
        self.waste_type = waste_type
        self.weight = weight
        self.collected = False
        self.reserved = False

    def draw(self, window):
        if not self.collected:
            # if self.reserved:
            #     # color = GRAY  # Residuos reservados
            #     pass
            if self.waste_type == 'orgánico':
                color = GREEN
            elif self.waste_type == 'inorgánico':
                color = BLUE
            else:
                color = YELLOW
            pygame.draw.circle(window, color, (self.x, self.y), 5)

# Centros de Tratamiento
class TreatmentCenter:
    def __init__(self, x, y, waste_type, color):
        self.x = x
        self.y = y
        self.color = color
        self.waste_type = waste_type
        self.capacity = 10000  # Capacidad máxima
        self.received_waste = []  # Residuos almacenados

    def draw(self, window):
        # Dibujar una casa sencilla
        pygame.draw.rect(window, self.color, (self.x - 10, self.y, 20, 20))  # Cuerpo de la casa
        pygame.draw.polygon(window, BROWN, [(self.x - 15, self.y), (self.x + 15, self.y), (self.x, self.y - 20)])  # Techo
        # Dibujar el texto del tipo de residuo
        font = pygame.font.SysFont('Arial', 14)
        text_surface = font.render(self.waste_type.capitalize(), True, BLACK)
        window.blit(text_surface, (self.x - 20, self.y - 30))

# Estación Central
class CentralStation:
    def __init__(self, classification_agents):
        self.transport_agent_state = 'waiting'  # 'waiting', 'moving_to_center', 'returning'
        self.transport_agent_position = (WIDTH // 2, HEIGHT // 2)  # Posición inicial del camión
        self.classification_agents = classification_agents  # Agentes de clasificación asociados

    def update_state(self, state):
        self.transport_agent_state = state

    def get_state(self):
        return self.transport_agent_state

    def assign_waste_to_classification(self, wastes):
        """Asignar residuos al agente clasificador adecuado"""
        for waste in wastes:
            for agent in self.classification_agents:
                if agent.associated_center.waste_type == waste.waste_type:
                    agent.receive_waste([waste])
                    print(f"CentralStation asignó un residuo de tipo {waste.waste_type} al {agent.name}.")
                    break
            else:
                print(f"CentralStation no encontró un agente adecuado para el residuo de tipo {waste.waste_type}.")

    def draw(self, window):
        pass  # No es necesario dibujar la estación central

# Clase base para agentes
class Agent:
    def __init__(self, x, y, environment, name):
        self.x = x
        self.y = y
        self.environment = environment
        self.path = []
        self.speed = 2
        self.name = name

    def move(self):
        if self.path:
            next_pos = self.path[0]
            dx = next_pos[0] - self.x
            dy = next_pos[1] - self.y
            dist = math.hypot(dx, dy)
            if dist < self.speed:
                self.x, self.y = next_pos
                self.path.pop(0)
            else:
                self.x += dx / dist * self.speed
                self.y += dy / dist * self.speed

    def draw(self, window):
        pass

# Agente de Recolección
class CollectionAgent(Agent):
    def __init__(self, x, y, environment, name):
        super().__init__(x, y, environment, name)
        self.state = 'idle'  # 'idle', 'moving_to_waste', 'collecting', 'moving_to_truck', 'delivering', 'waiting_for_truck'
        self.target = None
        self.collected_waste = []

    def perceive(self):
        # Percibe los puntos de residuos no recolectados
        self.visible_waste = [point for point in self.environment.waste_points if not point.collected]
        # Percibe el estado del agente de transporte
        self.transport_state = self.environment.central_station.get_state()
        # Percibe la posición del camión
        self.transport_position = self.environment.central_station.transport_agent_position

    def decide(self):
        if self.state == 'idle' and self.visible_waste:
            # Selecciona el residuo más cercano no reservado
            unreserved_waste = [p for p in self.visible_waste if not p.reserved]
            if unreserved_waste:
                self.target = min(unreserved_waste, key=lambda p: self.distance((self.x, self.y), (p.x, p.y)))
                self.target.reserved = True  # Reserva el punto de residuo
                # Calcula la ruta utilizando A*
                self.path = self.a_star_search((self.x, self.y), (self.target.x, self.target.y))
                self.state = 'moving_to_waste'

        elif self.state == 'moving_to_waste' and not self.path:
            self.state = 'collecting'

        elif self.state == 'collecting':
            self.target.collected = True
            self.target.reserved = False  # Libera la reserva
            self.collected_waste.append(self.target)
            print(f"{self.name} recogió un residuo de tipo {self.target.waste_type}.")  # Depuración
            self.target = None
            # Después de recolectar, ir al camión si está disponible
            if self.transport_state == 'waiting':
                truck_position = self.environment.central_station.transport_agent_position
                self.path = self.a_star_search((self.x, self.y), truck_position)
                self.state = 'moving_to_truck'
            else:
                self.state = 'waiting_for_truck'

        elif self.state == 'waiting_for_truck':
            # Espera hasta que el camión esté disponible
            if self.transport_state == 'waiting':
                truck_position = self.environment.central_station.transport_agent_position
                self.path = self.a_star_search((self.x, self.y), truck_position)
                self.state = 'moving_to_truck'

        elif self.state == 'moving_to_truck' and not self.path:
            
            # Solo entrega si el camión puede recibir residuos
            if transport_agent.current_load < transport_agent.capacity:
                self.state = 'delivering'
            else:
                self.state = 'waiting_for_truck'

        elif self.state == 'delivering':
            # Entrega los residuos al camión
            transport_agent.receive_waste(self.collected_waste)
            print(f"{self.name} entregó {len(self.collected_waste)} residuos al camión.")
            self.collected_waste = []
            self.state = 'idle'

    def act(self):
        if self.state in ['moving_to_waste', 'moving_to_truck']:
            self.move()

    def draw(self, window):
        # Dibujar una figura humana sencilla
        pygame.draw.circle(window, BLACK, (int(self.x), int(self.y - 10)), 5)  # Cabeza
        pygame.draw.line(window, BLACK, (int(self.x), int(self.y - 5)), (int(self.x), int(self.y + 10)), 2)  # Cuerpo
        pygame.draw.line(window, BLACK, (int(self.x), int(self.y + 10)), (int(self.x - 5), int(self.y + 20)), 2)  # Pierna izquierda
        pygame.draw.line(window, BLACK, (int(self.x), int(self.y + 10)), (int(self.x + 5), int(self.y + 20)), 2)  # Pierna derecha
        pygame.draw.line(window, BLACK, (int(self.x), int(self.y)), (int(self.x - 5), int(self.y + 5)), 2)  # Brazo izquierdo
        pygame.draw.line(window, BLACK, (int(self.x), int(self.y)), (int(self.x + 5), int(self.y + 5)), 2)  # Brazo derecho

        # Posición de la mano derecha
        hand_right_x = int(self.x + 5)
        hand_right_y = int(self.y + 5)

        # Dibujar los residuos en la mano derecha
        offset_step = 5  # Espaciado entre residuos
        for i, waste in enumerate(self.collected_waste):
            # Calcular posición relativa para cada residuo
            pos_x = hand_right_x + (i % 2) * offset_step
            pos_y = hand_right_y + (i // 2) * offset_step

            # Asignar color según el tipo de residuo
            if waste.waste_type == 'orgánico':
                color = GREEN
            elif waste.waste_type == 'inorgánico':
                color = BLUE
            else:
                color = YELLOW

            # Dibujar el residuo como un pequeño círculo
            pygame.draw.circle(window, color, (pos_x, pos_y), 7)
            
    def distance(self, pos1, pos2):
        return math.hypot(pos1[0]-pos2[0], pos1[1]-pos2[1])

    def a_star_search(self, start, goal):
        # Implementación simplificada del algoritmo A*
        grid_size = self.environment.grid_size
        start_grid = (int(start[0] // grid_size), int(start[1] // grid_size))
        goal_grid = (int(goal[0] // grid_size), int(goal[1] // grid_size))

        frontier = []
        frontier.append(start_grid)
        came_from = {}
        cost_so_far = {}
        came_from[start_grid] = None
        cost_so_far[start_grid] = 0

        while frontier:
            current = frontier.pop(0)

            if current == goal_grid:
                break

            neighbors = self.get_neighbors(current)
            for next in neighbors:
                new_cost = cost_so_far[current] + self.distance_grid(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal_grid, next)
                    frontier.append(next)
                    came_from[next] = current

        # Reconstruir el camino
        current = goal_grid
        path = []
        while current != start_grid:
            x = current[0] * grid_size + grid_size // 2
            y = current[1] * grid_size + grid_size // 2
            path.insert(0, (x, y))
            current = came_from[current]
            if current is None:
                break
        return path

    def get_neighbors(self, grid_pos):
        neighbors = []
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        for dir in dirs:
            nx = grid_pos[0] + dir[0]
            ny = grid_pos[1] + dir[1]
            if 0 <= nx * self.environment.grid_size < WIDTH and 0 <= ny * self.environment.grid_size < HEIGHT:
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def distance_grid(self, a, b):
        return 1  # En cuadrícula, cada movimiento tiene un coste uniforme

# Agente de Transporte
class TransportAgent(Agent):
    def __init__(self, x, y, environment, capacity):
        super().__init__(x, y, environment, "TransportAgent")
        self.state = 'waiting'  # Estados posibles: 'waiting', 'moving_to_center', 'returning'
        self.capacity = capacity  # Capacidad máxima del camión
        self.current_load = 0  # Residuos recogidos
        self.collected_waste = []  # Lista de residuos actuales
        self.environment.central_station.transport_agent_position = (self.x, self.y)  # Actualiza la posición en la estación central
        self.target_centers = []  # Centros a visitar
        self.current_center_index = 0  # Índice del centro objetivo

    def deliver_waste_to_station(self):
        """Entregar residuos a la estación central"""
        for waste in self.collected_waste:
            self.environment.central_station.assign_waste_to_classification([waste])  # Asegurarse de pasar una lista
        print(f"TransportAgent entregó {len(self.collected_waste)} residuos a la estación central.")
        self.collected_waste = []
        self.current_load = 0

        # Obtener el agente de clasificación asociado a cada tipo de residuo entregado
        delivered_types = set(waste.waste_type for waste in self.collected_waste)
        for waste_type in delivered_types:
            classification_agent = self.get_classification_agent_by_type(waste_type)
            if classification_agent:
                # Verificar si se entregaron residuos
                if any(waste.waste_type == waste_type for waste in self.collected_waste):
                    classification_agent.trigger_blink(success=True)
                else:
                    classification_agent.trigger_blink(success=False)

    def get_classification_agent(self, center):
        """Obtener el agente clasificador asociado a un centro específico"""
        for agent in self.environment.central_station.classification_agents:
            if agent.associated_center == center:
                return agent
        return None

    def perceive(self):
        """Percibir el entorno y actualizar el estado"""
        self.environment.central_station.update_state(self.state)
        self.environment.central_station.transport_agent_position = (self.x, self.y)

        if self.state == 'waiting' and self.current_load == self.capacity:
            # Inicia la visita a los centros de tratamiento
            self.target_centers = self.environment.centers
            self.current_center_index = 0
            self.visit_next_center()

        elif self.state == 'moving_to_center' and not self.path:
            # Llega al centro de tratamiento
            self.state = 'delivering'

        elif self.state == 'delivering':
            # Entregar residuos a los clasificadores
            if self.current_center_index >= len(self.target_centers):
                print("Error: No hay más centros para visitar.")
                self.state = 'returning'
                return

            current_center = self.target_centers[self.current_center_index]
            classification_agent = self.get_classification_agent(current_center)
            
            # Filtrar residuos para el centro actual
            wastes_for_center = [
                waste for waste in self.collected_waste if waste.waste_type == current_center.waste_type
            ]
            if wastes_for_center:
                # Asignar residuos al agente clasificador
                self.environment.central_station.assign_waste_to_classification(wastes_for_center)
                # Remover residuos ya entregados
                self.collected_waste = [
                    waste for waste in self.collected_waste if waste.waste_type != current_center.waste_type
                ]
                self.current_load = len(self.collected_waste)

                print(f"TransportAgent entregó residuos al centro de tratamiento {current_center.waste_type}.")
                
                # Notificar al ClassificationAgent que la entrega fue exitosa
                classification_agent.trigger_blink(success=True)
        
            else:
                print(f"No hay residuos de tipo {current_center.waste_type} para entregar al centro de tratamiento.")
                
                # Notificar al ClassificationAgent que no hubo residuos para entregar
                classification_agent.trigger_blink(success=False)

            
            # Si aún quedan residuos, ir al siguiente centro
            if self.current_load > 0 and self.current_center_index < len(self.target_centers) - 1:
                self.current_center_index += 1
                self.visit_next_center()
            else:
                # Regresa al punto inicial
                self.path = self.a_star_search((self.x, self.y), (WIDTH // 2, HEIGHT // 2))
                self.state = 'returning'

        elif self.state == 'returning' and not self.path:
            # Regresa al estado de espera
            self.state = 'waiting'

    def receive_waste(self, waste):
        """Recibir residuos en el camión"""
        total_items = self.current_load + len(waste)
        if total_items <= self.capacity:
            self.collected_waste.extend(waste)
            self.current_load = total_items
            for w in waste:
                print(f"TransportAgent recibió residuo de tipo {w.waste_type}.")
            print(f"Carga actual: {self.current_load}/{self.capacity}")
            # Si alcanza la capacidad máxima, iniciar el proceso de entrega
            if self.current_load == self.capacity:
                self.state = 'moving_to_center'
                self.visit_next_center()
        else:
            print("El camión alcanzó su capacidad máxima y no puede cargar más residuos.")

    def act(self):
        """Actuar según el estado actual"""
        if self.state in ['moving_to_center', 'returning']:
            self.move()

    def visit_next_center(self):
        """Moverse al siguiente centro de tratamiento"""
        if self.current_center_index < len(self.target_centers):
            next_center = self.target_centers[self.current_center_index]
            self.path = self.a_star_search((self.x, self.y), (next_center.x, next_center.y))
            self.state = 'moving_to_center'
        else:
            print("Error: Índice del centro de tratamiento fuera de rango.")
            self.state = 'returning'

    def draw(self, window):
        """Dibujar el camión en la pantalla"""
        pygame.draw.rect(window, ORANGE, (int(self.x) - 10, int(self.y) - 5, 20, 10))  # Cuerpo del camión
        pygame.draw.rect(window, GRAY, (int(self.x) - 15, int(self.y) - 5, 5, 10))  # Cabina
        pygame.draw.circle(window, BLACK, (int(self.x) - 5, int(self.y) + 5), 3)  # Rueda trasera
        pygame.draw.circle(window, BLACK, (int(self.x) + 5, int(self.y) + 5), 3)  # Rueda delantera
        
        DARK_GRAY = (30, 30, 30)
        # Definir posición y tamaño del fondo para los residuos
        background_width = 40  # Ancho del fondo
        background_height = 15  # Alto del fondo
        waste_offset_x = int(self.x) - background_width // 2  # Centrar el fondo sobre el camión
        waste_offset_y = int(self.y) - 20  # Ajustar la posición vertical del fondo

        # Dibujar el fondo oscuro detrás de los residuos
        background_rect = pygame.Rect(waste_offset_x, waste_offset_y, background_width, background_height)
        pygame.draw.rect(window, DARK_GRAY, background_rect)
        pygame.draw.rect(window, WHITE, background_rect, 1)  # Opcional: borde blanco para resaltar el fondo

        # Dibujar los residuos cargados en el camión
        offset_step = 10  # Espaciado entre residuos para mayor visibilidad

        for i, waste in enumerate(self.collected_waste):
            # Calcular posición relativa para cada residuo
            row = i // 4  # Número de fila (asumiendo máximo 4 columnas)
            col = i % 4   # Número de columna
            pos_x = waste_offset_x + 5 + col * offset_step  # +5 para margen izquierdo
            pos_y = waste_offset_y + 5 + row * offset_step  # +5 para margen superior

            # Asignar color según el tipo de residuo
            if waste.waste_type == 'orgánico':
                color = GREEN
            elif waste.waste_type == 'inorgánico':
                color = BLUE
            else:
                color = YELLOW

            # Dibujar el residuo como un pequeño círculo
            pygame.draw.circle(window, color, (pos_x, pos_y), 5)

    def distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def a_star_search(self, start, goal):
        """Implementación simplificada del algoritmo A*"""
        grid_size = self.environment.grid_size
        start_grid = (int(start[0] // grid_size), int(start[1] // grid_size))
        goal_grid = (int(goal[0] // grid_size), int(goal[1] // grid_size))

        frontier = []
        frontier.append(start_grid)
        came_from = {}
        cost_so_far = {}
        came_from[start_grid] = None
        cost_so_far[start_grid] = 0

        while frontier:
            current = frontier.pop(0)

            if current == goal_grid:
                break

            neighbors = self.get_neighbors(current)
            for next in neighbors:
                new_cost = cost_so_far[current] + self.distance_grid(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal_grid, next)
                    frontier.append(next)
                    came_from[next] = current

        # Reconstruir el camino
        current = goal_grid
        path = []
        while current != start_grid:
            x = current[0] * grid_size + grid_size // 2
            y = current[1] * grid_size + grid_size // 2
            path.insert(0, (x, y))
            current = came_from[current]
            if current is None:
                break
        return path
    
    def get_classification_agent_by_type(self, waste_type):
        """Obtener el agente clasificador asociado a un tipo de residuo"""
        for agent in self.environment.central_station.classification_agents:
            if agent.associated_center.waste_type == waste_type:
                return agent
        return None
    
    def get_neighbors(self, grid_pos):
        neighbors = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dir in dirs:
            nx = grid_pos[0] + dir[0]
            ny = grid_pos[1] + dir[1]
            if 0 <= nx * self.environment.grid_size < WIDTH and 0 <= ny * self.environment.grid_size < HEIGHT:
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def distance_grid(self, a, b):
        return 1  # Coste uniforme para movimientos en la cuadrícula

# Agentes de Clasificación
class ClassificationAgent:
    def __init__(self, name, x, y, associated_center, color):
        self.received_waste = []
        self.name = name
        self.color = color
        self.original_color = color  # Guardar el color original
        self.x = x
        self.y = y
        self.associated_center = associated_center 
        
        # Parámetros para el parpadeo
        self.blink = False
        self.blink_color = color
        self.blink_timer = 0
        self.blink_interval = 200  # Intervalo de parpadeo en milisegundos
        self.blink_duration = 1000  # Duración total del parpadeo en milisegundos
        self.last_blink_time = 0

    def receive_waste(self, waste_list):
        self.received_waste.extend(waste_list)
    
    def classify_waste(self):
        if self.received_waste:
            for waste in self.received_waste:
                print(f"{self.name} clasificando residuo {waste.waste_type} de peso {waste.weight} en posición ({waste.x}, {waste.y})")
                self.deposit_waste(waste)
            self.received_waste = []

    def deposit_waste(self, waste):
        """Depositar residuos en el centro asociado"""
        if len(self.associated_center.received_waste) < self.associated_center.capacity:
            self.associated_center.received_waste.append(waste)
            print(f"{self.name} depositó el residuo en el centro {self.associated_center.waste_type}.")
        else:
            print(f"El centro {self.associated_center.waste_type} está lleno y no puede recibir más residuos.")

    def trigger_blink(self, success):
        """Iniciar el parpadeo basado en el éxito de la entrega"""
        self.blink = True
        self.blink_color = GREEN if success else RED
        self.blink_timer = self.blink_duration
        self.last_blink_time = pygame.time.get_ticks()

    def update_blink(self):
        """Actualizar el estado del parpadeo"""
        if self.blink:
            current_time = pygame.time.get_ticks()
            elapsed = current_time - self.last_blink_time

            if elapsed >= self.blink_interval:
                # Alternar el color entre el color de parpadeo y el color original
                if self.blink_color == self.original_color:
                    self.blink_color = GREEN if self.blink_color == GREEN else RED
                else:
                    self.blink_color = self.original_color

                self.last_blink_time = current_time
                self.blink_timer -= self.blink_interval

                if self.blink_timer <= 0:
                    self.blink = False
                    self.blink_color = self.original_color
                    
    def draw(self, window):
        # Actualizar el parpadeo si está activo
        if self.blink:
            self.update_blink()

        # Dibujar el robot con el color actual (puede ser original o de parpadeo)
        current_color = self.blink_color if self.blink else self.original_color

        # Dibujar un robot con figuras geométricas
        pygame.draw.rect(window, current_color, (self.x - 10, self.y - 15, 20, 15))  # Cuerpo
        pygame.draw.circle(window, current_color, (self.x, self.y - 20), 5)  # Cabeza
        pygame.draw.line(window, current_color, (self.x - 5, self.y), (self.x - 10, self.y + 10), 2)  # Pierna izquierda
        pygame.draw.line(window, current_color, (self.x + 5, self.y), (self.x + 10, self.y + 10), 2)  # Pierna derecha
        pygame.draw.line(window, current_color, (self.x - 10, self.y - 10), (self.x - 15, self.y - 5), 2)  # Brazo izquierdo
        pygame.draw.line(window, current_color, (self.x + 10, self.y - 10), (self.x + 15, self.y - 5), 2)  # Brazo derecho


# Inicialización del entorno y agentes
environment = CityEnvironment(num_points=20)

# Agentes de clasificación asociados a los centros
classification_agents = environment.classification_agents

# Crear agentes de recolección
collection_agents = []
for i in range(3):  # Tres agentes de recolección
    agent = CollectionAgent(
        x=random.randint(50, WIDTH - 50),
        y=random.randint(50, HEIGHT - 50),
        environment=environment,
        name=f"CollectionAgent_{i + 1}"
    )
    collection_agents.append(agent)

# Agente de Transporte ubicado en el centro del mapa
transport_agent = TransportAgent(
    x=WIDTH // 2,
    y=HEIGHT // 2,
    environment=environment,
    capacity=4  # Capacidad máxima de residuos
)

# Bucle principal de la simulación
def main():
    run = True
    while run:
        clock.tick(FPS)
        WINDOW.fill(WHITE)
        environment.draw(WINDOW)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Actualización de agentes de recolección
        for agent in collection_agents:
            agent.perceive()
            agent.decide()
            agent.act()
            agent.draw(WINDOW)

        # Actualización del agente de transporte
        transport_agent.perceive()
        transport_agent.act()
        transport_agent.draw(WINDOW)

        # Agentes de clasificación procesan los residuos
        for classification_agent in classification_agents:
            classification_agent.classify_waste()
            classification_agent.draw(WINDOW)

        # Mostrar el número de residuos en cada centro
        font = pygame.font.SysFont('Arial', 18)
        for center, classification_agent in zip(environment.centers, classification_agents):
            text = f"Residuos: {len(center.received_waste)}"
            text_surface = font.render(text, True, BLACK)
            WINDOW.blit(text_surface, (center.x - 40, center.y + 30))
        
        pygame.display.update()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
