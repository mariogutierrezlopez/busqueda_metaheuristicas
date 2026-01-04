import numpy as np
import random
import copy

#####################################################
#             DEFINICIÓN DEL PROBLEMA               #
#####################################################

# Tareas 
NUM_TASKS = 4

# Operarios 
NUM_OPERATORS = 5

# Matriz de Costes c_ij (coste de asignar tarea i al operario j)
COSTS = np.array([
    # T1, T2, T3, T4
    [ 2,  3,  4,  1], # O1
    [ 3,  2,  3,  2], # O2
    [ 2,  2,  1,  2], # O3
    [ 3,  3,  3,  3], # O4
    [ 2,  1,  2,  2]  # O5
])

# Tiempo requerido por tarea
REQ_TIMES = np.array([3, 2, 2, 3])

# Tiempo disponible por operario
TIME_AVAILABLE = np.array([4, 5, 3, 4, 4])

#####################################################
#         DEFINICIÓN DE LA CLASE HORMIGA            #
#####################################################

class Ant:
    """Representa una solución (hormiga)."""
    def __init__(self):
        self.assignment = [-1] * NUM_TASKS # assignment[tarea] = operario
        self.operator_usage = np.zeros(NUM_OPERATORS) # Tiempo consumido por cada operario
        self.total_cost = 0.0
        self.is_valid = True 

    def calculate_metrics(self):
        """Calcula los objetivos finales de la solución completa."""
        self.total_cost = 0
        self.operator_usage = np.zeros(NUM_OPERATORS)
        
        for task_idx, op_idx in enumerate(self.assignment):
            if op_idx == -1:
                self.is_valid = False
                return
            self.total_cost += COSTS[op_idx][task_idx]
            self.operator_usage[op_idx] += REQ_TIMES[task_idx]

        if np.any(self.operator_usage > TIME_AVAILABLE):
            self.is_valid = False

    def get_load_imbalance(self):
        """Enfoque Multiobjetivo: Calcula la desviación estándar de los tiempos de uso."""
        return np.std(self.operator_usage)

#####################################################
#       DEFINICIÓN DEL SISTEMA DE HORMIGAS          #
#####################################################

class AntSystem:
    def __init__(self, num_ants=10, alpha=1.0, beta=2.0, rho=0.1, q=100):
        self.num_ants = num_ants
        self.alpha = alpha # Peso de la feromona (Historia)
        self.beta = beta   # Peso de la heurística (Coste inmediato)
        self.rho = rho     # Coeficiente de persistencia (1 - evaporación)
        self.Q = q         # Cantidad de feromona a depositar
        
        # Inicialización de Feromona: tau_ij(0) = constante pequeña
        self.pheromones = np.ones((NUM_OPERATORS, NUM_TASKS)) * 0.1
        
        # Matriz de Visibilidad: eta = 1 / Coste
        self.heuristic = 1.0 / COSTS

    def select_operator(self, task_idx, current_usage):
        """
        Regla de transición probabilística.
        Solo considera operarios que tengan capacidad disponible (Lista Tabú).
        """
        probs = []
        allowed_ops = []

        for op_idx in range(NUM_OPERATORS):
            # Restricción del GAP:
            # ¿Tiene el operario 'op_idx' tiempo suficiente para la tarea 'task_idx'?
            remaining_cap = TIME_AVAILABLE[op_idx] - current_usage[op_idx]
            
            if remaining_cap >= REQ_TIMES[task_idx]:
                # Cálculo del numerador: [tau]^alpha * [eta]^beta
                tau = self.pheromones[op_idx][task_idx]
                eta = self.heuristic[op_idx][task_idx]
                
                prob_value = (tau ** self.alpha) * (eta ** self.beta)
                
                probs.append(prob_value)
                allowed_ops.append(op_idx)
        
        # Si ningún operario puede hacer la tarea, la hormiga ha llegado a un callejón sin salida
        if not allowed_ops:
            return -1

        # Ruleta: Normalizar probabilidades
        total_prob = sum(probs)
        probs = [p / total_prob for p in probs]
        
        # Selección aleatoria ponderada
        return random.choices(allowed_ops, weights=probs, k=1)[0]

    def construct_solution(self):
        """Una hormiga construye una asignación completa tarea por tarea."""
        ant = Ant()
        # Ordenamos las tareas secuencialmente (0, 1, 2, 3)
        tasks_order = list(range(NUM_TASKS)) 
        
        for task_idx in tasks_order:
            op_selected = self.select_operator(task_idx, ant.operator_usage)
            
            if op_selected == -1:
                ant.is_valid = False
                break
            
            ant.assignment[task_idx] = op_selected
            ant.operator_usage[op_selected] += REQ_TIMES[task_idx]
            
        if ant.is_valid:
            ant.calculate_metrics()
            
        return ant

# ==========================================
# 3. ENFOQUE 1: MONO-OBJETIVO (SISTEMA ELITISTA)
# ==========================================
class MonoObjectiveAS(AntSystem):
    """
    Resolución estricta del Problema 1: Minimizar Coste Total.
    Incluye estrategia elitista.
    """
    def run(self, generations=50, elitist_ants=2):
        best_global_solution = None
        
        print(f"--- Iniciando AS Elitista (Objetivo: Coste Mínimo) ---")
        
        for gen in range(generations):
            ants_solutions = []
            
            # 1. Fase de Construcción
            for _ in range(self.num_ants):
                sol = self.construct_solution()
                if sol.is_valid:
                    ants_solutions.append(sol)
                    
                    # Actualizar mejor global si aplica
                    if best_global_solution is None or sol.total_cost < best_global_solution.total_cost:
                        best_global_solution = copy.deepcopy(sol)
            
            # 2. Fase de Evaporación
            # tau_nueva = (1 - rho) * tau_actual
            self.pheromones *= (1 - self.rho) 
            
            # 3. Fase de Depósito (Hormigas de esta iteración)
            for sol in ants_solutions:
                delta_tau = self.Q / sol.total_cost
                for t_idx, op_idx  in enumerate(sol.assignment):
                    self.pheromones[op_idx][t_idx] += delta_tau
            
            # 4. Estrategia Elitista
            # Reforzamos el mejor camino global con feromona extra.
            # Formula: e * (Q / L_best)
            if best_global_solution:
                delta_elite = (elitist_ants * self.Q) / best_global_solution.total_cost
                for t_idx, op_idx in enumerate(best_global_solution.assignment):
                    self.pheromones[op_idx][t_idx] += delta_elite
            
            # Log simple
            if gen % 10 == 0 and best_global_solution:
                 print(f"Gen {gen}: Mejor Coste encontrado = {best_global_solution.total_cost}")

        return best_global_solution

# ==========================================
# 4. ENFOQUE 2: MULTI-OBJETIVO (MOACO)
# ==========================================
class MultiObjectiveAS(AntSystem):
    """
    Coste vs. Equilibrio de Carga.
    """
    def run(self, generations=50):
        pareto_archive = [] # Lista de soluciones no dominadas
        
        print(f"\n--- Iniciando AS Multi-objetivo (Coste vs Equilibrio) ---")
        
        for gen in range(generations):
            current_gen_ants = []
            
            # 1. Construcción
            for _ in range(self.num_ants):
                sol = self.construct_solution()
                if sol.is_valid:
                    current_gen_ants.append(sol)
            
            # 2. Actualización del Archivo de Pareto
            # Mezclamos archivo anterior con nuevas hormigas y filtramos dominados
            combined_pool = pareto_archive + current_gen_ants
            pareto_archive = self.update_pareto(combined_pool)
            
            # 3. Evaporación
            self.pheromones *= (1 - self.rho)
            
            # 4. Actualización Multi-objetivo
            # Solo las hormigas del Frente de Pareto depositan feromona
            for sol in pareto_archive:
                delta_tau = self.Q / sol.total_cost 
                for t_idx, op_idx in enumerate(sol.assignment):
                    self.pheromones[op_idx][t_idx] += delta_tau
            
            if gen % 10 == 0:
                costs = [s.total_cost for s in pareto_archive]
                print(f"Gen {gen}: Soluciones en Pareto = {len(pareto_archive)} | Min Coste en Frente: {min(costs) if costs else 'N/A'}")

        return pareto_archive

    def update_pareto(self, candidates):
        """Filtra y retorna solo las soluciones no dominadas."""
        non_dominated = []
        for sol_a in candidates:
            is_dominated = False
            for sol_b in candidates:
                if sol_a == sol_b: continue
                
                # Objetivos a MINIMIZAR:
                # 1. Coste Total
                # 2. Desequilibrio (StdDev)
                
                cost_a, imb_a = sol_a.total_cost, sol_a.get_load_imbalance()
                cost_b, imb_b = sol_b.total_cost, sol_b.get_load_imbalance()
                
                # B domina a A si es mejor o igual en todo y mejor estrictamente en algo
                if (cost_b <= cost_a and imb_b <= imb_a) and (cost_b < cost_a or imb_b < imb_a):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Evitar añadir soluciones idénticas (misma asignación)
                is_duplicate = False
                for existing in non_dominated:
                    if np.array_equal(sol_a.assignment, existing.assignment):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    non_dominated.append(sol_a)
                    
        return non_dominated

# ==========================================
# 5. EJECUCIÓN
# ==========================================
if __name__ == "__main__":
    
    # --- EJECUCIÓN A: Enfoque estricto del Problema 1 ---
    # Parámetros: 20 hormigas, evaporación 0.1, peso heurística mayor que feromona
    mono_solver = MonoObjectiveAS(num_ants=20, rho=0.1, alpha=1, beta=2)
    best_sol = mono_solver.run(generations=50)
    
    print("\n>>> RESULTADO ENFOQUE MONO-OBJETIVO (Coste Mínimo) <<<")
    if best_sol:
        print(f"Asignación Óptima (Tarea->Operario): {best_sol.assignment}")
        print(f"Coste Total: {best_sol.total_cost}")
        print(f"Tiempo usado por operario: {best_sol.operator_usage}")
        print(f"Capacidades disponibles:   {TIME_AVAILABLE}")
    else:
        print("No se encontró solución válida.")

    # --- EJECUCIÓN B: Enfoque docente Multiobjetivo ---
    multi_solver = MultiObjectiveAS(num_ants=20, rho=0.1, alpha=1, beta=2)
    pareto_front = multi_solver.run(generations=50)
    
    print("\n>>> RESULTADO ENFOQUE MULTI-OBJETIVO (Frente de Pareto) <<<")
    print(f"{'ID':<5} {'Coste':<10} {'Desequilibrio':<15} {'Asignación'}")
    print("-" * 60)
    # Ordenar por coste para visualizar mejor el trade-off
    pareto_front.sort(key=lambda x: x.total_cost)
    
    for i, sol in enumerate(pareto_front):
        print(f"{i:<5} {sol.total_cost:<10} {sol.get_load_imbalance():<15.4f} {sol.assignment}")