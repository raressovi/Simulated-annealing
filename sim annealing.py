import random
import matplotlib.pyplot as plt
import numpy as np


class SimulatedAnnealing:
    def __init__(self, temp, cooling_rate, iterations, local_searches, multiplier, function_name,
                 lower_bound_x, upper_bound_x,lower_bound_y, upper_bound_y):
        self.temperature_0 = temp
        self.temperature = temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.local_searches = local_searches
        self.multiplier = multiplier
        self.function_name = function_name
        self.lower_bound_x = lower_bound_x
        self.upper_bound_x = upper_bound_x
        self.lower_bound_y = lower_bound_y
        self.upper_bound_y = upper_bound_y
        self.history = []
        self.acceptance_probability_history = []

    def starting_point(self):
        x = random.uniform(self.lower_bound_x, self.upper_bound_x)
        y = random.uniform(self.lower_bound_y, self.upper_bound_y)
        return x, y

    def neighbour(self, x, y, multiplier=1.0):
        if multiplier == 1.0:
            return (x + random.uniform(-1, 1) * self.temperature / self.temperature_0,
                    y + random.uniform(-1, 1) * self.temperature / self.temperature_0)
        return x + random.uniform(-1, 1) * multiplier, y + random.uniform(-1, 1) * multiplier

    def himmelblau_function(self, x, y):
        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def ackley_function(self, x, y):
        return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) -
                np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))) + np.e + 20

    def beale_function(self, x, y):
        return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2)

    def choose_function(self, function_name, x, y):
        match function_name:
            case 'himmelblau':
                return self.himmelblau_function(x, y)
            case 'ackley':
                return self.ackley_function(x, y)
            case 'beale':
                return self.beale_function(x, y)
            case _:
                return self.himmelblau_function(x, y)

    def acceptance_probability(self, old_cost, new_cost):
        np.seterr(divide="ignore")
        if new_cost < old_cost:
            return 1.0
        else:
            prob = 0.5
            try:
                prob = 1 / (1 + np.exp((new_cost - old_cost) / self.temperature))
            except:
                pass
            return prob

    def optimize(self):
        x, y = self.starting_point()
        current_cost = self.choose_function(self.function_name, x, y)
        self.history.append((x, y))

        # while self.temperature > 1:
        for iteration in range(self.iterations):
            new_x, new_y = self.neighbour(x, y, self.multiplier[0])
            new_cost = self.choose_function(self.function_name, new_x, new_y)
            acc_prob = self.acceptance_probability(current_cost, new_cost)
            if acc_prob > random.random():
                x, y = new_x, new_y
                current_cost = new_cost
                self.acceptance_probability_history.append(acc_prob)
                self.history.append((x, y))

            for neighbour in range(self.local_searches):
                new_x, new_y = self.neighbour(x, y, self.multiplier[1])
                new_cost = self.choose_function(self.function_name, new_x, new_y)
                acc_prob = self.acceptance_probability(current_cost, new_cost)
                if acc_prob > random.random():
                    x, y = new_x, new_y
                    current_cost = new_cost
                    self.acceptance_probability_history.append(acc_prob)
                    self.history.append((x, y))

            self.temperature *= self.cooling_rate

        return x, y, current_cost

    def plot(self):
        x = np.linspace(-6, 6, 400)
        y = np.linspace(-6, 6, 400)
        x, y = np.meshgrid(x, y)
        z = self.choose_function(self.function_name, x, y)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

        hx, hy = zip(*self.history)
        hz = [self.choose_function(self.function_name, x, y) for x, y in self.history]
        ax.plot(hx, hy, hz, color='r', marker='.', markersize=5, linestyle='-', linewidth=1)

        final_x, final_y = self.history[-1]
        final_z = self.choose_function(self.function_name, final_x, final_y)
        ax.text2D(0.2, -0.05, f"Optimum: x= {final_x:.4f}, y= {final_y:.4f}, f(x, y)= {final_z:.4f}",
                   transform=ax.transAxes)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(self.acceptance_probability_history, color='r', marker='.', linestyle='-', linewidth=1)
        ax2.set_title('Acceptance Probability Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Acceptance Probability')

        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(hz, color='g', marker='.', linestyle='-', linewidth=1)
        ax3.set_title('Solution History')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Function value')

        plt.tight_layout()

        plt.show()
        plt.close()


if __name__ == '__main__':
    himmelblau_params = {
        'temp': 1000,
        'cooling_rate': 0.97,
        'iterations': 200,
        'local_searches': 25,
        'multiplier': [0.5, 0.05],
        'function_name': 'himmelblau',
        'lower_bound_x': -6,
        'upper_bound_x': 6,
        'lower_bound_y': -6,
        'upper_bound_y': 6
    }
    ackley_params = {
        'temp': 1000,
        'cooling_rate': 0.95,
        'iterations': 200,
        'local_searches': 10,
        'multiplier': [0.8, 0.05],
        'function_name': 'ackley',
        'lower_bound_x': -5,
        'upper_bound_x': 5,
        'lower_bound_y': -5,
        'upper_bound_y': 5
    }
    beale_params = {
        'temp': 1000,
        'cooling_rate': 0.95,
        'iterations': 200,
        'local_searches': 10,
        'multiplier': [0.5, 0.1],
        'function_name': 'beale',
        'lower_bound_x': -4.5,
        'upper_bound_x': 4.5,
        'lower_bound_y': -4.5,
        'upper_bound_y': 4.5
    }
    ini = beale_params
    for i in range(10):
        sa = SimulatedAnnealing(ini['temp'], ini['cooling_rate'], ini['iterations'], ini['local_searches'],
                                ini['multiplier'], ini['function_name'],
                                ini['lower_bound_x'], ini['upper_bound_x'],
                                ini['lower_bound_y'], ini['upper_bound_y'])
        result = sa.optimize()
        print("The local optimum is: ", result)
        sa.plot()
