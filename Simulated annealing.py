import random
import matplotlib.pyplot as plt
import numpy as np
import os


class SimulatedAnnealing:
    def __init__(self, temp, cooling_rate, iterations, local_searches, multiplier, bounds, cost_function):
        self.temperature = temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.local_searches = local_searches
        self.multiplier = multiplier
        self.bounds = bounds
        self.cost_function = cost_function
        self.history = []  # Istoric: (x, y, cost, temperature)

    def starting_point(self):
        return (
            random.uniform(self.bounds[0][0], self.bounds[0][1]),
            random.uniform(self.bounds[1][0], self.bounds[1][1])
        )

    def neighbour(self, x, y, multiplier):
        return (
            x + random.uniform(-1, 1) * multiplier,
            y + random.uniform(-1, 1) * multiplier
        )

    def acceptance_probability(self, old_cost, new_cost):
        if new_cost < old_cost:
            return 1.0
        return np.exp((old_cost - new_cost) / self.temperature)

    def optimize(self):
        x, y = self.starting_point()
        current_cost = self.cost_function(x, y)
        self.history.append((x, y, current_cost, self.temperature))

        for _ in range(self.iterations):
            new_x, new_y = self.neighbour(x, y, self.multiplier[0])
            new_cost = self.cost_function(new_x, new_y)
            acc_prob = self.acceptance_probability(current_cost, new_cost)

            if acc_prob > random.random():
                x, y = new_x, new_y
                current_cost = new_cost
                self.history.append((x, y, current_cost, self.temperature))

            for _ in range(self.local_searches):
                local_x, local_y = self.neighbour(x, y, self.multiplier[1])
                local_cost = self.cost_function(local_x, local_y)
                acc_prob = self.acceptance_probability(current_cost, local_cost)

                if acc_prob > random.random():
                    x, y = local_x, local_y
                    current_cost = local_cost
                    self.history.append((x, y, current_cost, self.temperature))

            self.temperature *= self.cooling_rate

        return x, y, current_cost

    def plot(self, title="Optimizare Simulated Annealing"):
        x_vals, y_vals, z_vals, temperatures = zip(*self.history)

        # Grid pentru funcția de cost
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], 400)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], 400)
        x, y = np.meshgrid(x, y)
        z = self.cost_function(x, y)

        # Punctul minim
        min_idx = np.argmin(z_vals)
        min_x, min_y, min_cost = x_vals[min_idx], y_vals[min_idx], z_vals[min_idx]

        fig = plt.figure(figsize=(16, 8))

        # Suprafața funcției și traseul optimizării
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, y, z, cmap='viridis', alpha=0.8)
        ax1.scatter(x_vals, y_vals, z_vals, c='r', s=15, label='Traseul optimizării')
        ax1.scatter(min_x, min_y, min_cost, color='gold', s=200, marker='*', label=f'Cel mai bun punct ({min_cost:.2f})')
        ax1.text(min_x, min_y, min_cost + 50, f"({min_x:.2f}, {min_y:.2f})", color='black')
        ax1.set_title(title)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('Cost')
        ax1.legend()

        # Evoluția costului și temperaturii
        ax2 = fig.add_subplot(122)
        ax2.plot(range(len(z_vals)), z_vals, label='Cost', color='green')
        ax2.scatter(min_idx, min_cost, color='red', label=f'Minim: {min_cost:.2f}')
        ax2.plot(range(len(temperatures)), temperatures, label='Temperatura', color='blue', linestyle='--')
        ax2.set_title("Evoluția costului și temperaturii")
        ax2.set_xlabel("Iterația")
        ax2.set_ylabel("Valori")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def save_results(self, filename="results.csv"):
        """Salvează istoricul optimizării într-un fișier CSV."""
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        with open(filepath, "w") as f:
            f.write("x,y,cost,temperature\n")
            for x, y, cost, temp in self.history:
                f.write(f"{x},{y},{cost},{temp}\n")
        print(f"Istoric salvat în {filepath}")


def himmelblau_function(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def rosenbrock_function(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


if __name__ == '__main__':
    bounds = [(-6, 6), (-6, 6)]

    sa = SimulatedAnnealing(
        temp=500,
        cooling_rate=0.9,
        iterations=300,
        local_searches=10,
        multiplier=[1.0, 0.5],
        bounds=bounds,
        cost_function=himmelblau_function  # Schimbă aici funcția dacă vrei.
    )

    result = sa.optimize()
    print(f"Rezultat final: x = {result[0]}, y = {result[1]}, cost = {result[2]}")
    sa.plot(title="Optimizare Himmelblau")
    sa.save_results()
