import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class ConwaysGameOfLifeOnSphere:
    def __init__(self, lat_cells=20, lon_cells=40):
        self.lat_cells = lat_cells
        self.lon_cells = lon_cells
        self.grid = np.random.choice([0, 1], size=(lat_cells, lon_cells), p=[0.7, 0.3])
        
    def get_neighbors(self, lat, lon):
        neighbors = []
        for d_lat in [-1, 0, 1]:
            for d_lon in [-1, 0, 1]:
                if d_lat == 0 and d_lon == 0:
                    continue
                
                new_lat = (lat + d_lat) % self.lat_cells
                new_lon = (lon + d_lon) % self.lon_cells
                neighbors.append(self.grid[new_lat, new_lon])
        
        return sum(neighbors)
    
    def update(self):
        new_grid = self.grid.copy()
        for lat in range(self.lat_cells):
            for lon in range(self.lon_cells):
                neighbors = self.get_neighbors(lat, lon)
                
                if self.grid[lat, lon] == 1:
                    if neighbors < 2 or neighbors > 3:
                        new_grid[lat, lon] = 0
                else:
                    if neighbors == 3:
                        new_grid[lat, lon] = 1
        
        self.grid = new_grid
    
    def get_sphere_coordinates(self):
        phi = np.linspace(0, np.pi, self.lat_cells)
        theta = np.linspace(0, 2*np.pi, self.lon_cells)
        phi, theta = np.meshgrid(phi, theta)
        
        x = np.sin(phi) * np.cos(theta).T
        y = np.sin(phi) * np.sin(theta).T
        z = np.cos(phi).T
        
        return x, y, z

def run_simulation():
    game = ConwaysGameOfLifeOnSphere(lat_cells=20, lon_cells=40)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def animate(frame):
        ax.clear()
        
        x, y, z = game.get_sphere_coordinates()
        
        colors = np.array([[1, 1, 1, 0.3] if cell == 0 else [0.2, 0.4, 0.8, 0.8] 
                          for row in game.grid for cell in row])
        colors = colors.reshape(game.lat_cells, game.lon_cells, 4)
        
        ax.plot_surface(x, y, z, facecolors=colors, shade=True, 
                       linewidth=0.5, edgecolors=[0.7, 0.7, 0.7, 0.2])
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Conway's Game of Life on a Sphere - Generation {frame}")
        
        game.update()
        
        return ax,
    
    anim = FuncAnimation(fig, animate, frames=200, interval=200, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()