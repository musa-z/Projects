import numpy as np
import heapq
import matplotlib.pyplot as plt
import time

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* algorithm implementation (no diagonal movement)
def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    # Possible movements: only 4 cardinal directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        for dx, dy in directions:  # Adjacent cells
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:  # Stay within bounds
                # Ensure we're skipping obstacles
                if grid[neighbor] == 1:  # Skip obstacles
                    continue
                
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

# Visualization function (matplotlib)
def visualize(grid, path, start, goal):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap=plt.cm.binary)

    # Plot start and goal
    ax.scatter(start[1], start[0], c='green', label='Start')
    ax.scatter(goal[1], goal[0], c='red', label='Goal')

    # Plot path
    if path:
        path = np.array(path)
        ax.plot(path[:, 1], path[:, 0], c='blue', label='Path')

    plt.legend()
    plt.title("Dynamic A* Pathfinding")
    plt.show()

# Function to add random dynamic obstacles
def add_dynamic_obstacles(grid, obstacles_to_add):
    for obstacle in obstacles_to_add:
        grid[obstacle[0], obstacle[1]] = 1  # Add obstacle
    return grid

# Main function to run the project with dynamic obstacles
def main():
    # Create a 20x20 grid (0: free space, 1: obstacle)
    grid = np.zeros((20, 20), dtype=int)

    # Add some initial static obstacles
    grid[5, 5:10] = 1
    grid[10, 8:15] = 1
    grid[15, 3:7] = 1

    # Start and goal positions
    start = (0, 0)
    goal = (19, 19)
    
    while True:
        # Step 1: Find the initial path
        path = a_star(grid, start, goal)

        if path is None:
            print("No path found!")
            break
        
        # Step 2: Visualize the grid and the path
        visualize(grid, path, start, goal)

        # Step 3: Introduce new obstacles dynamically
        print("Introducing new obstacles...")
        new_obstacles = [(7, 7), (8, 7), (9, 7), (7, 8), (15, 15), (15, 16)]
        grid = add_dynamic_obstacles(grid, new_obstacles)

        # Pause for a while to simulate real-time updates
        time.sleep(2)

        print("Replanning due to new obstacles...")
        # Step 4: Recalculate path with the new obstacles in place
        path = a_star(grid, start, goal)
        if path is None:
            print("No path found after obstacle update!")
            break
        
        # Step 5: Visualize the new path
        visualize(grid, path, start, goal)

        # Pause and exit after a few iterations
        time.sleep(2)
        break  # You can remove this break to keep simulating

if __name__ == "__main__":
    main()
