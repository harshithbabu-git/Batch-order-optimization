import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Function to solve the TSP using the Nearest Neighbor heuristic
def solve_tsp_nearest_neighbor(points):
    num_points = len(points)
    if num_points == 1:
        return [0, 0], 0  # Trivial solution for single point
    if num_points == 2:
        distance = np.linalg.norm(points[0] - points[1])
        return [0, 1, 0], 2 * distance  # Direct route and return
    
    visited = [False] * num_points
    path = [0]  # Start from the first point
    visited[0] = True
    total_distance = 0

    for _ in range(num_points - 1):
        last_point = path[-1]
        min_distance = float("inf")
        next_point = None

        for i in range(num_points):
            if not visited[i]:
                distance = np.linalg.norm(points[last_point] - points[i])
                if distance < min_distance:
                    min_distance = distance
                    next_point = i

        path.append(next_point)
        visited[next_point] = True
        total_distance += min_distance

    total_distance += np.linalg.norm(points[path[-1]] - points[path[0]])
    path.append(path[0])  # Complete the loop

    return path, total_distance


# Example dataset
np.random.seed(42)
data = {
    "Order_ID": [f"Order_{i}" for i in range(1, 21)],
    "Batch": np.random.randint(0, 3, 20),  # 3 batches
    "Item_Location_X": np.random.uniform(0, 100, 20),
    "Item_Location_Y": np.random.uniform(0, 100, 20),
}
df = pd.DataFrame(data)

# Number of batches
num_batches = df["Batch"].nunique()

# Process each batch and solve the TSP using the Nearest Neighbor heuristic
batch_routes = {}
batch_centroids = []

for batch in range(num_batches):
    batch_data = df[df["Batch"] == batch]
    batch_locations = batch_data[["Item_Location_X", "Item_Location_Y"]].values
    tsp_path, tsp_distance = solve_tsp_nearest_neighbor(batch_locations)
    
    batch_routes[batch] = {
        "Route": [batch_data.iloc[i]["Order_ID"] for i in tsp_path[:-1]],
        "Distance": tsp_distance,
    }
    
    centroid_x = batch_data["Item_Location_X"].mean()
    centroid_y = batch_data["Item_Location_Y"].mean()
    batch_centroids.append((centroid_x, centroid_y))

batch_centroids = np.array(batch_centroids)

# Solve TSP for centroids
centroid_path, total_centroid_distance = solve_tsp_nearest_neighbor(batch_centroids)

# Print results
print("\nOrder Batching and Routing Results:")
for batch, info in batch_routes.items():
    print(f"\nBatch {batch + 1}:")
    print(f"  Approximate Route (Order IDs): {info['Route'][:10]}... (truncated)")
    print(f"  Total Distance: {info['Distance']:.2f}")

print("\nOptimal Route Connecting All Batches:")
print(f"  Batch Visit Order: {[c + 1 for c in centroid_path[:-1]]}")
print(f"  Total Distance Connecting Batches: {total_centroid_distance:.2f}")

# Visualization
plt.figure(figsize=(12, 10))
colors = plt.cm.get_cmap("tab10", num_batches)

# Plot each batch and the optimal connection area
for batch in range(num_batches):
    batch_data = df[df["Batch"] == batch]
    x_coords = batch_data["Item_Location_X"]
    y_coords = batch_data["Item_Location_Y"]
    
    # Plot batch points
    plt.scatter(x_coords, y_coords, color=colors(batch), label=f"Batch {batch + 1}")
    
    # Create polygon for batch area
    if len(x_coords) > 2:  # Polygon requires at least 3 points
        polygon = Polygon(np.c_[x_coords, y_coords], alpha=0.2, color=colors(batch))
        plt.gca().add_patch(polygon)

# Overlay the optimal connection path
optimal_centroid_points = batch_centroids[centroid_path]
plt.plot(
    optimal_centroid_points[:, 0],
    optimal_centroid_points[:, 1],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Optimal Batch Connection",
)

# Highlight the centroid areas in a unified color
for centroid in batch_centroids:
    plt.scatter(centroid[0], centroid[1], color="black", s=100, marker="x")

plt.title("Batch Areas and Optimal Route")
plt.xlabel("Item Location X")
plt.ylabel("Item Location Y")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("Batch Order Optimization")