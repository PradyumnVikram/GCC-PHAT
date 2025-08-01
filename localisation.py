import numpy as np


speed_of_sound = 343.0  
cube_side_cm = 9
resolution_cm = 1


cube_side = cube_side_cm / 100.0
resolution = resolution_cm / 100.0


microphones = np.array([
    [0, 0, 0],
    [cube_side, cube_side, 0],
    [cube_side, 0, cube_side],
    [0, cube_side, cube_side]
])


mic_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def expected_tdoa(point, pair):
    dist_i = np.linalg.norm(point - microphones[pair[0]])
    dist_j = np.linalg.norm(point - microphones[pair[1]])
    return (dist_i - dist_j) / speed_of_sound

def estimate_3d_coordinates(measured_tdoas):
    x = np.arange(0, cube_side + resolution/2, resolution)
    y = np.arange(0, cube_side + resolution/2, resolution)
    z = np.arange(0, cube_side + resolution/2, resolution)
    grid_points = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)

    errors = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        error_sum = 0.0
        for pair in mic_pairs:
            expected = expected_tdoa(point, pair)
            measured = measured_tdoas[pair]
            error = expected - measured
            error_sum += error*error
        errors[idx] = error_sum

    best_index = np.argmin(errors)
    print(np.sum(errors)/len(errors))
    best_point = grid_points[best_index]
    return best_point * 100  

input_tdoas = {
    (0,1): 0.000123,
    (0,2): 0.000111,
    (0,3): 0.000130,
    (1,2): 0.000020,
    (1,3): 0.000025,
    (2,3): 0.000011
}

estimated_position_cm = estimate_3d_coordinates(input_tdoas)
print("Estimated source position (cm):", estimated_position_cm)
