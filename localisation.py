import numpy as np
from tdoa import find_delay_between_mp3s
from scipy.optimize import least_squares


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

def expected_tdoa(point, pair, microphones, speed_of_sound):
    dist_i = np.linalg.norm(point - microphones[pair[0]])
    dist_j = np.linalg.norm(point - microphones[pair[1]])
    return (dist_i - dist_j) / speed_of_sound

def residuals(point, measured_tdoas, microphones, speed_of_sound):
    res = []
    for pair in mic_pairs:
        expected = expected_tdoa(point, pair, microphones, speed_of_sound)
        measured = measured_tdoas[pair]
        res.append(expected - measured)
    return res

def coarse_to_fine_search(measured_tdoas, microphones, speed_of_sound, cube_side,
                          coarse_res=0.05, fine_search_radius=0.04):
    lower_bound_volume = np.array([0, 0, -2.5]) 
    upper_bound_volume = np.array([0.1, 0.1, 0.1])  

    x_coarse = np.arange(lower_bound_volume[0], upper_bound_volume[0] + coarse_res/2, coarse_res)
    y_coarse = np.arange(lower_bound_volume[1], upper_bound_volume[1] + coarse_res/2, coarse_res)
    z_coarse = np.arange(lower_bound_volume[2], upper_bound_volume[2] + coarse_res/2, coarse_res)
    grid_points = np.array(np.meshgrid(x_coarse, y_coarse, z_coarse)).T.reshape(-1,3)

    errors = np.zeros(len(grid_points))
    for idx, point in enumerate(grid_points):
        err = residuals(point, measured_tdoas, microphones, speed_of_sound)
        errors[idx] = np.sum(np.square(err))

    best_index = np.argmin(errors)
    coarse_best_point = grid_points[best_index]

    # Set fine bounds around coarse best, clipped to the search volume
    lower_bound = np.maximum(coarse_best_point - fine_search_radius, lower_bound_volume)
    upper_bound = np.minimum(coarse_best_point + fine_search_radius, upper_bound_volume)
    bounds = (lower_bound, upper_bound)

    # Nonlinear least squares refinement
    result = least_squares(residuals, coarse_best_point, bounds=bounds,
                           args=(measured_tdoas, microphones, speed_of_sound), method='trf')

    return result.x * 100  # cm


if __name__ == "__main__":
    '''
        #path to all the mics or some function to get input from each individual mic will come here
        #currently it contains the path to the test audio files
        mics = ["/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals.mp3",
                "/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals.mp3",
                "/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals.mp3",
                "/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals.mp3"]

        input_tdoas = {}

        for i in range(4):
            for j in range(i, 4):
                if i!=j:
                    input_tdoas[(i,j)], _ = find_delay_between_mp3s(mics[i], mics[j], max_tau=5.0, analysis_length=30)


        #once we get the mics and set it up the code below will be removed and only estimated_position_cm will run
        #only dummy variables set here to verify working of functions
        reference_file = "/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals.mp3"
        delayed_file = "/home/azidozide/projects/GCC-PHAT/Test_Audio/B_R-vocals_delayed.mp3"
        delay, _ = find_delay_between_mp3s(
                reference_file, 
                delayed_file, 
                max_tau=5.0,      
                analysis_length=30 
            )
    '''
    input_tdoas = {
        (0,1): 0,
        (0,2): -0.000272,
        (0,3): -0.000272,
        (1,2): -0.000272,
        (1,3): -0.000272,
        (2,3): 0
    }
    
    estimated_position_cm = coarse_to_fine_search(input_tdoas, microphones, speed_of_sound, cube_side)
    print("Estimated source position (cm):", estimated_position_cm)
