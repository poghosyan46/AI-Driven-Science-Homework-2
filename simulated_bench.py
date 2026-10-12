import numpy as np
import matplotlib.pyplot as plt

class OpticalBenchSimulator:
    def __init__(self, dmd_res=32, cam_res=64, noise_level=0.05):
        self.dmd_res = dmd_res
        self.cam_res = cam_res
        self.noise_level = noise_level
        
        # Transmission Matrix T: Complex Gaussian random matrix
        np.random.seed(42)
        n_dmd = dmd_res * dmd_res
        n_cam = cam_res * cam_res
        self.T = (np.random.randn(n_cam, n_dmd) + 1j * np.random.randn(n_cam, n_dmd)) / np.sqrt(2 * n_dmd)
        
        # Focus Region: Central 8x8 pixels
        cx, cy = cam_res // 2, cam_res // 2
        r = 4
        self.focus_mask = np.zeros((cam_res, cam_res), dtype=bool)
        self.focus_mask[cx-r:cx+r, cy-r:cy+r] = True
        
        # Proximal Background Mask (Ring around focus)
        self.bg_mask = np.zeros((cam_res, cam_res), dtype=bool)
        self.bg_mask[cx-2*r:cx+2*r, cy-2*r:cy+2*r] = True
        self.bg_mask[self.focus_mask] = False

    def propagate(self, dmd_pattern: np.ndarray) -> np.ndarray:
        # E_in: DMD binary amplitude modulation
        E_in = dmd_pattern.flatten().astype(np.float64)
        E_out = self.T @ E_in
        I_cam = np.abs(E_out) ** 2
        
        # Add measurement noise
        noise = np.random.normal(0, self.noise_level * np.mean(I_cam), size=I_cam.shape)
        I_cam = np.clip(I_cam + noise, 0, None)
        return I_cam.reshape((self.cam_res, self.cam_res))

    def compute_enhancement(self, I_cam: np.ndarray) -> float:
        I_focus = np.mean(I_cam[self.focus_mask])
        I_bg = np.mean(I_cam[self.bg_mask])
        return float(I_focus / (I_bg + 1e-8))

    def run_focusing_search(self, macropixel_size=4, num_iterations=200):
        grid_dim = self.dmd_res // macropixel_size
        num_macros = grid_dim * grid_dim
        
        # Random initial pattern
        macro_pattern = np.random.choice([0.0, 1.0], size=(grid_dim, grid_dim))
        
        def get_full_pattern(m_pat):
            return np.repeat(np.repeat(m_pat, macropixel_size, axis=0), macropixel_size, axis=1)

        dmd_pattern = get_full_pattern(macro_pattern)
        I_initial = self.propagate(dmd_pattern)
        best_enhancement = self.compute_enhancement(I_initial)
        
        enhancement_history = [best_enhancement]

        for it in range(num_iterations):
            # Pick a random macropixel to flip
            mx = np.random.randint(0, grid_dim)
            my = np.random.randint(0, grid_dim)
            
            macro_pattern[mx, my] = 1.0 - macro_pattern[mx, my]
            candidate_pattern = get_full_pattern(macro_pattern)
            
            I_candidate = self.propagate(candidate_pattern)
            candidate_enhancement = self.compute_enhancement(I_candidate)
            
            if candidate_enhancement > best_enhancement:
                best_enhancement = candidate_enhancement
                dmd_pattern = candidate_pattern
            else:
                # Revert flip
                macro_pattern[mx, my] = 1.0 - macro_pattern[mx, my]
                
            enhancement_history.append(best_enhancement)

        I_final = self.propagate(dmd_pattern)
        return {
            "initial_enhancement": float(enhancement_history[0]),
            "final_enhancement": float(best_enhancement),
            "I_initial": I_initial,
            "I_final": I_final,
            "history": enhancement_history,
            "macropixel_size": macropixel_size,
            "iterations": num_iterations
        }