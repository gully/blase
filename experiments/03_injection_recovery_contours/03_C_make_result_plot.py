import torch
import numpy as np
import matplotlib.pyplot as plt

state_dict_init = torch.load("emulator_T4100g3p5_prom0p01_HPF.pt")
state_dict_injected = torch.load("./emulator_T4100g3p5_prom0p01_HPF_injection.pt")
dict_of_recovered = torch.load("./emulator_T4100g3p5_prom0p01_HPF_recovery.pt")

# Remove the lines outside the region-of-interest
roi_mask = (state_dict_init["lam_centers"].cpu() > 8656) & (
    state_dict_init["lam_centers"].cpu() < 8767
)
injected = np.exp(state_dict_injected["amplitudes"].cpu())
initialization = np.exp(state_dict_init["amplitudes"].cpu())

changed = injected != initialization

mask = roi_mask & changed
injected = injected[mask]
initialization = initialization[mask]

# Figure 1: The initialization
plt.figure(figsize=(6, 6))
plt.plot(
    np.exp(state_dict_injected["amplitudes"].cpu()[roi_mask]),
    np.exp(state_dict_init["amplitudes"].cpu()[roi_mask]),
    ".",
    alpha=0.2,
)
plt.plot(injected, initialization, ".", alpha=0.5)
plt.plot([1.0e-4, 1.0], [1.0e-4, 1.0], color="k", linestyle="dashed")
plt.yscale("log")
plt.xscale("log")
plt.ylim(5e-4, 1)
plt.xlim(5e-4, 1)
plt.xlabel("Injected Amplitude")
plt.ylabel("Initialized Amplitude")

plt.savefig("03_initialization_injection.png", dpi=300, bbox_inches="tight")


# Figure 2: The recovery
plt.figure(figsize=(6, 6))

n_iterations = 5
recovered_array = np.zeros((len(injected), n_iterations))
for i in range(5):
    recovered_array[:, i] = np.exp(dict_of_recovered[i]["amplitudes"].cpu()[mask])

mean_recovered = np.mean(recovered_array, axis=1)
std_recovered = np.std(recovered_array, axis=1)

plt.plot(
    np.exp(state_dict_injected["amplitudes"].cpu()[roi_mask]),
    np.exp(dict_of_recovered[0]["amplitudes"].cpu()[roi_mask]),
    ".",
    alpha=0.2,
)

plt.errorbar(
    injected,
    np.exp(dict_of_recovered[0]["amplitudes"].cpu()[mask]),
    yerr=std_recovered,
    alpha=0.5,
    color="k",
    linestyle="none",
)
plt.plot(
    injected, np.exp(dict_of_recovered[0]["amplitudes"].cpu()[mask]), ".", alpha=0.5
)

plt.plot([1.0e-4, 1.0], [1.0e-4, 1.0], color="k", linestyle="dashed", alpha=0.5)
plt.yscale("log")
plt.xscale("log")
plt.ylim(5e-4, 1)
plt.xlim(5e-4, 1)
plt.xlabel("Injected Amplitude")
plt.ylabel("Recovered Amplitude")


plt.savefig("03_injection_recovery_results.png", dpi=300, bbox_inches="tight")

