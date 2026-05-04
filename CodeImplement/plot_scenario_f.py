
from pathlib import Path
import pandas as pd
from fl_scenario_f_noise_fedavg import (
    apply_paper_style,
    plot_final_mse,
    plot_final_physical_accuracy,
    plot_learning_curves,
    plot_per_ibr_mse,
)

out = Path("scenario_f_noise_fedavg_results")
apply_paper_style()

plot_final_mse(pd.read_csv(out / "scenario_f_noise_summary.csv"), out)
plot_final_physical_accuracy(pd.read_csv(out / "scenario_f_noise_summary.csv"), out)
plot_learning_curves(pd.read_csv(out / "scenario_f_noise_fedavg_history.csv"), out)
plot_per_ibr_mse(pd.read_csv(out / "scenario_f_per_ibr_final_test_mse.csv"), out)

