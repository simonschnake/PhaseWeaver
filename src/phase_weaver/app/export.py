"""NPZ export seam for application results."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NpzExporter:
    """Serialize plots, reconstruction metadata, state, and measurements."""

    def export_npz(
        self,
        path: str | Path,
        time_model: Any,
        spectrum_model: Any,
        *,
        summary: Any,
        controls_state: Any = None,
        loaded_measurements: tuple[Any, ...] = (),
        crisp_diagnostics_payload: Any = None,
        state_payload: Any = None,
    ) -> None:
        payload = {
            "t": time_model.t_ui if time_model is not None else np.array([]),
            "current_recon": time_model.current_recon_ui if time_model is not None else np.array([]),
            "current_input": time_model.current_input_ui if time_model is not None else np.array([]),
            "f": spectrum_model.f_ui if spectrum_model is not None else np.array([]),
            "mag_recon": spectrum_model.mag_recon_ui if spectrum_model is not None else np.array([]),
            "phase_recon": spectrum_model.phase_recon_ui if spectrum_model is not None else np.array([]),
            "mag_input": spectrum_model.mag_input_ui if spectrum_model is not None else np.array([]),
            "phase_input": spectrum_model.phase_input_ui if spectrum_model is not None else np.array([]),
            "reconstruction_algorithm": np.array(summary.algorithm),
            "measurement_source": np.array(summary.measurement_source),
            "measurement_count": np.array(summary.measurement_count),
            "reconstruction_status": np.array(summary.status),
            "reconstruction_iterations": np.array(summary.iterations),
            "reconstruction_stop_reason": np.array(summary.stop_reason),
            "reconstruction_measurement_error": np.array(
                np.nan if summary.measurement_error is None else summary.measurement_error
            ),
            "reconstruction_ir_relative_constraint_used": np.array(
                summary.ir_relative_constraint_used
            ),
            "reconstruction_relative_measurement_count": np.array(
                summary.relative_measurement_count
            ),
        }
        if summary.history is not None:
            payload.update(summary.history.as_arrays())
        if summary.crisp_diagnostics is not None and crisp_diagnostics_payload is not None:
            payload.update(crisp_diagnostics_payload(summary.crisp_diagnostics))
        if controls_state is not None and state_payload is not None:
            payload.update(state_payload(controls_state))
        for index, item in enumerate(loaded_measurements):
            payload[f"measurement_label_{index}"] = np.array(item.label)
            payload[f"measurement_freq_hz_{index}"] = item.measured.freq
            payload[f"measurement_mag_{index}"] = item.measured.mag
            if item.measured.mag_std is not None:
                payload[f"measurement_mag_std_{index}"] = item.measured.mag_std
            if item.measured.detection_limit is not None:
                payload[f"measurement_detection_limit_{index}"] = item.measured.detection_limit
            payload[f"measurement_kind_{index}"] = np.array(item.kind)
            payload[f"measurement_calibration_{index}"] = np.array(item.calibration)
            payload[f"measurement_use_in_reconstruction_{index}"] = np.array(
                item.use_in_reconstruction
            )
            if item.reference_current is not None:
                reference = item.reference_current
                payload[f"measurement_reference_current_label_{index}"] = np.array(reference.label)
                payload[f"measurement_reference_current_time_s_{index}"] = reference.time_s
                payload[f"measurement_reference_current_a_{index}"] = reference.current_a
                payload[f"measurement_reference_current_max_frequency_thz_{index}"] = np.array(
                    reference.inferred_max_frequency_thz
                )
        np.savez(file=path, **payload)
