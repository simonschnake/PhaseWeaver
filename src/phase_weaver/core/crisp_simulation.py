from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ChannelSet = Literal["low", "high", "both"]

ELECTRONIC_NOISE_V = 1.2e-3
SIMULATION_DT_S = 1e-15
SIMULATION_HALF_WINDOW_S = 2e-12

# Calibrated CRISP channel centres and response from the 2019 simulator.
# The first 120 elements belong to the low-frequency grating set.
CRISP_CHANNEL_FREQUENCIES_HZ = np.array(
    [
        6.8428301e11, 6.86342276e11, 6.89291238e11, 6.93231268e11, 6.97867818e11, 7.03183896e11,
        7.09148197e11, 7.15734438e11, 7.23107481e11, 7.3124052e11, 7.40197234e11, 7.5001583e11,
        7.60740352e11, 7.72411055e11, 7.85016183e11, 7.98657943e11, 8.13405379e11, 8.29377459e11,
        8.46469777e11, 8.65200287e11, 8.8543061e11, 9.07299763e11, 9.31292658e11, 9.57343678e11,
        9.85437362e11, 1.01603519e12, 1.04944212e12, 1.08594355e12, 1.12511418e12, 1.16825162e12,
        1.25452235e12, 1.2594101e12, 1.26511563e12, 1.27198544e12, 1.28040955e12, 1.29006363e12,
        1.30097204e12, 1.312954e12, 1.3265105e12, 1.34139271e12, 1.35760346e12, 1.37547358e12,
        1.39503008e12, 1.41656334e12, 1.43974321e12, 1.46457178e12, 1.49167731e12, 1.5210142e12,
        1.55222847e12, 1.58655331e12, 1.62359096e12, 1.66369362e12, 1.70718601e12, 1.75508152e12,
        1.80652971e12, 1.86252358e12, 1.92362e12, 1.99001009e12, 2.06242973e12, 2.14196419e12,
        2.28119024e12, 2.28903842e12, 2.29950939e12, 2.31261258e12, 2.32772918e12, 2.34515972e12,
        2.36498904e12, 2.38681143e12, 2.41113054e12, 2.43827617e12, 2.46831212e12, 2.50106736e12,
        2.53639935e12, 2.57499338e12, 2.61694409e12, 2.66265777e12, 2.7118997e12, 2.76495993e12,
        2.8218177e12, 2.8843088e12, 2.95157763e12, 3.02570939e12, 3.10515027e12, 3.19080363e12,
        3.2841115e12, 3.38505628e12, 3.49557917e12, 3.61569473e12, 3.74494818e12, 3.87221178e12,
        3.87605079e12, 3.89182771e12, 3.91112792e12, 3.93302151e12, 3.95880137e12, 3.98757241e12,
        4.02090481e12, 4.05868059e12, 4.09887219e12, 4.14427297e12, 4.19590008e12, 4.25159913e12,
        4.31184945e12, 4.37798056e12, 4.45017789e12, 4.52767991e12, 4.61054903e12, 4.70100308e12,
        4.79815306e12, 4.90363011e12, 5.01847576e12, 5.14289228e12, 5.27853835e12, 5.42497233e12,
        5.58394521e12, 5.75566213e12, 5.94270061e12, 6.14675659e12, 6.36808841e12, 6.60398418e12,
        6.8423908e12, 6.86645607e12, 6.89844358e12, 6.93795435e12, 6.98381615e12, 7.03665301e12,
        7.09581595e12, 7.16182658e12, 7.23558808e12, 7.31656935e12, 7.40588391e12, 7.5036662e12,
        7.61017924e12, 7.72725265e12, 7.8531345e12, 7.98952729e12, 8.13636493e12, 8.29743559e12,
        8.46741872e12, 8.65427554e12, 8.85635877e12, 9.07462058e12, 9.31463114e12, 9.57583282e12,
        9.85392517e12, 1.01581001e13, 1.04872716e13, 1.0847004e13, 1.12261054e13, 1.13980414e13,
        1.14448608e13, 1.15007118e13, 1.15668861e13, 1.16436805e13, 1.16745582e13, 1.17301474e13,
        1.18286343e13, 1.19375252e13, 1.20584689e13, 1.21926665e13, 1.23420542e13, 1.25037214e13,
        1.26827481e13, 1.28758007e13, 1.30836435e13, 1.330982e13, 1.35576821e13, 1.38245741e13,
        1.41103234e13, 1.44217346e13, 1.47624195e13, 1.51316322e13, 1.55258557e13, 1.5955205e13,
        1.64196917e13, 1.69285923e13, 1.7478278e13, 1.80802622e13, 1.87316513e13, 1.94415752e13,
        2.05304449e13, 2.06041596e13, 2.06980683e13, 2.0814004e13, 2.09510109e13, 2.11093505e13,
        2.12864874e13, 2.14804199e13, 2.17008944e13, 2.19489705e13, 2.22216055e13, 2.25161992e13,
        2.28301018e13, 2.31801788e13, 2.35588868e13, 2.39691136e13, 2.44111492e13, 2.48897567e13,
        2.540055e13, 2.59618741e13, 2.65693466e13, 2.72227653e13, 2.79406515e13, 2.8722261e13,
        2.9552037e13, 3.04799241e13, 3.14692524e13, 3.2546398e13, 3.36766529e13, 3.4176302e13,
        3.43153008e13, 3.44839431e13, 3.46803032e13, 3.49137251e13, 3.49825754e13, 3.51760204e13,
        3.54682908e13, 3.57958567e13, 3.616103e13, 3.65661507e13, 3.70094847e13, 3.74942425e13,
        3.80275217e13, 3.8605589e13, 3.92357545e13, 3.99218111e13, 4.0658384e13, 4.14533934e13,
        4.23052569e13, 4.32362878e13, 4.42368558e13, 4.53735986e13, 4.65444168e13, 4.7837106e13,
        4.92264177e13, 5.07656062e13, 5.2423232e13, 5.41042876e13, 5.61221152e13, 5.826734e13,
    ],
    dtype=float,
)

CRISP_RESPONSE_V_PER_NC2 = np.array(
    [
        0.022713972, 0.0263574891, 0.0191519673, 0.0268880698, 0.0296823444, 0.0250777417,
        0.0514215877, 0.0365048653, 0.050139953, 0.0492891672, 0.0628404705, 0.0590190272,
        0.0709957402, 0.0988634708, 0.114630994, 0.0123265766, 0.123461331, 0.136320335,
        0.125770119, 0.132215392, 0.136440279, 0.134883965, 0.131848572, 0.164531877,
        0.155915488, 0.192955737, 0.210324614, 0.206284403, 0.121629041, 0.128159124,
        0.0282388474, 0.0324896486, 0.0632553505, 0.077018354, 0.0865472943, 0.115282182,
        0.12075313, 0.158432901, 0.175611721, 0.19146581, 0.257557223, 0.296939075,
        0.314254548, 0.366881036, 0.391964519, 0.394565181, 0.335777982, 0.434468414,
        0.434228563, 0.45540687, 0.628887904, 0.628342309, 0.746633419, 0.766892682,
        0.707245311, 0.648177114, 0.444611893, 0.551623941, 0.726412461, 0.613676906,
        0.0679542586, 0.081680749, 0.132899393, 0.191507017, 0.190696482, 0.200147063,
        0.273266787, 0.251203348, 0.291641304, 0.302340215, 0.290033553, 0.356812708,
        0.380332636, 0.47198827, 0.571053105, 0.576061366, 0.742794124, 0.82807694,
        0.861271616, 0.934415109, 0.988661829, 1.29302949, 1.68370088, 1.60985763,
        1.66388597, 2.18810117, 1.77063413, 2.22234386, 1.79544513, 0.649211052,
        0.112905733, 0.135219749, 0.177552881, 0.233864266, 0.276136407, 0.308371345,
        0.422577865, 0.450832291, 0.407343365, 0.454518985, 0.47419525, 0.514916496,
        0.582809637, 0.685511764, 0.785862983, 0.918192559, 1.08807514, 1.32820958,
        1.63915706, 1.88506672, 2.11140637, 2.56819078, 3.47029067, 4.43909591,
        4.59933654, 5.20503438, 4.63311775, 4.98549453, 6.05139362, 2.00813681,
        0.317399571, 0.480405496, 0.633070146, 0.701504333, 0.878360578, 1.04252778,
        1.33420246, 1.44414774, 1.60301216, 1.95838326, 1.97299122, 2.01505983,
        2.48201305, 3.35207178, 4.50221954, 0.102154423, 5.83456121, 6.21937385,
        6.72302208, 6.93572009, 7.72672143, 9.53759783, 9.70607794, 9.88128478,
        10.2574353, 11.7366859, 9.75250128, 8.49536317, 5.87634647, 0.690542536,
        0.884953809, 1.10143838, 1.3254804, 1.39920693, 0.26474169, 1.49174256,
        1.42851071, 1.61623263, 2.14723132, 2.49236663, 2.41236131, 2.5299353,
        1.95052864, 2.08077797, 2.46425871, 1.86427999, 1.2988356, 1.6180666,
        2.69349944, 5.26651344, 8.6007671, 12.3955411, 15.5537653, 15.7520786,
        15.531857, 16.3010277, 19.5997146, 18.9657798, 19.8173958, 12.8444056,
        1.05663513, 1.73837417, 2.38571066, 3.12509262, 5.16370343, 6.27279615,
        4.55271819, 2.30736012, 1.00261293, 1.28871366, 6.8262223, 15.2316196,
        13.0645368, 15.6890352, 13.9606414, 15.271801, 7.30423998, 18.659383,
        20.1944545, 21.8768013, 20.9667009, 21.5890054, 23.9164749, 24.8425545,
        23.0255779, 17.1761298, 23.3330662, 23.272596, 21.0340416, 0.524993968,
        1.63131206, 3.44781069, 2.29974095, 4.99462286, 2.32302429, 4.967443,
        2.68405786, 8.62683761, 11.5968873, 14.3637056, 11.6110817, 9.31426757,
        10.7818775, 10.2543508, 7.20074507, 10.1373496, 16.6319551, 5.87900394,
        17.8322017, 6.29817365, 0.987433101, 8.83113064, 6.72258056, 7.85860449,
        9.5341625, 25.9286607, 26.655056, 20.5970702, 2.45164563, 1.38458228,
    ],
    dtype=float,
)


@dataclass(frozen=True, slots=True)
class CrispSimulationConfig:
    n_shots: int = 1
    seed: int = 0
    channel_set: ChannelSet = "both"

    def __post_init__(self) -> None:
        if self.n_shots < 1:
            raise ValueError("n_shots must be at least 1")
        if self.channel_set not in {"low", "high", "both"}:
            raise ValueError("channel_set must be 'low', 'high', or 'both'")


@dataclass(slots=True)
class CrispSimulationResult:
    freq_hz: np.ndarray
    ffabs: np.ndarray
    ffabs_std: np.ndarray
    detection_limit: np.ndarray
    ffsq: np.ndarray
    ffsq_std: np.ndarray
    ffsq_detection_limit: np.ndarray


def simulate_crisp_measurement(
    time_s: np.ndarray,
    current_a: np.ndarray,
    charge_c: float,
    config: CrispSimulationConfig = CrispSimulationConfig(),
) -> CrispSimulationResult:
    """Simulate a calibrated CRISP form-factor measurement from a current profile."""
    time_s = np.asarray(time_s, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    if time_s.ndim != 1 or current_a.ndim != 1 or time_s.shape != current_a.shape:
        raise ValueError("time_s and current_a must be equal-length 1D arrays")
    if len(time_s) < 2 or not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(current_a)):
        raise ValueError("current profile must contain at least two finite samples")
    if not np.isfinite(charge_c) or charge_c <= 0.0:
        raise ValueError("charge_c must be finite and positive")

    order = np.argsort(time_s)
    time_s = time_s[order]
    current_a = current_a[order]
    current_sum = float(np.sum(current_a))
    if current_sum <= 0.0:
        raise ValueError("current profile must have positive total current")
    center_s = float(np.sum(time_s * current_a) / current_sum)
    time_centered_s = time_s - center_s

    simulation_time_s = np.arange(
        -SIMULATION_HALF_WINDOW_S,
        SIMULATION_HALF_WINDOW_S,
        SIMULATION_DT_S,
        dtype=float,
    )
    sampled_current_a = np.interp(
        simulation_time_s,
        time_centered_s,
        current_a,
        left=0.0,
        right=0.0,
    )
    area_c = float(np.trapezoid(sampled_current_a, x=simulation_time_s))
    if area_c <= 0.0:
        raise ValueError("current profile has no support in the CRISP simulation window")
    density = sampled_current_a / area_c
    fft_frequency_hz = np.fft.rfftfreq(len(simulation_time_s), SIMULATION_DT_S)
    form_factor = np.fft.rfft(density) * SIMULATION_DT_S

    selector = _channel_selector(config.channel_set)
    frequencies_hz = CRISP_CHANNEL_FREQUENCIES_HZ[selector]
    response = CRISP_RESPONSE_V_PER_NC2[selector]
    ideal_ffabs = np.interp(frequencies_hz, fft_frequency_hz, np.abs(form_factor))

    charge_nc = charge_c * 1e9
    adc_signal_v = ideal_ffabs**2 * charge_nc**2 * response
    noise_std_v = ELECTRONIC_NOISE_V / np.sqrt(config.n_shots)
    adc_total_v = adc_signal_v + np.random.default_rng(config.seed).normal(
        0.0,
        noise_std_v,
        size=len(adc_signal_v),
    )
    ffabs_signed = np.sign(adc_total_v) * np.sqrt(
        np.abs(adc_total_v) / response
    ) / charge_nc
    denominator = np.maximum(np.abs(adc_total_v), np.finfo(float).tiny)
    ffabs_std = 0.5 * np.abs(ffabs_signed) * noise_std_v / denominator
    detection_limit = np.sqrt(noise_std_v / response) / charge_nc
    ffabs = np.abs(ffabs_signed)

    return CrispSimulationResult(
        freq_hz=frequencies_hz,
        ffabs=ffabs,
        ffabs_std=ffabs_std,
        detection_limit=detection_limit,
        ffsq=ffabs**2,
        ffsq_std=2.0 * ffabs * ffabs_std,
        ffsq_detection_limit=detection_limit**2,
    )


def _channel_selector(channel_set: ChannelSet) -> slice:
    if channel_set == "low":
        return slice(0, 120)
    if channel_set == "high":
        return slice(120, None)
    return slice(None)
