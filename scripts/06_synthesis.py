from pathlib import Path

import hydra
import joblib
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ttslearn.synthesis.gen import gen_waveform, predict_acoustic, predict_duration
from ttslearn.utils.io import write_wav
from ttslearn.utils.utils import init_seed, load_utt_list


def synthesis(
    device,
    sample_rate,
    labels,
    binary_dict,
    numeric_dict,
    duration_model,
    duration_config,
    duration_in_scaler,
    duration_out_scaler,
    acoustic_model,
    acoustic_config,
    acoustic_in_scaler,
    acoustic_out_scaler,
    post_filter=False,
):
    # Predict durations
    if duration_model is not None:
        durations = predict_duration(
            device,
            labels,
            duration_model,
            duration_config,
            duration_in_scaler,
            duration_out_scaler,
            binary_dict,
            numeric_dict,
        )
        labels.set_durations(durations)

    # Predict acoustic features
    acoustic_features = predict_acoustic(
        device,
        labels,
        acoustic_model,
        acoustic_config,
        acoustic_in_scaler,
        acoustic_out_scaler,
        binary_dict,
        numeric_dict,
    )

    # Waveform generation
    gen_wav = gen_waveform(
        sample_rate,
        acoustic_features,
        acoustic_config.stream_sizes,
        acoustic_config.has_dynamic_features,
        acoustic_config.num_windows,
        post_filter,
    )

    return gen_wav


@hydra.main(version_base=None, config_path="../config/", config_name="sample")
def main(cfg: DictConfig):
    init_seed(cfg.seed)

    if torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    binary_dict, numeric_dict = hts.load_question_set(to_absolute_path(cfg.qst_path))

    # duration model
    duration_cfg = OmegaConf.load(to_absolute_path(cfg.duration.model_yaml))
    duration_model = hydra.utils.instantiate(duration_cfg.model).to(device)
    duration_model.build_model(cfg.duration.checkpoint)
    duration_in_scaler = joblib.load(to_absolute_path(cfg.duration.in_scaler_path))
    duration_out_scaler = joblib.load(to_absolute_path(cfg.duration.out_scaler_path))
    duration_model.eval()

    # acoustic model
    acoustic_cfg = OmegaConf.load(to_absolute_path(cfg.acoustic.model_yaml))
    print(acoustic_cfg)
    acoustic_model = hydra.utils.instantiate(acoustic_cfg.model).to(device)
    acoustic_model.build_model(cfg.acoustic.checkpoint)
    acoustic_in_scaler = joblib.load(to_absolute_path(cfg.acoustic.in_scaler_path))
    acoustic_out_scaler = joblib.load(to_absolute_path(cfg.acoustic.out_scaler_path))
    acoustic_model.eval()

    in_dir = Path(to_absolute_path(cfg.in_dir))
    out_dir = Path(to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    utt_ids = load_utt_list(to_absolute_path(cfg.utt_list))
    lab_paths = [in_dir / f"{utt_id.strip()}.lab" for utt_id in utt_ids]
    if cfg.num_eval_utts is not None and cfg.num_eval_utts > 0:
        lab_paths = lab_paths[: cfg.num_eval_utts]

    # Run synthesis for each utt.
    for lab_path in tqdm(lab_paths):
        labels = hts.load(lab_path).round_()

        wav = synthesis(
            device,
            cfg.sample_rate,
            labels,
            binary_dict,
            numeric_dict,
            duration_model,
            duration_cfg,
            duration_in_scaler,
            duration_out_scaler,
            acoustic_model,
            acoustic_cfg,
            acoustic_in_scaler,
            acoustic_out_scaler,
            cfg.post_filter,
        )
        wav_path = out_dir / f"{lab_path.stem}.wav"
        write_wav(wav_path, cfg.sample_rate, wav)


if __name__ == "__main__":
    main()
