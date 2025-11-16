import huggingface_hub as hf_hub

if __name__ == "__main__":

    model_id = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
    model_path = "models/LLMs/OpenVino_Mistral-7B-Instruct-v0.3-int4-cw-ov"

    hf_hub.snapshot_download(model_id, local_dir=model_path)

