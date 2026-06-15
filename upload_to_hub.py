from huggingface_hub import HfApi

api = HfApi()

# Your Hugging Face repository name
repo_id = "tdTeyjus/modernbert-squad-finetuned"

# Local folder path containing your model weights and configs
local_folder = "./model_weights/best_model_checkpoint"

print(f"🚀 Creating repository {repo_id} if it doesn't exist...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

print(f"📁 Uploading all files from '{local_folder}'...")
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
)

print("✅ Entire folder successfully uploaded!")