import kagglehub

# Download latest version
path = kagglehub.model_download("rishijain27/idqn-pistonball/pyTorch/default")

print("Path to model files:", path)